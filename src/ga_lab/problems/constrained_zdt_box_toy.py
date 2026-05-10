from __future__ import annotations

import math

from ga_lab.constraints import (
    DEFAULT_EQUALITY_TOLERANCE,
    ConstraintEvaluation,
    evaluate_constraint_violations,
    finite_float_list,
)
from ga_lab.problems.base import Fitness, Genome, ProblemMetadata


def _validate_objective_vector(values: list[float]) -> list[float]:
    normalized = [float(value) for value in values]
    for index, value in enumerate(normalized):
        if math.isfinite(value):
            continue
        raise ValueError(
            "Non-finite objective detected in constrained_zdt_box_toy "
            f"(objective_index={index}, value={value!r})"
        )
    return normalized


class ConstrainedZDTBoxToyProblem:
    name = "constrained_zdt_box_toy"
    compatible_representations = ("real",)
    default_objective_directions = (False, False)
    reference_front_name = "zdt1"

    def __init__(
        self,
        dimension: int = 6,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        first_pair_budget: float = 1.0,
        second_half_budget: float = 1.2,
        equality_tolerance: float = DEFAULT_EQUALITY_TOLERANCE,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be < upper_bound")

        self.dimension = int(dimension)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.bounds = (self.lower_bound, self.upper_bound)
        self.first_pair_budget = float(first_pair_budget)
        self.second_half_budget = float(second_half_budget)
        self.equality_tolerance = float(equality_tolerance)
        self.inequality_count = 2
        self.equality_count = 0
        self.constraint_count = 2
        self.objective_count = 2
        self.objective_direction = ["minimize", "minimize"]
        self.group_split_index = self.dimension // 2

        finite_float_list(
            [
                self.lower_bound,
                self.upper_bound,
                self.first_pair_budget,
                self.second_half_budget,
                self.equality_tolerance,
            ],
            label="constrained_zdt_box_toy_parameters",
        )
        if self.equality_tolerance < 0.0:
            raise ValueError("equality_tolerance must be >= 0")

    def _validate_genome(self, genome: Genome) -> list[float]:
        normalized = finite_float_list(genome, label="constrained_zdt_box_toy_genome")
        if len(normalized) != self.dimension:
            raise ValueError("Genome length must match constrained_zdt_box_toy dimension")
        return normalized

    def source_bounds(self) -> list[tuple[float, float]]:
        return [self.bounds for _ in range(self.dimension)]

    def default_bounds(self) -> list[tuple[float, float]]:
        return self.source_bounds()

    def constraint_names(self) -> tuple[str, ...]:
        return ("x0_x1_budget", "second_half_budget")

    def example_cases(self) -> list[dict[str, object]]:
        boundary_second_half_last = self.second_half_budget - 0.4 - 0.4
        return [
            {
                "case": "clearly feasible",
                "solution": [0.2, 0.3, 0.0, 0.2, 0.2, 0.2],
                "expected_feasible": True,
                "expected_violation": "both inequality violations are 0.0",
            },
            {
                "case": "clearly infeasible",
                "solution": [0.8, 0.5, 0.6, 0.6, 0.6, 0.6],
                "expected_feasible": False,
                "expected_violation": "both constraints violate by positive amounts",
            },
            {
                "case": "boundary case",
                "solution": [0.6, 0.4, 0.1, 0.4, 0.4, boundary_second_half_last],
                "expected_feasible": True,
                "expected_violation": "both constraints are on the boundary and report 0.0 violation",
            },
            {
                "case": "both constraints violated",
                "solution": [0.9, 0.6, 0.5, 0.5, 0.5, 0.5],
                "expected_feasible": False,
                "expected_violation": "x0_x1_budget and second_half_budget both violate",
            },
        ]

    def problem_options(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "first_pair_budget": self.first_pair_budget,
            "second_half_budget": self.second_half_budget,
            "equality_tolerance": self.equality_tolerance,
        }

    def evaluate_objectives(self, genome: Genome) -> list[float]:
        normalized = self._validate_genome(genome)
        f1 = float(normalized[0])
        tail = normalized[1:]
        mean_tail = sum(tail) / len(tail) if tail else 0.0
        g = 1.0 + 9.0 * mean_tail
        ratio = max(0.0, f1 / g) if g > 0.0 else 0.0
        f2 = g * (1.0 - math.sqrt(ratio))
        return _validate_objective_vector([f1, f2])

    def evaluate_constraints(self, genome: Genome) -> ConstraintEvaluation:
        normalized = self._validate_genome(genome)
        tail = normalized[self.group_split_index :]
        inequality_values = [
            normalized[0] + normalized[1] - self.first_pair_budget,
            sum(tail) - self.second_half_budget,
        ]
        return evaluate_constraint_violations(
            inequality_values=inequality_values,
            equality_values=[],
            equality_tolerance=self.equality_tolerance,
            metadata={
                "problem": self.name,
                "dimension": self.dimension,
                "bounds": [self.lower_bound, self.upper_bound],
                "objective_count": self.objective_count,
                "objective_direction": list(self.objective_direction),
                "first_pair_budget": self.first_pair_budget,
                "second_half_budget": self.second_half_budget,
                "constraint_names": list(self.constraint_names()),
                "reference_front_name": self.reference_front_name,
            },
        )

    def evaluate(self, genome: Genome) -> dict[str, object]:
        return {
            "objectives": self.evaluate_objectives(genome),
            "constraint_evaluation": self.evaluate_constraints(genome).to_dict(),
        }

    def fitness(self, genome: Genome) -> Fitness:
        return self.evaluate_objectives(genome)

    def hypervolume_reference_point(self, directions: list[bool]) -> list[float]:
        if len(directions) != 2:
            raise ValueError("directions length must match constrained_zdt_box_toy objective count")
        bounds = [(0.0, 1.0), (0.0, 10.0)]
        reference: list[float] = []
        for (low, high), maximize in zip(bounds, directions, strict=True):
            margin = (high - low) * 0.1
            reference.append(low - margin if maximize else high + margin)
        return reference

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            exact_genome_length=self.dimension,
            default_objective_directions=self.default_objective_directions,
        )


def make_constrained_zdt_box_toy_problem(**kwargs: object) -> ConstrainedZDTBoxToyProblem:
    return ConstrainedZDTBoxToyProblem(**kwargs)


__all__ = [
    "ConstrainedZDTBoxToyProblem",
    "make_constrained_zdt_box_toy_problem",
]
