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
            "Non-finite objective detected in constrained_dtlz_box_toy "
            f"(objective_index={index}, value={value!r})"
        )
    return normalized


class ConstrainedDTLZBoxToyProblem:
    name = "constrained_dtlz_box_toy"
    compatible_representations = ("real",)
    default_objective_directions = (False, False)
    reference_front_name = "dtlz2"

    def __init__(
        self,
        dimension: int = 7,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        first_pair_budget: float = 1.0,
        tail_mean_budget: float = 0.55,
        equality_tolerance: float = DEFAULT_EQUALITY_TOLERANCE,
    ) -> None:
        if dimension < 4:
            raise ValueError("dimension must be >= 4")
        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be < upper_bound")

        self.dimension = int(dimension)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.bounds = (self.lower_bound, self.upper_bound)
        self.first_pair_budget = float(first_pair_budget)
        self.tail_mean_budget = float(tail_mean_budget)
        self.equality_tolerance = float(equality_tolerance)
        self.inequality_count = 2
        self.equality_count = 0
        self.constraint_count = 2
        self.objective_count = 2
        self.objective_direction = ["minimize", "minimize"]
        self.objective_family = "DTLZ2-like 2-objective minimization toy"
        self.group_split_index = self.dimension // 2

        finite_float_list(
            [
                self.lower_bound,
                self.upper_bound,
                self.first_pair_budget,
                self.tail_mean_budget,
                self.equality_tolerance,
            ],
            label="constrained_dtlz_box_toy_parameters",
        )
        if self.equality_tolerance < 0.0:
            raise ValueError("equality_tolerance must be >= 0")

    def _validate_genome(self, genome: Genome) -> list[float]:
        normalized = finite_float_list(genome, label="constrained_dtlz_box_toy_genome")
        if len(normalized) != self.dimension:
            raise ValueError("Genome length must match constrained_dtlz_box_toy dimension")
        return normalized

    def source_bounds(self) -> list[tuple[float, float]]:
        return [self.bounds for _ in range(self.dimension)]

    def default_bounds(self) -> list[tuple[float, float]]:
        return self.source_bounds()

    def constraint_names(self) -> tuple[str, ...]:
        return ("x0_x1_budget", "tail_mean_budget")

    def example_cases(self) -> list[dict[str, object]]:
        return [
            {
                "case": "clearly feasible",
                "solution": [0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2],
                "expected_feasible": True,
                "expected_violation": "both inequality violations are 0.0",
            },
            {
                "case": "clearly infeasible",
                "solution": [0.8, 0.5, 0.2, 0.7, 0.7, 0.7, 0.7],
                "expected_feasible": False,
                "expected_violation": "both constraints violate by positive amounts",
            },
            {
                "case": "boundary case",
                "solution": [0.6, 0.4, 0.1, 0.55, 0.55, 0.55, 0.55],
                "expected_feasible": True,
                "expected_violation": "both constraints are on the boundary and report 0.0 violation",
            },
            {
                "case": "both constraints violated",
                "solution": [0.8, 0.5, 0.2, 0.7, 0.7, 0.7, 0.7],
                "expected_feasible": False,
                "expected_violation": "x0_x1_budget and tail_mean_budget both violate",
            },
        ]

    def problem_options(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "first_pair_budget": self.first_pair_budget,
            "tail_mean_budget": self.tail_mean_budget,
            "equality_tolerance": self.equality_tolerance,
        }

    def evaluate_objectives(self, genome: Genome) -> list[float]:
        normalized = self._validate_genome(genome)
        g = sum((float(value) - 0.5) ** 2 for value in normalized[1:])
        base = 1.0 + g
        angle = (math.pi / 2.0) * normalized[0]
        f1 = base * math.cos(angle)
        f2 = base * math.sin(angle)
        return _validate_objective_vector([f1, f2])

    def evaluate_constraints(self, genome: Genome) -> ConstraintEvaluation:
        normalized = self._validate_genome(genome)
        tail = normalized[self.group_split_index :]
        inequality_values = [
            normalized[0] + normalized[1] - self.first_pair_budget,
            (sum(tail) / len(tail)) - self.tail_mean_budget,
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
                "tail_mean_budget": self.tail_mean_budget,
                "constraint_names": list(self.constraint_names()),
                "reference_front_name": self.reference_front_name,
                "objective_family": self.objective_family,
                "toy_benchmark": True,
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
            raise ValueError(
                "directions length must match constrained_dtlz_box_toy objective count"
            )
        reference: list[float] = []
        for maximize in directions:
            reference.append(-0.1 if maximize else 1.1)
        return reference

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            exact_genome_length=self.dimension,
            default_objective_directions=self.default_objective_directions,
        )


def make_constrained_dtlz_box_toy_problem(**kwargs: object) -> ConstrainedDTLZBoxToyProblem:
    return ConstrainedDTLZBoxToyProblem(**kwargs)


__all__ = [
    "ConstrainedDTLZBoxToyProblem",
    "make_constrained_dtlz_box_toy_problem",
]
