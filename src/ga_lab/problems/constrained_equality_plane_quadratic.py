from __future__ import annotations

from statistics import mean

from ga_lab.constraints import (
    ConstraintEvaluation,
    evaluate_constraint_violations,
    finite_float_list,
    summarize_constraint_violations,
)
from ga_lab.problems.base import Fitness, Genome, ProblemMetadata


DEFAULT_EQUALITY_PLANE_TOLERANCE = 1e-2


def _default_weights(dimension: int) -> list[float]:
    base = (1.0, 1.2, 1.4, 1.1)
    return [float(base[index % len(base)]) for index in range(dimension)]


def _default_targets(dimension: int) -> list[float]:
    first_pattern = (0.4, 0.3, 0.3, 0.2)
    second_pattern = (0.2, 0.15, 0.1, 0.05)
    split = dimension // 2
    first_half = [float(first_pattern[index % len(first_pattern)]) for index in range(split)]
    second_half = [
        float(second_pattern[index % len(second_pattern)])
        for index in range(dimension - split)
    ]
    return [*first_half, *second_half]


class ConstrainedEqualityPlaneQuadraticProblem:
    name = "constrained_equality_plane_quadratic"
    compatible_representations = ("real",)
    default_objective_directions = (False,)

    def __init__(
        self,
        dimension: int = 6,
        lower_bound: float = -5.0,
        upper_bound: float = 5.0,
        equality_target: float | None = None,
        group_budget: float | None = None,
        equality_tolerance: float = DEFAULT_EQUALITY_PLANE_TOLERANCE,
        weights: list[float] | None = None,
        targets: list[float] | None = None,
    ) -> None:
        if dimension <= 0 or dimension % 2 != 0:
            raise ValueError("dimension must be a positive even integer")
        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be < upper_bound")

        self.dimension = int(dimension)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.bounds = (self.lower_bound, self.upper_bound)
        self.group_split_index = self.dimension // 2
        self.equality_tolerance = float(equality_tolerance)
        self.weights = (
            _default_weights(self.dimension)
            if weights is None
            else finite_float_list(
                weights,
                label="constrained_equality_plane_quadratic_weights",
            )
        )
        self.targets = (
            _default_targets(self.dimension)
            if targets is None
            else finite_float_list(
                targets,
                label="constrained_equality_plane_quadratic_targets",
            )
        )
        if len(self.weights) != self.dimension:
            raise ValueError("weights length must match dimension")
        if len(self.targets) != self.dimension:
            raise ValueError("targets length must match dimension")
        if any(weight <= 0.0 for weight in self.weights):
            raise ValueError("weights must all be > 0")

        self.equality_target = (
            float(equality_target)
            if equality_target is not None
            else float(sum(self.targets[: self.group_split_index]))
        )
        self.group_budget = (
            float(group_budget)
            if group_budget is not None
            else float(sum(self.targets[self.group_split_index :]) + 0.5)
        )

        finite_float_list(
            [
                self.lower_bound,
                self.upper_bound,
                self.equality_target,
                self.group_budget,
                self.equality_tolerance,
            ],
            label="constrained_equality_plane_quadratic_parameters",
        )
        if self.equality_tolerance < 0.0:
            raise ValueError("equality_tolerance must be >= 0")

        self.inequality_count = 1
        self.equality_count = 1
        self.constraint_count = self.inequality_count + self.equality_count

    def _validate_genome(self, genome: Genome) -> list[float]:
        normalized = finite_float_list(
            genome,
            label="constrained_equality_plane_quadratic_genome",
        )
        if len(normalized) != self.dimension:
            raise ValueError(
                "Genome length must match constrained_equality_plane_quadratic dimension"
            )
        return normalized

    def source_bounds(self) -> list[tuple[float, float]]:
        return [self.bounds for _ in range(self.dimension)]

    def default_bounds(self) -> list[tuple[float, float]]:
        return self.source_bounds()

    def constraint_names(self) -> tuple[str, ...]:
        return ("second_half_budget", "plane_sum_target")

    def example_cases(self) -> list[dict[str, object]]:
        tolerance = self.equality_tolerance
        target = self.equality_target
        feasible = list(self.targets)
        boundary = list(self.targets)
        boundary[0] += tolerance
        near_outside = list(self.targets)
        near_outside[0] += tolerance + 1e-6
        infeasible = [2.0 for _ in range(self.dimension)]
        mixed = list(self.targets)
        for index in range(self.group_split_index, self.dimension):
            mixed[index] = self.group_budget

        return [
            {
                "case": "clearly feasible",
                "solution": feasible,
                "expected_feasible": True,
                "expected_violation": "equality value is 0.0 and inequality is <= 0",
            },
            {
                "case": "tolerance boundary",
                "solution": boundary,
                "expected_feasible": True,
                "expected_violation": f"equality residual equals tolerance {tolerance}",
            },
            {
                "case": "near-feasible outside tolerance",
                "solution": near_outside,
                "expected_feasible": False,
                "expected_violation": "equality violation is approximately 1e-6",
            },
            {
                "case": "clearly infeasible",
                "solution": infeasible,
                "expected_feasible": False,
                "expected_violation": f"equality sum differs from target {target} and inequality violates",
            },
            {
                "case": "mixed equality/inequality",
                "solution": mixed,
                "expected_feasible": False,
                "expected_violation": "equality is satisfied but second_half_budget violates",
            },
        ]

    def problem_options(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "equality_target": self.equality_target,
            "group_budget": self.group_budget,
            "equality_tolerance": self.equality_tolerance,
            "weights": list(self.weights),
            "targets": list(self.targets),
        }

    def evaluate_objective(self, genome: Genome) -> float:
        normalized = self._validate_genome(genome)
        return float(
            sum(
                weight * (value - target) ** 2
                for weight, value, target in zip(
                    self.weights,
                    normalized,
                    self.targets,
                    strict=True,
                )
            )
        )

    def evaluate_constraints(self, genome: Genome) -> ConstraintEvaluation:
        normalized = self._validate_genome(genome)
        first_half = normalized[: self.group_split_index]
        second_half = normalized[self.group_split_index :]
        inequality_values = [sum(second_half) - self.group_budget]
        equality_values = [sum(first_half) - self.equality_target]
        return evaluate_constraint_violations(
            inequality_values=inequality_values,
            equality_values=equality_values,
            equality_tolerance=self.equality_tolerance,
            metadata={
                "problem": self.name,
                "dimension": self.dimension,
                "bounds": [self.lower_bound, self.upper_bound],
                "equality_target": self.equality_target,
                "group_budget": self.group_budget,
                "equality_tolerance": self.equality_tolerance,
                "weights": list(self.weights),
                "targets": list(self.targets),
                "constraint_names": list(self.constraint_names()),
                "inequality_constraint_names": ["second_half_budget"],
                "equality_constraint_names": ["plane_sum_target"],
            },
        )

    def evaluate(self, genome: Genome) -> dict[str, object]:
        objective = self.evaluate_objective(genome)
        constraints = self.evaluate_constraints(genome)
        return {
            "objective": objective,
            "constraint_evaluation": constraints.to_dict(),
        }

    def fitness(self, genome: Genome) -> Fitness:
        return self.evaluate_objective(genome)

    def solution_metrics(self, genome: Genome) -> dict[str, object]:
        evaluation = self.evaluate_constraints(genome)
        return {
            "constraint_feasible": evaluation.feasible,
            "constraint_total_violation": evaluation.total_violation,
            "constraint_max_violation": evaluation.max_violation,
            "constraint_violation_count": evaluation.violation_count,
            "constraint_second_half_budget_violation": evaluation.inequality_violations[0],
            "constraint_plane_sum_target_violation": evaluation.equality_violations[0],
        }

    def population_metrics(self, population: list[Genome]) -> dict[str, float]:
        if not population:
            return {}
        evaluations = [self.evaluate_constraints(genome) for genome in population]
        summary = summarize_constraint_violations(evaluations)
        feasible_objectives = [
            self.evaluate_objective(genome)
            for genome, evaluation in zip(population, evaluations, strict=True)
            if evaluation.feasible
        ]
        payload = {
            "population_constraint_feasible_rate": summary.feasible_rate or 0.0,
            "population_constraint_mean_total_violation": summary.mean_total_violation or 0.0,
            "population_constraint_max_total_violation": summary.max_total_violation or 0.0,
            "population_equality_mean_violation": mean(
                evaluation.equality_violations[0] for evaluation in evaluations
            ),
        }
        if feasible_objectives:
            payload["population_best_feasible_objective"] = min(feasible_objectives)
            payload["population_mean_feasible_objective"] = mean(feasible_objectives)
        return payload

    def optimal_fitness(self, genome_length: int) -> float | None:
        if genome_length != self.dimension:
            return None
        return 0.0

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            exact_genome_length=self.dimension,
            default_objective_directions=self.default_objective_directions,
        )


def make_constrained_equality_plane_quadratic_problem(
    **options: object,
) -> ConstrainedEqualityPlaneQuadraticProblem:
    return ConstrainedEqualityPlaneQuadraticProblem(**options)


__all__ = [
    "ConstrainedEqualityPlaneQuadraticProblem",
    "DEFAULT_EQUALITY_PLANE_TOLERANCE",
    "make_constrained_equality_plane_quadratic_problem",
]
