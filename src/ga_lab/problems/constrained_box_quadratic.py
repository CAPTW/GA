from __future__ import annotations

from statistics import mean

from ga_lab.constraints import (
    DEFAULT_EQUALITY_TOLERANCE,
    ConstraintEvaluation,
    evaluate_constraint_violations,
    finite_float_list,
    summarize_constraint_violations,
)
from ga_lab.problems.base import Fitness, Genome, ProblemMetadata


def _default_weights(dimension: int) -> list[float]:
    base = (1.0, 1.25, 1.5)
    return [float(base[index % len(base)]) for index in range(dimension)]


def _default_targets(dimension: int) -> list[float]:
    half = dimension // 2
    first_half = [1.2 - 0.15 * index for index in range(half)]
    second_half = [0.9 - 0.1 * index for index in range(dimension - half)]
    return [float(value) for value in [*first_half, *second_half]]


class ConstrainedBoxQuadraticProblem:
    name = "constrained_box_quadratic"
    compatible_representations = ("real",)
    default_objective_directions = (False,)

    def __init__(
        self,
        dimension: int = 6,
        lower_bound: float = -5.0,
        upper_bound: float = 5.0,
        group1_budget: float | None = None,
        group2_budget: float | None = None,
        equality_tolerance: float = DEFAULT_EQUALITY_TOLERANCE,
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
        self.group1_budget = (
            float(group1_budget)
            if group1_budget is not None
            else float(2.0 * (self.group_split_index / 3.0))
        )
        self.group2_budget = (
            float(group2_budget)
            if group2_budget is not None
            else float(1.5 * ((self.dimension - self.group_split_index) / 3.0))
        )
        self.equality_tolerance = float(equality_tolerance)
        self.weights = (
            _default_weights(self.dimension) if weights is None else finite_float_list(weights, label="constrained_box_quadratic_weights")
        )
        self.targets = (
            _default_targets(self.dimension) if targets is None else finite_float_list(targets, label="constrained_box_quadratic_targets")
        )
        if len(self.weights) != self.dimension:
            raise ValueError("weights length must match dimension")
        if len(self.targets) != self.dimension:
            raise ValueError("targets length must match dimension")
        if self.equality_tolerance < 0.0:
            raise ValueError("equality_tolerance must be >= 0")

        finite_float_list(
            [
                self.lower_bound,
                self.upper_bound,
                self.group1_budget,
                self.group2_budget,
                self.equality_tolerance,
            ],
            label="constrained_box_quadratic_parameters",
        )
        if any(weight <= 0.0 for weight in self.weights):
            raise ValueError("weights must all be > 0")

        self.inequality_count = 2
        self.equality_count = 0
        self.constraint_count = self.inequality_count + self.equality_count

    def _validate_genome(self, genome: Genome) -> list[float]:
        normalized = finite_float_list(genome, label="constrained_box_quadratic_genome")
        if len(normalized) != self.dimension:
            raise ValueError("Genome length must match constrained_box_quadratic dimension")
        return normalized

    def source_bounds(self) -> list[tuple[float, float]]:
        return [self.bounds for _ in range(self.dimension)]

    def default_bounds(self) -> list[tuple[float, float]]:
        return self.source_bounds()

    def constraint_names(self) -> tuple[str, ...]:
        return ("group1_budget", "group2_budget")

    def example_cases(self) -> list[dict[str, object]]:
        return [
            {
                "case": "clearly feasible",
                "solution": [0.4, 0.4, 0.4, 0.3, 0.3, 0.3],
                "expected_feasible": True,
                "expected_violation": "all violations are 0.0",
            },
            {
                "case": "clearly infeasible",
                "solution": [1.5, 1.5, 1.5, 1.4, 1.4, 1.4],
                "expected_feasible": False,
                "expected_violation": "both group budgets violate by large positive amounts",
            },
            {
                "case": "boundary case",
                "solution": [0.7, 0.7, 0.6, 0.5, 0.5, 0.5],
                "expected_feasible": True,
                "expected_violation": "both constraints are exactly on the boundary and report 0.0 violation",
            },
            {
                "case": "multi-constraint violation case",
                "solution": [1.0, 0.8, 0.7, 0.8, 0.5, 0.5],
                "expected_feasible": False,
                "expected_violation": "both group constraints violate and total violation is their sum",
            },
        ]

    def problem_options(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "group1_budget": self.group1_budget,
            "group2_budget": self.group2_budget,
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
        inequality_values = [
            sum(first_half) - self.group1_budget,
            sum(second_half) - self.group2_budget,
        ]
        return evaluate_constraint_violations(
            inequality_values=inequality_values,
            equality_values=[],
            equality_tolerance=self.equality_tolerance,
            metadata={
                "problem": self.name,
                "dimension": self.dimension,
                "bounds": [self.lower_bound, self.upper_bound],
                "group1_budget": self.group1_budget,
                "group2_budget": self.group2_budget,
                "weights": list(self.weights),
                "targets": list(self.targets),
                "constraint_names": list(self.constraint_names()),
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
            "constraint_group1_violation": evaluation.inequality_violations[0],
            "constraint_group2_violation": evaluation.inequality_violations[1],
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
        }
        if feasible_objectives:
            payload["population_best_feasible_objective"] = min(feasible_objectives)
            payload["population_mean_feasible_objective"] = mean(feasible_objectives)
        return payload

    def optimal_fitness(self, genome_length: int) -> float | None:
        if genome_length != self.dimension:
            return None
        return None

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            exact_genome_length=self.dimension,
            default_objective_directions=self.default_objective_directions,
        )
