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


class ConstrainedSphereProblem:
    name = "constrained_sphere"
    compatible_representations = ("real",)
    default_objective_directions = (False,)

    def __init__(
        self,
        dimension: int = 4,
        lower_bound: float = -5.0,
        upper_bound: float = 5.0,
        budget: float = 1.0,
        equality_target: float | None = None,
        equality_tolerance: float = DEFAULT_EQUALITY_TOLERANCE,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be < upper_bound")

        self.dimension = int(dimension)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.bounds = (self.lower_bound, self.upper_bound)
        self.budget = float(budget)
        self.equality_target = (
            None if equality_target is None else float(equality_target)
        )
        self.equality_tolerance = float(equality_tolerance)

        finite_float_list(
            [
                self.lower_bound,
                self.upper_bound,
                self.budget,
                self.equality_tolerance,
            ],
            label="constrained_sphere_parameters",
        )
        if self.equality_tolerance < 0.0:
            raise ValueError("equality_tolerance must be >= 0")
        if self.equality_target is not None:
            finite_float_list(
                [self.equality_target],
                label="constrained_sphere_equality_target",
            )

    def _validate_genome(self, genome: Genome) -> list[float]:
        normalized = finite_float_list(genome, label="constrained_sphere_genome")
        if len(normalized) != self.dimension:
            raise ValueError("Genome length must match constrained sphere dimension")
        return normalized

    def source_bounds(self) -> list[tuple[float, float]]:
        return [self.bounds for _ in range(self.dimension)]

    def evaluate_objective(self, genome: Genome) -> float:
        normalized = self._validate_genome(genome)
        return float(sum(value * value for value in normalized))

    def evaluate_constraints(self, genome: Genome) -> ConstraintEvaluation:
        normalized = self._validate_genome(genome)
        inequality_values = [sum(normalized) - self.budget]
        equality_values = (
            [] if self.equality_target is None else [normalized[0] - self.equality_target]
        )
        return evaluate_constraint_violations(
            inequality_values=inequality_values,
            equality_values=equality_values,
            equality_tolerance=self.equality_tolerance,
            metadata={
                "problem": self.name,
                "dimension": self.dimension,
                "budget": self.budget,
                "has_equality_constraint": self.equality_target is not None,
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
        return 0.0

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            exact_genome_length=self.dimension,
            default_objective_directions=self.default_objective_directions,
        )
