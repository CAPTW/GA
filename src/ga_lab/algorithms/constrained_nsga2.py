from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Sequence

from ga_lab.algorithms._shared import (
    apply_mutation_with_rate,
    maximize_vectors,
    resolve_algorithm_reference_point,
    resolve_objective_directions,
    validate_fitness_vector,
)
from ga_lab.algorithms.nsga2 import (
    _crowding_distance,
    _nondominated_sort,
    _select_by_rank_and_crowding,
)
from ga_lab.constraints import ConstraintEvaluation
from ga_lab.core.crossover import CrossoverFn
from ga_lab.core.mutation import MutationFn
from ga_lab.core.representation import InitFn, Population
from ga_lab.core.selection import SelectionFn, SelectionState
from ga_lab.experiment.constrained_mo_metrics import summarize_constrained_mo_population
from ga_lab.experiment.mo_metrics import nondominated_vectors
from ga_lab.problems.base import as_fitness_vector


@dataclass(slots=True)
class ConstrainedMOCandidateRecord:
    solution: list[float]
    objectives: list[float]
    constraint_evaluation: ConstraintEvaluation
    generation: int
    evaluation_index: int
    stage: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution": list(self.solution),
            "objectives": list(self.objectives),
            "constraint_evaluation": self.constraint_evaluation.to_dict(),
            "generation": self.generation,
            "evaluation_index": self.evaluation_index,
            "stage": self.stage,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ConstrainedNsga2GenerationSummary:
    generation: int
    feasible_rate: float | None
    feasible_nondominated_count: int
    mean_total_violation: float | None
    feasible_only_HV: float | None
    feasible_only_reference_distance: float | None
    actual_evaluations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "feasible_rate": self.feasible_rate,
            "feasible_nondominated_count": self.feasible_nondominated_count,
            "mean_total_violation": self.mean_total_violation,
            "feasible_only_HV": self.feasible_only_HV,
            "feasible_only_reference_distance": self.feasible_only_reference_distance,
            "actual_evaluations": self.actual_evaluations,
        }


@dataclass(slots=True)
class ConstrainedNsga2Result:
    summary: dict[str, Any]
    history: list[dict[str, Any]]
    evaluation_records: list[ConstrainedMOCandidateRecord]
    final_population: list[list[float]]
    final_objective_vectors: list[list[float]]
    final_constraint_evaluations: list[ConstraintEvaluation]


def _constraint_violation_key(evaluation: ConstraintEvaluation) -> tuple[float, float]:
    return (float(evaluation.total_violation), float(evaluation.max_violation))


def _problem_constraint_names(problem, evaluation: ConstraintEvaluation | None = None) -> tuple[list[str], list[str]]:
    constraint_names = getattr(problem, "constraint_names", None)
    if callable(constraint_names):
        names = [str(item) for item in constraint_names()]
    else:
        inequality_count = len(evaluation.inequality_violations) if evaluation else int(
            getattr(problem, "inequality_count", 0)
        )
        equality_count = len(evaluation.equality_violations) if evaluation else int(
            getattr(problem, "equality_count", 0)
        )
        names = [
            *[f"inequality_{index}" for index in range(inequality_count)],
            *[f"equality_{index}" for index in range(equality_count)],
        ]
    inequality_count = len(evaluation.inequality_violations) if evaluation else int(
        getattr(problem, "inequality_count", 0)
    )
    return names[:inequality_count], names[inequality_count:]


def _reference_front(problem) -> list[list[float]] | None:
    reference_front_fn = getattr(problem, "reference_front", None)
    if callable(reference_front_fn):
        payload = reference_front_fn()
        return [[float(value) for value in row] for row in payload]

    reference_front_name = getattr(problem, "reference_front_name", None)
    if isinstance(reference_front_name, str):
        from ga_lab.experiment.mo_metrics import reference_front_for_problem

        return reference_front_for_problem(
            reference_front_name,
            objective_count=int(getattr(problem, "objective_count", 2)),
            variable_count=int(getattr(problem, "dimension", 6)),
        )
    return None


def _evaluate_constrained_individual(
    genome: list[float],
    *,
    config,
    problem,
    generation: int,
    evaluation_index: int,
    location: str,
) -> tuple[list[float], ConstraintEvaluation]:
    objectives = as_fitness_vector(problem.fitness(genome))
    validate_fitness_vector(
        objectives,
        problem_name=str(getattr(problem, "name", config.problem)),
        genome=genome,
        location=location,
        generation=generation,
        evaluation_index=evaluation_index,
    )
    evaluate_constraints = getattr(problem, "evaluate_constraints", None)
    if not callable(evaluate_constraints):
        raise ValueError("constrained NSGA-II requires problem.evaluate_constraints")
    constraint_evaluation = evaluate_constraints(genome)
    if not isinstance(constraint_evaluation, ConstraintEvaluation):
        raise ValueError("problem.evaluate_constraints must return ConstraintEvaluation")
    return [float(value) for value in objectives], constraint_evaluation


def _constrained_dominates(
    left_objectives: Sequence[float],
    left_evaluation: ConstraintEvaluation,
    right_objectives: Sequence[float],
    right_evaluation: ConstraintEvaluation,
    *,
    directions: Sequence[bool],
) -> bool:
    if left_evaluation.feasible and not right_evaluation.feasible:
        return True
    if right_evaluation.feasible and not left_evaluation.feasible:
        return False
    if left_evaluation.feasible and right_evaluation.feasible:
        left_adjusted = maximize_vectors([list(left_objectives)], list(directions))[0]
        right_adjusted = maximize_vectors([list(right_objectives)], list(directions))[0]
        better_or_equal = True
        strictly_better = False
        for left_value, right_value in zip(left_adjusted, right_adjusted, strict=True):
            if left_value < right_value:
                better_or_equal = False
                break
            if left_value > right_value:
                strictly_better = True
        return better_or_equal and strictly_better
    return _constraint_violation_key(left_evaluation) < _constraint_violation_key(right_evaluation)


def _rank_population_constraint_domination(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    *,
    directions: Sequence[bool],
) -> tuple[list[list[int]], list[int], list[float]]:
    size = len(objective_vectors)
    fronts: list[list[int]] = []
    ranks = [-1] * size
    crowding = [0.0] * size

    feasible_indices = [index for index, evaluation in enumerate(evaluations) if evaluation.feasible]
    infeasible_indices = [
        index for index, evaluation in enumerate(evaluations) if not evaluation.feasible
    ]

    rank_cursor = 0
    if feasible_indices:
        feasible_vectors = [list(objective_vectors[index]) for index in feasible_indices]
        feasible_adjusted = maximize_vectors(feasible_vectors, list(directions))
        local_fronts, _ = _nondominated_sort(feasible_adjusted)
        for local_front in local_fronts:
            global_front = [feasible_indices[index] for index in local_front]
            fronts.append(global_front)
            for global_index in global_front:
                ranks[global_index] = rank_cursor
            distances = _crowding_distance(local_front, feasible_adjusted)
            for local_index, value in distances.items():
                crowding[feasible_indices[local_index]] = value
            rank_cursor += 1

    if infeasible_indices:
        grouped: dict[tuple[float, float], list[int]] = {}
        for index in infeasible_indices:
            grouped.setdefault(_constraint_violation_key(evaluations[index]), []).append(index)
        for key in sorted(grouped):
            group = grouped[key]
            fronts.append(group)
            for global_index in group:
                ranks[global_index] = rank_cursor
            group_vectors = [list(objective_vectors[index]) for index in group]
            adjusted_group_vectors = maximize_vectors(group_vectors, list(directions))
            distances = _crowding_distance(list(range(len(group))), adjusted_group_vectors)
            for local_index, value in distances.items():
                crowding[group[local_index]] = value
            rank_cursor += 1

    if not fronts:
        fronts = [list(range(size))]
        ranks = [0] * size
    return fronts, ranks, crowding


def _select_next_population_constrained(
    population: Population,
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    *,
    population_size: int,
    directions: Sequence[bool],
) -> tuple[Population, list[list[float]], list[ConstraintEvaluation], list[int], list[float]]:
    fronts, ranks, crowding = _rank_population_constraint_domination(
        objective_vectors,
        evaluations,
        directions=directions,
    )
    survivor_indices = _select_by_rank_and_crowding(
        population_size,
        fronts,
        crowding,
        ranks,
        population=population,
        objective_vectors=[list(values) for values in objective_vectors],
    )
    return (
        [population[index][:] for index in survivor_indices],
        [list(objective_vectors[index]) for index in survivor_indices],
        [evaluations[index] for index in survivor_indices],
        ranks,
        crowding,
    )


def run_constrained_nsga2(
    config,
    problem,
    selection_fn: SelectionFn,
    crossover_fn: CrossoverFn,
    mutation_fn: MutationFn,
    init_fn: InitFn,
    rng,
) -> ConstrainedNsga2Result:
    population = [init_fn(rng, config.genome_length) for _ in range(config.population_size)]
    requested_budget = config.population_size * ((3 * config.generations) + 2)
    actual_evaluations_used = 0
    evaluation_records: list[ConstrainedMOCandidateRecord] = []
    history: list[dict[str, Any]] = []
    directions: list[bool] | None = None
    reference_point: list[float] | None = None
    reference_front = _reference_front(problem)

    def evaluate_population(
        population_slice: Population,
        *,
        generation: int,
        stage: str,
    ) -> tuple[list[list[float]], list[ConstraintEvaluation]]:
        nonlocal actual_evaluations_used, directions, reference_point
        objective_vectors: list[list[float]] = []
        evaluations: list[ConstraintEvaluation] = []
        for genome in population_slice:
            evaluation_index = actual_evaluations_used
            objectives, constraint_evaluation = _evaluate_constrained_individual(
                genome,
                config=config,
                problem=problem,
                generation=generation,
                evaluation_index=evaluation_index,
                location=stage,
            )
            actual_evaluations_used += 1
            objective_vectors.append(objectives)
            evaluations.append(constraint_evaluation)
            evaluation_records.append(
                ConstrainedMOCandidateRecord(
                    solution=genome[:],
                    objectives=objectives,
                    constraint_evaluation=constraint_evaluation,
                    generation=generation,
                    evaluation_index=evaluation_index,
                    stage=stage,
                    metadata={"seed": config.seed},
                )
            )
        if directions is None and objective_vectors:
            directions = resolve_objective_directions(len(objective_vectors[0]), config, problem)
        if reference_point is None and objective_vectors and directions is not None:
            reference_point, _ = resolve_algorithm_reference_point(
                config,
                problem,
                objective_vectors,
                directions,
            )
        return objective_vectors, evaluations

    for generation in range(config.generations + 1):
        objective_vectors, evaluations = evaluate_population(
            population,
            generation=generation,
            stage="constrained_nsga2_population",
        )
        if directions is None:
            directions = resolve_objective_directions(len(objective_vectors[0]), config, problem)

        inequality_names, equality_names = _problem_constraint_names(problem, evaluations[0])
        generation_summary = summarize_constrained_mo_population(
            objective_vectors,
            evaluations,
            directions=directions,
            reference_point=reference_point,
            reference_front=reference_front,
            inequality_names=inequality_names,
            equality_names=equality_names,
        )
        history.append(
            ConstrainedNsga2GenerationSummary(
                generation=generation,
                feasible_rate=generation_summary["feasible_rate"],
                feasible_nondominated_count=generation_summary["feasible_nondominated_count"],
                mean_total_violation=generation_summary["mean_total_violation"],
                feasible_only_HV=generation_summary["feasible_only_HV"],
                feasible_only_reference_distance=generation_summary[
                    "feasible_only_reference_distance"
                ],
                actual_evaluations=actual_evaluations_used,
            ).to_dict()
        )

        if generation == config.generations:
            break

        fronts, ranks, crowding = _rank_population_constraint_domination(
            objective_vectors,
            evaluations,
            directions=directions,
        )
        selection_state = SelectionState.from_pareto(ranks, crowding)
        offspring: Population = []
        while len(offspring) < config.population_size:
            parent_a = selection_fn(population, selection_state, rng)
            parent_b = selection_fn(population, selection_state, rng)
            if rng.random() < config.crossover_rate:
                child_a, child_b = crossover_fn(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a[:], parent_b[:]
            offspring.append(
                apply_mutation_with_rate(
                    config,
                    mutation_fn,
                    child_a,
                    rng,
                    mutation_rate=config.mutation_rate,
                )
            )
            if len(offspring) < config.population_size:
                offspring.append(
                    apply_mutation_with_rate(
                        config,
                        mutation_fn,
                        child_b,
                        rng,
                        mutation_rate=config.mutation_rate,
                    )
                )

        combined_population = population + offspring
        combined_objectives, combined_evaluations = evaluate_population(
            combined_population,
            generation=generation,
            stage="constrained_nsga2_combined_population",
        )
        population, _, _, _, _ = _select_next_population_constrained(
            combined_population,
            combined_objectives,
            combined_evaluations,
            population_size=config.population_size,
            directions=directions,
        )

    final_objective_vectors, final_constraint_evaluations = evaluate_population(
        population,
        generation=config.generations,
        stage="constrained_nsga2_final_population",
    )
    if directions is None:
        directions = resolve_objective_directions(len(final_objective_vectors[0]), config, problem)
    inequality_names, equality_names = _problem_constraint_names(
        problem,
        final_constraint_evaluations[0],
    )
    final_summary = summarize_constrained_mo_population(
        final_objective_vectors,
        final_constraint_evaluations,
        directions=directions,
        reference_point=reference_point,
        reference_front=reference_front,
        inequality_names=inequality_names,
        equality_names=equality_names,
    )

    feasible_records = [
        record
        for record in evaluation_records
        if record.constraint_evaluation.feasible
    ]
    best_record = min(
        evaluation_records,
        key=lambda record: (
            not record.constraint_evaluation.feasible,
            record.constraint_evaluation.total_violation,
            record.constraint_evaluation.max_violation,
            tuple(record.objectives),
        ),
    )

    summary = {
        "is_nsga2": True,
        "is_constrained_nsga2": True,
        "constraint_policy": "constraint_domination",
        "requested_budget": requested_budget,
        "configured_evaluation_budget": requested_budget,
        "actual_evaluations_used": actual_evaluations_used,
        "objective_directions": [bool(value) for value in directions],
        "feasible_rate": final_summary["feasible_rate"],
        "feasible_count": final_summary["feasible_count"],
        "infeasible_count": final_summary["infeasible_count"],
        "mean_total_violation": final_summary["mean_total_violation"],
        "median_total_violation": final_summary["median_total_violation"],
        "min_total_violation": final_summary["min_total_violation"],
        "max_total_violation": final_summary["max_total_violation"],
        "mean_max_violation": final_summary["mean_max_violation"],
        "violation_count_total": final_summary["violation_count_total"],
        "best_total_violation": final_summary["best_total_violation"],
        "constraint_summary": final_summary["constraint_summary"],
        "per_constraint_violation_summary": final_summary["per_constraint_violation_summary"],
        "feasible_nondominated_count": final_summary["feasible_nondominated_count"],
        "infeasible_nondominated_count": final_summary["infeasible_nondominated_count"],
        "feasible_only_HV": final_summary["feasible_only_HV"],
        "feasible_only_reference_distance": final_summary["feasible_only_reference_distance"],
        "feasible_only_IGD": final_summary["feasible_only_IGD"],
        "spacing_feasible_only": final_summary["spacing_feasible_only"],
        "equality_satisfaction_rate": final_summary["equality_satisfaction_rate"],
        "equality_mean_violation": final_summary["equality_mean_violation"],
        "equality_max_violation": final_summary["equality_max_violation"],
        "pareto_front_vectors": nondominated_vectors(
            [record.objectives for record in feasible_records],
            directions,
        )
        if feasible_records
        else [],
        "final_population": [genome[:] for genome in population],
        "final_objective_vectors": [list(values) for values in final_objective_vectors],
        "final_constraint_evaluations": [
            evaluation.to_dict() for evaluation in final_constraint_evaluations
        ],
        "best_record": best_record.to_dict(),
        "constraint_evaluation_failures": 0,
        "non_finite_constraint_fail_fast_policy": "value_error",
        "feasible_only_metric_policy": "null_when_no_feasible_front",
        "default_changed": False,
        "nsga2_default_changed": False,
        "nsga2_constraint_domination_done": True,
    }
    if feasible_records:
        summary["mean_feasible_objective_0"] = mean(
            record.objectives[0] for record in feasible_records
        )
    return ConstrainedNsga2Result(
        summary=summary,
        history=history,
        evaluation_records=evaluation_records,
        final_population=[genome[:] for genome in population],
        final_objective_vectors=[list(values) for values in final_objective_vectors],
        final_constraint_evaluations=final_constraint_evaluations,
    )


__all__ = [
    "ConstrainedMOCandidateRecord",
    "ConstrainedNsga2GenerationSummary",
    "ConstrainedNsga2Result",
    "_constrained_dominates",
    "_evaluate_constrained_individual",
    "_rank_population_constraint_domination",
    "_select_next_population_constrained",
    "run_constrained_nsga2",
]
