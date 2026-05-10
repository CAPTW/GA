from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from ga_lab.algorithms._shared import (
    apply_mutation_with_rate,
    problem_population_metrics,
    problem_solution_metrics,
    validate_fitness_vector,
)
from ga_lab.config import GAConfig
from ga_lab.constraints import (
    ConstraintEvaluation,
    finite_float_list,
    summarize_constraint_violations,
    validate_constraint_values,
)
from ga_lab.core.crossover import CrossoverFn
from ga_lab.core.mutation import MutationFn
from ga_lab.core.representation import Genome, InitFn, Population
from ga_lab.core.selection import SelectionFn, SelectionState
from ga_lab.experiment.constrained_metrics import (
    best_feasible_objective,
    summarize_constrained_population,
)
from ga_lab.experiment.constrained_protocol import (
    ConstrainedCandidateRecord,
    rank_constrained_candidates,
    select_best_feasible_first,
)
from ga_lab.experiment.constrained_trace import (
    build_per_constraint_trace_records,
    per_constraint_summary_to_dict,
    summarize_per_constraint_trace,
)
from ga_lab.problems.base import as_fitness_vector


SUPPORTED_CONSTRAINT_POLICY = "feasibility_first"
SUPPORTED_VIOLATION_AGGREGATION = "total_violation"


@dataclass(slots=True)
class ConstrainedGenerationSummary:
    generation: int
    population_size: int
    actual_evaluations: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    feasible_rate: float | None
    feasible_count: int
    infeasible_count: int
    best_feasible_objective: float | None
    best_record_feasible: bool
    best_record_total_violation: float
    mean_total_violation: float | None
    min_total_violation: float | None
    max_total_violation: float | None
    constraint_summary: dict[str, Any]
    best_record: dict[str, Any]
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "generation": self.generation,
            "population_size": self.population_size,
            "actual_evaluations": self.actual_evaluations,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "worst_fitness": self.worst_fitness,
            "feasible_rate": self.feasible_rate,
            "feasible_count": self.feasible_count,
            "infeasible_count": self.infeasible_count,
            "best_feasible_objective": self.best_feasible_objective,
            "best_record_feasible": self.best_record_feasible,
            "best_record_total_violation": self.best_record_total_violation,
            "mean_total_violation": self.mean_total_violation,
            "min_total_violation": self.min_total_violation,
            "max_total_violation": self.max_total_violation,
            "constraint_summary": dict(self.constraint_summary),
            "best_record": dict(self.best_record),
        }
        payload.update(self.extra_metrics)
        return payload


@dataclass(slots=True)
class ConstrainedGAResult:
    summary: dict[str, Any]
    history: list[ConstrainedGenerationSummary]

    def to_algorithm_output(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return self.summary, [item.to_dict() for item in self.history]


def _normalize_constraint_policy(config: GAConfig) -> str:
    policy = config.algorithm_options.get(
        "constraint_policy",
        config.algorithm_options.get("feasibility_policy", SUPPORTED_CONSTRAINT_POLICY),
    )
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError("constraint_policy must be a non-empty string for constrained GA")
    normalized = policy.strip().lower().replace("-", "_")
    if normalized != SUPPORTED_CONSTRAINT_POLICY:
        raise ValueError(
            "Only feasibility_first constraint handling is supported in the constrained GA "
            f"opt-in path, got {policy!r}"
        )
    return normalized


def _requested_budget(config: GAConfig) -> int:
    value = config.algorithm_options.get(
        "requested_budget",
        config.algorithm_options.get("evaluation_budget"),
    )
    if value is None:
        return config.population_size * (config.generations + 1)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("requested_budget must be a positive integer when provided")
    return value


def _normalize_violation_aggregation(config: GAConfig) -> str:
    value = config.algorithm_options.get(
        "violation_aggregation",
        SUPPORTED_VIOLATION_AGGREGATION,
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("violation_aggregation must be a non-empty string")
    normalized = value.strip().lower().replace("-", "_")
    if normalized != SUPPORTED_VIOLATION_AGGREGATION:
        raise ValueError(
            "Only total_violation aggregation is supported in the constrained GA opt-in path, "
            f"got {value!r}"
        )
    return normalized


def _constraint_evaluator(problem):
    evaluate_constraints = getattr(problem, "evaluate_constraints", None)
    if not callable(evaluate_constraints):
        raise ValueError("Constrained GA requires problem.evaluate_constraints(genome)")
    return evaluate_constraints


def _constraint_names(problem) -> list[str] | None:
    constraint_names = getattr(problem, "constraint_names", None)
    if not callable(constraint_names):
        return None
    return [str(item) for item in constraint_names()]


def _validate_constraint_evaluation(
    evaluation: ConstraintEvaluation,
    *,
    genome: Genome,
    location: str,
    generation: int | None,
    evaluation_index: int,
) -> None:
    if not isinstance(evaluation, ConstraintEvaluation):
        raise ValueError(
            "problem.evaluate_constraints must return ConstraintEvaluation "
            f"(location={location}, generation={generation}, evaluation_index={evaluation_index}, "
            f"genome={genome!r})"
        )
    validate_constraint_values(
        evaluation.inequality_values,
        label=f"{location}.inequality_values",
    )
    validate_constraint_values(
        evaluation.equality_values,
        label=f"{location}.equality_values",
    )
    finite_float_list(
        evaluation.inequality_violations,
        label=f"{location}.inequality_violations",
    )
    finite_float_list(
        evaluation.equality_violations,
        label=f"{location}.equality_violations",
    )
    finite_float_list(
        [evaluation.total_violation, evaluation.max_violation],
        label=f"{location}.violation_summary",
    )
    if not isinstance(evaluation.violation_count, int) or evaluation.violation_count < 0:
        raise ValueError(
            "ConstraintEvaluation.violation_count must be a non-negative integer "
            f"(location={location}, generation={generation}, evaluation_index={evaluation_index})"
        )


def _make_constrained_record(
    *,
    genome: Genome,
    objective: float,
    constraint_evaluation: ConstraintEvaluation,
    seed: int,
    evaluation_index: int,
    generation: int | None,
    location: str,
) -> ConstrainedCandidateRecord:
    return ConstrainedCandidateRecord(
        solution=[float(gene) for gene in genome],
        objective=float(objective),
        constraint_evaluation=constraint_evaluation,
        seed=seed,
        evaluation_index=evaluation_index,
        metadata={
            "generation": generation,
            "location": location,
        },
    )


def _evaluate_constrained_candidate(
    *,
    config: GAConfig,
    problem,
    genome: Genome,
    evaluation_index: int,
    generation: int | None,
    location: str,
) -> ConstrainedCandidateRecord:
    values = as_fitness_vector(problem.fitness(genome))
    validate_fitness_vector(
        values,
        problem_name=str(getattr(problem, "name", config.problem)),
        genome=genome,
        location=location,
        generation=generation,
        evaluation_index=evaluation_index,
    )
    if len(values) != 1:
        raise ValueError(
            "Constrained single-objective GA requires a scalar objective "
            f"(location={location}, evaluation_index={evaluation_index})"
        )
    constraint_evaluation = _constraint_evaluator(problem)(genome)
    _validate_constraint_evaluation(
        constraint_evaluation,
        genome=genome,
        location=location,
        generation=generation,
        evaluation_index=evaluation_index,
    )
    return _make_constrained_record(
        genome=genome,
        objective=values[0],
        constraint_evaluation=constraint_evaluation,
        seed=config.seed,
        evaluation_index=evaluation_index,
        generation=generation,
        location=location,
    )


def _selection_scores_from_rank(
    ranked_records: list[ConstrainedCandidateRecord],
    *,
    maximize: bool,
) -> list[float]:
    count = len(ranked_records)
    if maximize:
        return [float(count - index) for index in range(count)]
    return [float(index) for index in range(count)]


def _population_from_records(records: list[ConstrainedCandidateRecord]) -> Population:
    return [record.solution[:] for record in records]


def _best_feasible_solution(records: list[ConstrainedCandidateRecord]) -> list[float] | None:
    feasible_records = [record for record in records if record.constraint_evaluation.feasible]
    if not feasible_records:
        return None
    return select_best_feasible_first(feasible_records).solution[:]


def _best_infeasible_total_violation(records: list[ConstrainedCandidateRecord]) -> float | None:
    infeasible = [record.constraint_evaluation.total_violation for record in records if not record.constraint_evaluation.feasible]
    if not infeasible:
        return None
    return min(infeasible)


def _generation_summary(
    *,
    generation: int,
    records: list[ConstrainedCandidateRecord],
    actual_evaluations: int,
    problem,
    maximize: bool,
) -> ConstrainedGenerationSummary:
    ranked_records = rank_constrained_candidates(records, maximize=maximize)
    best_record = ranked_records[0]
    objectives = [record.objective for record in records]
    evaluation_summary = summarize_constraint_violations(
        [record.constraint_evaluation for record in records]
    )
    best_feasible = best_feasible_objective(
        objectives,
        [record.constraint_evaluation for record in records],
        maximize=maximize,
    )
    extra_metrics = dict(problem_population_metrics(problem, _population_from_records(records)))
    extra_metrics.update(problem_solution_metrics(problem, best_record.solution))

    return ConstrainedGenerationSummary(
        generation=generation,
        population_size=len(records),
        actual_evaluations=actual_evaluations,
        best_fitness=best_record.objective,
        mean_fitness=mean(objectives),
        worst_fitness=max(objectives) if not maximize else min(objectives),
        feasible_rate=evaluation_summary.feasible_rate,
        feasible_count=evaluation_summary.feasible_count,
        infeasible_count=evaluation_summary.infeasible_count,
        best_feasible_objective=best_feasible,
        best_record_feasible=best_record.constraint_evaluation.feasible,
        best_record_total_violation=best_record.constraint_evaluation.total_violation,
        mean_total_violation=evaluation_summary.mean_total_violation,
        min_total_violation=evaluation_summary.min_total_violation,
        max_total_violation=evaluation_summary.max_total_violation,
        constraint_summary=summarize_constrained_population(
            [record.constraint_evaluation for record in records]
        ),
        best_record=best_record.to_dict(),
        extra_metrics=extra_metrics,
    )


def run_constrained_single_objective_ga(
    config: GAConfig,
    problem,
    selection_fn: SelectionFn,
    crossover_fn: CrossoverFn,
    mutation_fn: MutationFn,
    init_fn: InitFn,
    rng,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    constraint_policy = _normalize_constraint_policy(config)
    violation_aggregation = _normalize_violation_aggregation(config)
    requested_budget = _requested_budget(config)

    initial_population_size = min(config.population_size, requested_budget)
    population = [init_fn(rng, config.genome_length) for _ in range(initial_population_size)]

    actual_evaluations = 0
    all_records: list[ConstrainedCandidateRecord] = []
    current_records: list[ConstrainedCandidateRecord] = []
    history: list[ConstrainedGenerationSummary] = []

    for genome in population:
        record = _evaluate_constrained_candidate(
            config=config,
            problem=problem,
            genome=genome,
            evaluation_index=actual_evaluations,
            generation=0,
            location="constrained_ga_initial_population",
        )
        actual_evaluations += 1
        current_records.append(record)
        all_records.append(record)

    if not current_records:
        raise ValueError("requested_budget must allow at least one constrained evaluation")

    history.append(
        _generation_summary(
            generation=0,
            records=current_records,
            actual_evaluations=actual_evaluations,
            problem=problem,
            maximize=config.maximize,
        )
    )

    generation = 0
    stop_reason = "requested_budget_reached" if actual_evaluations >= requested_budget else "max_generations"

    while (
        actual_evaluations < requested_budget
        and generation < config.generations
        and len(current_records) >= 2
    ):
        ranked_current = rank_constrained_candidates(current_records, maximize=config.maximize)
        ranked_population = _population_from_records(ranked_current)
        selection_scores = _selection_scores_from_rank(
            ranked_current,
            maximize=config.maximize,
        )
        selection_state = SelectionState.from_fitnesses(selection_scores, config.maximize)
        offspring_records: list[ConstrainedCandidateRecord] = []

        while (
            len(offspring_records) < config.population_size
            and actual_evaluations < requested_budget
        ):
            parent_a = selection_fn(ranked_population, selection_state, rng)
            parent_b = selection_fn(ranked_population, selection_state, rng)

            if rng.random() < config.crossover_rate:
                child_a, child_b = crossover_fn(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a[:], parent_b[:]

            child_a = apply_mutation_with_rate(
                config,
                mutation_fn,
                child_a,
                rng,
                mutation_rate=config.mutation_rate,
            )
            child_b = apply_mutation_with_rate(
                config,
                mutation_fn,
                child_b,
                rng,
                mutation_rate=config.mutation_rate,
            )

            for child in (child_a, child_b):
                if len(offspring_records) >= config.population_size:
                    break
                if actual_evaluations >= requested_budget:
                    break
                record = _evaluate_constrained_candidate(
                    config=config,
                    problem=problem,
                    genome=child,
                    evaluation_index=actual_evaluations,
                    generation=generation + 1,
                    location="constrained_ga_offspring",
                )
                actual_evaluations += 1
                offspring_records.append(record)
                all_records.append(record)

        if not offspring_records:
            stop_reason = "requested_budget_reached" if actual_evaluations >= requested_budget else "no_offspring_generated"
            break

        combined_records = [*ranked_current, *offspring_records]
        current_records = rank_constrained_candidates(
            combined_records,
            maximize=config.maximize,
        )[: config.population_size]
        generation += 1
        history.append(
            _generation_summary(
                generation=generation,
                records=current_records,
                actual_evaluations=actual_evaluations,
                problem=problem,
                maximize=config.maximize,
            )
        )
        if actual_evaluations >= requested_budget:
            stop_reason = "requested_budget_reached"

    final_ranked = rank_constrained_candidates(all_records, maximize=config.maximize)
    best_record = select_best_feasible_first(all_records, maximize=config.maximize)
    all_evaluations = [record.constraint_evaluation for record in all_records]
    all_summary = summarize_constraint_violations(all_evaluations)
    objective_values = [record.objective for record in all_records]
    best_feasible = best_feasible_objective(
        objective_values,
        all_evaluations,
        maximize=config.maximize,
    )
    trace_records, trace_warnings = build_per_constraint_trace_records(
        all_records,
        strategy="constrained_ga_feasibility_first",
        constraint_names=_constraint_names(problem),
        metadata={"source": "constrained_ga_all_evaluations"},
    )
    per_constraint_trace_summary = per_constraint_summary_to_dict(
        summarize_per_constraint_trace(
            trace_records,
            warnings=trace_warnings,
            sample_limit=20,
        )
    )
    summary = {
        "best_fitness": best_record.objective,
        "mean_fitness": mean(objective_values),
        "worst_fitness": final_ranked[-1].objective,
        "best_fitness_vector": [best_record.objective],
        "best_genome": best_record.solution[:],
        "best_record": best_record.to_dict(),
        "best_record_feasible": best_record.constraint_evaluation.feasible,
        "best_record_total_violation": best_record.constraint_evaluation.total_violation,
        "best_record_max_violation": best_record.constraint_evaluation.max_violation,
        "best_feasible_objective": best_feasible,
        "best_feasible_solution": _best_feasible_solution(all_records),
        "best_infeasible_total_violation": _best_infeasible_total_violation(all_records),
        "constraint_summary": summarize_constrained_population(all_evaluations),
        "per_constraint_trace_summary": per_constraint_trace_summary,
        "final_population_constraint_summary": summarize_constrained_population(
            [record.constraint_evaluation for record in current_records]
        ),
        "feasible_rate": all_summary.feasible_rate,
        "feasible_count": all_summary.feasible_count,
        "infeasible_count": all_summary.infeasible_count,
        "mean_total_violation": all_summary.mean_total_violation,
        "median_total_violation": all_summary.median_total_violation,
        "min_total_violation": all_summary.min_total_violation,
        "max_total_violation": all_summary.max_total_violation,
        "mean_max_violation": all_summary.mean_max_violation,
        "violation_count_total": all_summary.violation_count_total,
        "all_feasible": all_summary.all_feasible,
        "any_feasible": all_summary.any_feasible,
        "all_infeasible": all_summary.all_infeasible,
        "requested_budget": requested_budget,
        "actual_evaluations": actual_evaluations,
        "objective_count": 1,
        "objective_directions": [config.maximize],
        "constraint_policy": constraint_policy,
        "feasibility_policy": constraint_policy,
        "violation_aggregation": violation_aggregation,
        "stop_reason": stop_reason,
        "final_generation": generation,
        "default_changed": False,
        "ga_integration_done": False,
        "constrained_ga_opt_in_path_used": True,
        "nsga2_constraint_domination_done": False,
    }
    summary.update(problem_solution_metrics(problem, best_record.solution))
    summary.update(problem_population_metrics(problem, _population_from_records(current_records)))

    return ConstrainedGAResult(summary=summary, history=history).to_algorithm_output()


__all__ = [
    "ConstrainedGAResult",
    "ConstrainedGenerationSummary",
    "run_constrained_single_objective_ga",
]
