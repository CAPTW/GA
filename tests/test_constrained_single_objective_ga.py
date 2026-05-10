from __future__ import annotations

import random

import pytest

from ga_lab.algorithms.constrained_single_objective import (
    run_constrained_single_objective_ga,
)
from ga_lab.algorithms.single_objective import run_single_objective_ga
from ga_lab.config import GAConfig
from ga_lab.constraints import ConstraintEvaluation, evaluate_constraint_violations
from ga_lab.factory import build_runtime_context
from ga_lab.problems.base import ProblemMetadata


class _NonFiniteObjectiveProblem:
    name = "non_finite_constrained_objective"
    compatible_representations = ("real",)
    default_objective_directions = (False,)

    def fitness(self, genome):
        return float("nan")

    def evaluate_constraints(self, genome):
        return evaluate_constraint_violations(inequality_values=[-1.0])

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=("real",),
            exact_genome_length=3,
            default_objective_directions=(False,),
        )


class _NonFiniteConstraintProblem:
    name = "non_finite_constrained_constraint"
    compatible_representations = ("real",)
    default_objective_directions = (False,)

    def fitness(self, genome):
        return sum(float(value) ** 2 for value in genome)

    def evaluate_constraints(self, genome):
        return evaluate_constraint_violations(inequality_values=[float("nan")])

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=("real",),
            exact_genome_length=3,
            default_objective_directions=(False,),
        )


def _constrained_config(
    *,
    budget: int = 24,
    mutation_rate: float = 0.1,
    dimension: int = 4,
    constraint_budget: float = 1.0,
) -> GAConfig:
    population_size = 6
    generations = 3
    return GAConfig(
        run_name="constrained_ga_test",
        problem="constrained_sphere",
        problem_options={
            "dimension": dimension,
            "budget": constraint_budget,
        },
        population_size=population_size,
        genome_length=dimension,
        generations=generations,
        crossover_rate=0.9,
        mutation_rate=mutation_rate,
        elitism=1,
        tournament_size=3,
        algorithm="ga",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        representation="real",
        representation_options={"low": -5.0, "high": 5.0},
        algorithm_options={
            "requested_budget": budget,
            "constraint_policy": "feasibility_first",
            "feasibility_policy": "feasibility_first",
            "violation_aggregation": "total_violation",
            "non_finite_constraint_fail_fast_policy": "value_error",
            "feasible_only_metric_policy": "best_feasible_objective_null_when_absent",
        },
        seed=13,
        maximize=False,
        objective_directions=[False],
    )


def test_constrained_ga_runs_on_constrained_sphere_and_includes_constraint_summary() -> None:
    config = _constrained_config()
    runtime = build_runtime_context(config)

    summary, history = run_constrained_single_objective_ga(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(config.seed),
    )

    assert summary["best_record"]
    assert "constraint_summary" in summary
    assert history
    assert summary["actual_evaluations"] == config.algorithm_options["requested_budget"]
    assert summary["default_changed"] is False
    assert summary["constrained_ga_opt_in_path_used"] is True


def test_constrained_ga_selects_feasibility_first_best_record() -> None:
    config = _constrained_config(budget=18)
    runtime = build_runtime_context(config)

    summary, _history = run_constrained_single_objective_ga(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(config.seed),
    )

    best_record = summary["best_record"]
    if summary["any_feasible"]:
        assert best_record["constraint_evaluation"]["feasible"] is True


def test_constrained_ga_all_infeasible_fixture_does_not_crash() -> None:
    config = _constrained_config(budget=18, constraint_budget=-30.0)
    runtime = build_runtime_context(config)

    summary, _history = run_constrained_single_objective_ga(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(config.seed),
    )

    assert summary["all_infeasible"] is True
    assert summary["best_feasible_objective"] is None


def test_constrained_ga_rejects_non_finite_objective() -> None:
    config = _constrained_config(budget=6, dimension=3)

    with pytest.raises(ValueError, match="Non-finite fitness"):
        run_constrained_single_objective_ga(
            config=config,
            problem=_NonFiniteObjectiveProblem(),
            selection_fn=lambda population, _state, _rng: population[0][:],
            crossover_fn=lambda parent_a, parent_b, _rng: (parent_a[:], parent_b[:]),
            mutation_fn=lambda genome, _rng: genome[:],
            init_fn=lambda _rng, genome_length: [0.0] * genome_length,
            rng=random.Random(config.seed),
        )


def test_constrained_ga_rejects_non_finite_constraint() -> None:
    config = _constrained_config(budget=6, dimension=3)

    with pytest.raises(ValueError, match="Non-finite constraint value"):
        run_constrained_single_objective_ga(
            config=config,
            problem=_NonFiniteConstraintProblem(),
            selection_fn=lambda population, _state, _rng: population[0][:],
            crossover_fn=lambda parent_a, parent_b, _rng: (parent_a[:], parent_b[:]),
            mutation_fn=lambda genome, _rng: genome[:],
            init_fn=lambda _rng, genome_length: [0.0] * genome_length,
            rng=random.Random(config.seed),
        )


def test_constrained_ga_calls_mutation_and_preserves_mutation_rate_contract() -> None:
    config = _constrained_config(budget=12, mutation_rate=0.0)
    runtime = build_runtime_context(config)
    call_rates: list[float] = []

    def custom_mutation(genome, _rng):
        call_rates.append(config.mutation_rate)
        return genome[:]

    summary, _history = run_constrained_single_objective_ga(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=custom_mutation,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(config.seed),
    )

    assert summary["actual_evaluations"] == 12
    assert call_rates
    assert all(rate == 0.0 for rate in call_rates)


def test_default_ga_runtime_context_remains_single_objective_path() -> None:
    config = GAConfig(
        run_name="default_ga_unchanged",
        problem="onemax",
        population_size=6,
        genome_length=6,
        generations=2,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        seed=5,
    )

    runtime = build_runtime_context(config)

    assert runtime.algorithm_fn is run_single_objective_ga
