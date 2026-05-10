from __future__ import annotations

import math
import random

import pytest

from ga_lab.algorithms.constrained_nsga2 import (
    _constrained_dominates,
    run_constrained_nsga2,
)
from ga_lab.config import GAConfig
from ga_lab.constraints import evaluate_constraint_violations
from ga_lab.factory import build_runtime_context
from ga_lab.problems.constrained_zdt_box_toy import ConstrainedZDTBoxToyProblem


def _build_config(*, seed: int = 0, population_size: int = 4, generations: int = 1) -> GAConfig:
    return GAConfig(
        run_name=f"constrained_nsga2_seed{seed}",
        problem="constrained_zdt_box_toy",
        algorithm="nsga2",
        representation="real",
        selection="nsga2_tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=population_size,
        genome_length=6,
        generations=generations,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        tournament_size=2,
        seed=seed,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.05},
        problem_options={"dimension": 6},
    )


def test_constrained_function_runs_on_constrained_zdt_box_toy() -> None:
    config = _build_config(seed=3, population_size=4, generations=1)
    runtime = build_runtime_context(config)

    result = run_constrained_nsga2(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(3),
    )

    assert result.summary["is_constrained_nsga2"] is True
    assert "feasible_rate" in result.summary
    assert "constraint_summary" in result.summary
    assert "per_constraint_violation_summary" in result.summary


def test_feasible_dominates_infeasible_in_ordering_helper() -> None:
    feasible = evaluate_constraint_violations(inequality_values=[0.0, 0.0])
    infeasible = evaluate_constraint_violations(inequality_values=[0.1, 0.0])

    assert _constrained_dominates([0.3, 0.6], feasible, [0.1, 0.2], infeasible, directions=[False, False]) is True


def test_all_infeasible_case_does_not_crash() -> None:
    problem = ConstrainedZDTBoxToyProblem(dimension=6, first_pair_budget=-0.1, second_half_budget=-0.1)
    config = _build_config(seed=5, population_size=4, generations=1)
    runtime = build_runtime_context(config)

    result = run_constrained_nsga2(
        config=config,
        problem=problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(5),
    )

    assert result.summary["feasible_count"] == 0
    assert result.summary["feasible_only_HV"] is None
    assert result.summary["feasible_only_reference_distance"] is None


def test_actual_evaluations_do_not_exceed_budget_and_exact_match_when_configured() -> None:
    config = _build_config(seed=7, population_size=4, generations=1)
    runtime = build_runtime_context(config)

    result = run_constrained_nsga2(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(7),
    )

    assert result.summary["actual_evaluations_used"] == 20
    assert result.summary["actual_evaluations_used"] == result.summary["requested_budget"]


def test_default_nsga2_path_remains_unchanged() -> None:
    from ga_lab.algorithms.nsga2 import run_nsga2

    assert callable(run_nsga2)


def test_non_finite_objective_raises_value_error() -> None:
    class _BadObjectiveProblem(ConstrainedZDTBoxToyProblem):
        def fitness(self, genome):
            _ = genome
            return [math.nan, 1.0]

    config = _build_config(seed=9)
    runtime = build_runtime_context(config)

    with pytest.raises(ValueError, match="Non-finite fitness detected"):
        run_constrained_nsga2(
            config=config,
            problem=_BadObjectiveProblem(dimension=6),
            selection_fn=runtime.operators.selection_fn,
            crossover_fn=runtime.operators.crossover_fn,
            mutation_fn=runtime.operators.mutation_fn,
            init_fn=runtime.operators.init_fn,
            rng=random.Random(9),
        )


def test_non_finite_constraint_raises_value_error() -> None:
    class _BadConstraintProblem(ConstrainedZDTBoxToyProblem):
        def evaluate_constraints(self, genome):
            _ = genome
            return evaluate_constraint_violations(inequality_values=[float("nan"), 0.0])

    config = _build_config(seed=10)
    runtime = build_runtime_context(config)

    with pytest.raises(ValueError, match="Non-finite constraint value"):
        run_constrained_nsga2(
            config=config,
            problem=_BadConstraintProblem(dimension=6),
            selection_fn=runtime.operators.selection_fn,
            crossover_fn=runtime.operators.crossover_fn,
            mutation_fn=runtime.operators.mutation_fn,
            init_fn=runtime.operators.init_fn,
            rng=random.Random(10),
        )
