from __future__ import annotations

import math

import pytest

from ga_lab.problems import build_problem
from ga_lab.problems.constrained_dtlz_box_toy import ConstrainedDTLZBoxToyProblem


def test_constrained_dtlz_box_toy_objective_is_deterministic() -> None:
    problem = ConstrainedDTLZBoxToyProblem(dimension=7)
    solution = [0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2]

    first = problem.evaluate_objectives(solution)
    second = problem.evaluate_objectives(solution)

    assert first == pytest.approx(second)


def test_feasible_solution_is_feasible() -> None:
    problem = ConstrainedDTLZBoxToyProblem(dimension=7)

    evaluation = problem.evaluate_constraints([0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2])

    assert evaluation.feasible is True
    assert evaluation.total_violation == pytest.approx(0.0)


def test_infeasible_solution_is_infeasible() -> None:
    problem = ConstrainedDTLZBoxToyProblem(dimension=7)

    evaluation = problem.evaluate_constraints([0.8, 0.5, 0.2, 0.7, 0.7, 0.7, 0.7])

    assert evaluation.feasible is False
    assert any(violation > 0.0 for violation in evaluation.inequality_violations)


def test_both_constraints_violated_case_reports_total_violation() -> None:
    problem = ConstrainedDTLZBoxToyProblem(dimension=7)

    evaluation = problem.evaluate_constraints([0.8, 0.5, 0.2, 0.7, 0.7, 0.7, 0.7])

    assert evaluation.feasible is False
    assert evaluation.inequality_violations == [pytest.approx(0.3), pytest.approx(0.15)]
    assert evaluation.total_violation == pytest.approx(0.45)


def test_boundary_case_has_zero_violation() -> None:
    problem = ConstrainedDTLZBoxToyProblem(dimension=7)
    solution = [0.6, 0.4, 0.1, 0.55, 0.55, 0.55, 0.55]

    evaluation = problem.evaluate_constraints(solution)

    assert evaluation.feasible is True
    assert evaluation.total_violation == pytest.approx(0.0)


def test_metadata_includes_objective_and_constraint_counts() -> None:
    problem = ConstrainedDTLZBoxToyProblem(dimension=7)
    metadata = problem.metadata()

    assert metadata.exact_genome_length == 7
    assert problem.objective_count == 2
    assert problem.constraint_count == 2
    assert problem.inequality_count == 2
    assert problem.equality_count == 0
    assert problem.source_bounds() == [(0.0, 1.0)] * 7


def test_registry_lookup_works() -> None:
    via_underscore = build_problem("constrained_dtlz_box_toy", {"dimension": 7})
    via_hyphen = build_problem("constrained-dtlz-box-toy", {"dimension": 7})

    assert isinstance(via_underscore, ConstrainedDTLZBoxToyProblem)
    assert isinstance(via_hyphen, ConstrainedDTLZBoxToyProblem)


def test_nan_and_inf_input_fail_fast() -> None:
    problem = ConstrainedDTLZBoxToyProblem(dimension=7)

    with pytest.raises(ValueError, match="Non-finite constraint value"):
        problem.evaluate_constraints([math.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="Non-finite constraint value"):
        problem.evaluate_objectives([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, math.inf])
