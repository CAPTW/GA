from __future__ import annotations

import pytest

from ga_lab.problems import build_problem
from ga_lab.problems.constrained_zdt_box_toy import ConstrainedZDTBoxToyProblem


def test_constrained_zdt_box_toy_objective_is_deterministic() -> None:
    problem = ConstrainedZDTBoxToyProblem(dimension=6)
    solution = [0.2, 0.3, 0.0, 0.2, 0.2, 0.2]

    first = problem.evaluate_objectives(solution)
    second = problem.evaluate_objectives(solution)

    assert first == pytest.approx(second)


def test_feasible_solution_is_feasible() -> None:
    problem = ConstrainedZDTBoxToyProblem(dimension=6)

    evaluation = problem.evaluate_constraints([0.2, 0.3, 0.0, 0.2, 0.2, 0.2])

    assert evaluation.feasible is True
    assert evaluation.total_violation == pytest.approx(0.0)


def test_infeasible_solution_is_infeasible() -> None:
    problem = ConstrainedZDTBoxToyProblem(dimension=6)

    evaluation = problem.evaluate_constraints([0.8, 0.5, 0.6, 0.6, 0.6, 0.6])

    assert evaluation.feasible is False
    assert evaluation.inequality_violations[0] > 0.0


def test_both_constraints_violated_case_reports_total_violation() -> None:
    problem = ConstrainedZDTBoxToyProblem(dimension=6)

    evaluation = problem.evaluate_constraints([0.9, 0.6, 0.5, 0.5, 0.5, 0.5])

    assert evaluation.feasible is False
    assert evaluation.inequality_violations == [pytest.approx(0.5), pytest.approx(0.3)]
    assert evaluation.total_violation == pytest.approx(0.8)


def test_boundary_case_has_zero_violation() -> None:
    problem = ConstrainedZDTBoxToyProblem(dimension=6)
    solution = [0.6, 0.4, 0.1, 0.4, 0.4, problem.second_half_budget - 0.8]

    evaluation = problem.evaluate_constraints(solution)

    assert evaluation.feasible is True
    assert evaluation.total_violation == pytest.approx(0.0)


def test_metadata_includes_objective_and_constraint_counts() -> None:
    problem = ConstrainedZDTBoxToyProblem(dimension=6)
    metadata = problem.metadata()

    assert metadata.exact_genome_length == 6
    assert problem.objective_count == 2
    assert problem.constraint_count == 2
    assert problem.inequality_count == 2
    assert problem.equality_count == 0
    assert problem.source_bounds() == [(0.0, 1.0)] * 6


def test_registry_lookup_works() -> None:
    via_underscore = build_problem("constrained_zdt_box_toy", {"dimension": 6})
    via_hyphen = build_problem("constrained-zdt-box-toy", {"dimension": 6})

    assert isinstance(via_underscore, ConstrainedZDTBoxToyProblem)
    assert isinstance(via_hyphen, ConstrainedZDTBoxToyProblem)


def test_nan_and_inf_input_fail_fast() -> None:
    problem = ConstrainedZDTBoxToyProblem(dimension=6)

    with pytest.raises(ValueError, match="Non-finite constraint value"):
        problem.evaluate_constraints([float("nan"), 0.1, 0.1, 0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="Non-finite constraint value"):
        problem.evaluate_objectives([0.1, 0.1, 0.1, 0.1, 0.1, float("inf")])
