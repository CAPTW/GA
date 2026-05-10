from __future__ import annotations

import pytest

from ga_lab.problems import build_problem
from ga_lab.problems.constrained_box_quadratic import ConstrainedBoxQuadraticProblem


def test_constrained_box_quadratic_objective_is_deterministic() -> None:
    problem = ConstrainedBoxQuadraticProblem(dimension=6)
    solution = [0.4, 0.4, 0.4, 0.3, 0.3, 0.3]

    first = problem.evaluate_objective(solution)
    second = problem.evaluate_objective(solution)

    assert first == pytest.approx(second)


def test_constrained_box_quadratic_clearly_feasible_solution_is_feasible() -> None:
    problem = ConstrainedBoxQuadraticProblem(dimension=6)

    evaluation = problem.evaluate_constraints([0.4, 0.4, 0.4, 0.3, 0.3, 0.3])

    assert evaluation.feasible is True
    assert evaluation.inequality_violations == [pytest.approx(0.0), pytest.approx(0.0)]


def test_constrained_box_quadratic_clearly_infeasible_solution_is_infeasible() -> None:
    problem = ConstrainedBoxQuadraticProblem(dimension=6)

    evaluation = problem.evaluate_constraints([1.5, 1.5, 1.5, 1.4, 1.4, 1.4])

    assert evaluation.feasible is False
    assert evaluation.inequality_violations[0] > 0.0
    assert evaluation.inequality_violations[1] > 0.0


def test_constrained_box_quadratic_reports_both_group_violations() -> None:
    problem = ConstrainedBoxQuadraticProblem(dimension=6)

    evaluation = problem.evaluate_constraints([1.0, 0.8, 0.7, 0.8, 0.5, 0.5])

    assert evaluation.feasible is False
    assert evaluation.inequality_violations == [
        pytest.approx(0.5),
        pytest.approx(0.3),
    ]
    assert evaluation.total_violation == pytest.approx(0.8)


def test_constrained_box_quadratic_boundary_case_has_zero_violation() -> None:
    problem = ConstrainedBoxQuadraticProblem(dimension=6)

    evaluation = problem.evaluate_constraints([0.7, 0.7, 0.6, 0.5, 0.5, 0.5])

    assert evaluation.feasible is True
    assert evaluation.total_violation == pytest.approx(0.0)


def test_constrained_box_quadratic_non_finite_values_fail_fast() -> None:
    problem = ConstrainedBoxQuadraticProblem(dimension=6)

    with pytest.raises(ValueError, match="Non-finite constraint value"):
        problem.evaluate_constraints([float("nan"), 0.2, 0.2, 0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match="Non-finite constraint value"):
        problem.evaluate_objective([0.2, 0.2, 0.2, 0.2, 0.2, float("inf")])


def test_constrained_box_quadratic_metadata_contains_dimension_bounds_and_constraint_counts() -> None:
    problem = ConstrainedBoxQuadraticProblem(dimension=6)
    metadata = problem.metadata()

    assert metadata.exact_genome_length == 6
    assert problem.source_bounds() == [(-5.0, 5.0)] * 6
    assert problem.constraint_count == 2
    assert problem.inequality_count == 2
    assert problem.equality_count == 0


def test_constrained_box_quadratic_registry_lookup_works_for_normalized_names() -> None:
    via_underscore = build_problem("constrained_box_quadratic", {"dimension": 6})
    via_hyphen = build_problem("constrained-box-quadratic", {"dimension": 6})

    assert isinstance(via_underscore, ConstrainedBoxQuadraticProblem)
    assert isinstance(via_hyphen, ConstrainedBoxQuadraticProblem)
    assert via_underscore.name == "constrained_box_quadratic"
    assert via_hyphen.name == "constrained_box_quadratic"
