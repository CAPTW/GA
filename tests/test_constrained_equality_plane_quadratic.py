from __future__ import annotations

import pytest

from ga_lab.problems import build_problem
from ga_lab.problems.constrained_equality_plane_quadratic import (
    ConstrainedEqualityPlaneQuadraticProblem,
)


def test_constrained_equality_plane_quadratic_objective_is_deterministic() -> None:
    problem = ConstrainedEqualityPlaneQuadraticProblem(dimension=6)
    solution = list(problem.targets)

    first = problem.evaluate_objective(solution)
    second = problem.evaluate_objective(solution)

    assert first == pytest.approx(second)


def test_clearly_feasible_solution_is_feasible() -> None:
    problem = ConstrainedEqualityPlaneQuadraticProblem(dimension=6)

    evaluation = problem.evaluate_constraints(list(problem.targets))

    assert evaluation.feasible is True
    assert evaluation.equality_violations == [pytest.approx(0.0)]
    assert evaluation.inequality_violations == [pytest.approx(0.0)]


def test_clearly_infeasible_solution_is_infeasible() -> None:
    problem = ConstrainedEqualityPlaneQuadraticProblem(dimension=6)

    evaluation = problem.evaluate_constraints([2.0 for _ in range(6)])

    assert evaluation.feasible is False
    assert evaluation.equality_violations[0] > 0.0
    assert evaluation.inequality_violations[0] > 0.0


def test_tolerance_boundary_case_has_zero_violation() -> None:
    problem = ConstrainedEqualityPlaneQuadraticProblem(
        dimension=6,
        equality_target=0.0,
        equality_tolerance=0.125,
        group_budget=1.0,
        targets=[0.0] * 6,
    )
    solution = [0.125, 0.0, 0.0, 0.0, 0.0, 0.0]

    evaluation = problem.evaluate_constraints(solution)

    assert evaluation.feasible is True
    assert evaluation.equality_violations == [pytest.approx(0.0)]


def test_near_feasible_outside_tolerance_has_positive_violation() -> None:
    problem = ConstrainedEqualityPlaneQuadraticProblem(
        dimension=6,
        equality_target=0.0,
        equality_tolerance=0.125,
        group_budget=1.0,
        targets=[0.0] * 6,
    )
    solution = [0.126, 0.0, 0.0, 0.0, 0.0, 0.0]

    evaluation = problem.evaluate_constraints(solution)

    assert evaluation.feasible is False
    assert evaluation.equality_violations[0] == pytest.approx(0.001)


def test_metadata_contains_equality_count_and_tolerance() -> None:
    problem = ConstrainedEqualityPlaneQuadraticProblem(dimension=6, equality_tolerance=0.02)
    metadata = problem.metadata()

    assert metadata.exact_genome_length == 6
    assert problem.source_bounds() == [(-5.0, 5.0)] * 6
    assert problem.constraint_count == 2
    assert problem.inequality_count == 1
    assert problem.equality_count == 1
    assert problem.equality_tolerance == pytest.approx(0.02)
    assert problem.constraint_names() == ("second_half_budget", "plane_sum_target")


def test_non_finite_values_fail_fast() -> None:
    problem = ConstrainedEqualityPlaneQuadraticProblem(dimension=6)

    with pytest.raises(ValueError, match="Non-finite constraint value"):
        problem.evaluate_constraints([float("nan"), 0.2, 0.2, 0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match="Non-finite constraint value"):
        problem.evaluate_objective([0.2, 0.2, 0.2, 0.2, 0.2, float("inf")])


def test_registry_lookup_works_for_normalized_names() -> None:
    via_underscore = build_problem("constrained_equality_plane_quadratic", {"dimension": 6})
    via_hyphen = build_problem("constrained-equality-plane-quadratic", {"dimension": 6})

    assert isinstance(via_underscore, ConstrainedEqualityPlaneQuadraticProblem)
    assert isinstance(via_hyphen, ConstrainedEqualityPlaneQuadraticProblem)
    assert via_underscore.name == "constrained_equality_plane_quadratic"
    assert via_hyphen.name == "constrained_equality_plane_quadratic"
