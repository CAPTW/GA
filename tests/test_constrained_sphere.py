from __future__ import annotations

import json

import pytest

from ga_lab.problems import build_problem
from ga_lab.problems.constrained_sphere import ConstrainedSphereProblem


def test_constrained_sphere_objective_matches_sum_of_squares() -> None:
    problem = ConstrainedSphereProblem(dimension=3)

    assert problem.evaluate_objective([1.0, -2.0, 0.5]) == pytest.approx(5.25)
    assert problem.fitness([1.0, -2.0, 0.5]) == pytest.approx(5.25)


def test_constrained_sphere_feasible_solution_has_zero_violation() -> None:
    problem = ConstrainedSphereProblem(dimension=3, budget=1.0)

    evaluation = problem.evaluate_constraints([0.2, 0.2, 0.2])

    assert evaluation.feasible is True
    assert evaluation.total_violation == 0.0
    assert evaluation.inequality_values == [pytest.approx(-0.4)]


def test_constrained_sphere_infeasible_solution_reports_budget_violation() -> None:
    problem = ConstrainedSphereProblem(dimension=3, budget=1.0)

    evaluation = problem.evaluate_constraints([1.0, 1.0, 1.0])

    assert evaluation.feasible is False
    assert evaluation.inequality_values == [pytest.approx(2.0)]
    assert evaluation.inequality_violations == [pytest.approx(2.0)]
    assert evaluation.total_violation == pytest.approx(2.0)


def test_constrained_sphere_exposes_bounds_and_dimension_metadata() -> None:
    problem = ConstrainedSphereProblem(dimension=5, lower_bound=-5.0, upper_bound=5.0)
    metadata = problem.metadata()

    assert problem.bounds == (-5.0, 5.0)
    assert problem.source_bounds() == [(-5.0, 5.0)] * 5
    assert metadata.exact_genome_length == 5
    assert metadata.default_objective_directions == (False,)


def test_constrained_sphere_registry_lookup_supports_normalized_names() -> None:
    via_underscore = build_problem("constrained_sphere", {"dimension": 2})
    via_hyphen = build_problem("constrained-sphere", {"dimension": 2})

    assert isinstance(via_underscore, ConstrainedSphereProblem)
    assert isinstance(via_hyphen, ConstrainedSphereProblem)
    assert via_underscore.name == "constrained_sphere"
    assert via_hyphen.name == "constrained_sphere"


def test_constrained_sphere_non_finite_genome_values_fail_fast() -> None:
    problem = ConstrainedSphereProblem(dimension=2)

    with pytest.raises(ValueError, match="Non-finite constraint value"):
        problem.evaluate_constraints([float("nan"), 0.0])
    with pytest.raises(ValueError, match="Non-finite constraint value"):
        problem.evaluate_objective([0.0, float("inf")])


def test_constrained_sphere_evaluate_payload_is_json_serializable() -> None:
    problem = ConstrainedSphereProblem(dimension=2, budget=1.0)

    payload = problem.evaluate([0.2, 0.3])

    assert payload["objective"] == pytest.approx(0.13)
    json.dumps(payload)
