from __future__ import annotations

from ga_lab.constraints import evaluate_constraint_violations
from ga_lab.experiment.constrained_protocol import (
    ConstrainedCandidateRecord,
    compare_feasibility_first,
    rank_constrained_candidates,
    select_best_feasible_first,
)


def _record(
    *,
    objective: float,
    feasible: bool = True,
    total_violation: float = 0.0,
    max_violation: float = 0.0,
    seed: int = 0,
    evaluation_index: int = 0,
) -> ConstrainedCandidateRecord:
    if feasible:
        evaluation = evaluate_constraint_violations(inequality_values=[-1.0])
    else:
        evaluation = evaluate_constraint_violations(inequality_values=[total_violation])
        evaluation.max_violation = max_violation
        evaluation.total_violation = total_violation
    return ConstrainedCandidateRecord(
        solution=[objective],
        objective=objective,
        constraint_evaluation=evaluation,
        seed=seed,
        evaluation_index=evaluation_index,
    )


def test_feasible_beats_infeasible() -> None:
    left = _record(objective=10.0, feasible=True)
    right = _record(objective=1.0, feasible=False, total_violation=0.01)

    assert compare_feasibility_first(left, right) == -1


def test_feasible_smaller_objective_wins() -> None:
    left = _record(objective=1.0, feasible=True)
    right = _record(objective=2.0, feasible=True)

    assert compare_feasibility_first(left, right) == -1


def test_infeasible_lower_total_violation_wins() -> None:
    left = _record(objective=10.0, feasible=False, total_violation=0.2, max_violation=0.2)
    right = _record(objective=1.0, feasible=False, total_violation=0.5, max_violation=0.5)

    assert compare_feasibility_first(left, right) == -1


def test_infeasible_lower_max_violation_tie_break_wins() -> None:
    left = _record(objective=10.0, feasible=False, total_violation=0.5, max_violation=0.1)
    right = _record(objective=1.0, feasible=False, total_violation=0.5, max_violation=0.2)

    assert compare_feasibility_first(left, right) == -1


def test_stable_tie_behavior_is_preserved() -> None:
    first = _record(objective=1.0, feasible=True, evaluation_index=0)
    second = _record(objective=1.0, feasible=True, evaluation_index=1)

    ranked = rank_constrained_candidates([first, second])

    assert ranked[0] is first
    assert ranked[1] is second


def test_select_best_feasible_first_returns_expected_record() -> None:
    feasible = _record(objective=2.0, feasible=True)
    infeasible = _record(objective=1.0, feasible=False, total_violation=0.1)
    better_feasible = _record(objective=1.0, feasible=True)

    best = select_best_feasible_first([feasible, infeasible, better_feasible])

    assert best is better_feasible
