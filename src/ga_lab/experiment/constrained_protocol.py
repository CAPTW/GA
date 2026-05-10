from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Any, Sequence

from ga_lab.constraints import ConstraintEvaluation


@dataclass(slots=True)
class ConstrainedCandidateRecord:
    solution: list[float]
    objective: float
    constraint_evaluation: ConstraintEvaluation
    seed: int
    evaluation_index: int
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution": list(self.solution),
            "objective": float(self.objective),
            "constraint_evaluation": self.constraint_evaluation.to_dict(),
            "seed": self.seed,
            "evaluation_index": self.evaluation_index,
            "metadata": dict(self.metadata),
        }


def compare_feasibility_first(
    left: ConstrainedCandidateRecord,
    right: ConstrainedCandidateRecord,
    *,
    maximize: bool = False,
) -> int:
    left_eval = left.constraint_evaluation
    right_eval = right.constraint_evaluation

    if left_eval.feasible and not right_eval.feasible:
        return -1
    if right_eval.feasible and not left_eval.feasible:
        return 1

    if left_eval.feasible and right_eval.feasible:
        if left.objective == right.objective:
            return 0
        if maximize:
            return -1 if left.objective > right.objective else 1
        return -1 if left.objective < right.objective else 1

    left_tuple = (
        left_eval.total_violation,
        left_eval.max_violation,
        -left.objective if maximize else left.objective,
    )
    right_tuple = (
        right_eval.total_violation,
        right_eval.max_violation,
        -right.objective if maximize else right.objective,
    )
    if left_tuple < right_tuple:
        return -1
    if left_tuple > right_tuple:
        return 1
    return 0


def rank_constrained_candidates(
    records: Sequence[ConstrainedCandidateRecord],
    *,
    maximize: bool = False,
) -> list[ConstrainedCandidateRecord]:
    return sorted(
        list(records),
        key=cmp_to_key(lambda left, right: compare_feasibility_first(left, right, maximize=maximize)),
    )


def select_best_feasible_first(
    records: Sequence[ConstrainedCandidateRecord],
    *,
    maximize: bool = False,
) -> ConstrainedCandidateRecord:
    if not records:
        raise ValueError("records cannot be empty")
    return rank_constrained_candidates(records, maximize=maximize)[0]


__all__ = [
    "ConstrainedCandidateRecord",
    "compare_feasibility_first",
    "rank_constrained_candidates",
    "select_best_feasible_first",
]
