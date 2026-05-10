from __future__ import annotations

import math
from typing import Any, Sequence

from ga_lab.constraints import ConstraintEvaluation, summarize_constraint_violations
from ga_lab.experiment.constrained_metrics import per_constraint_violation_summary
from ga_lab.experiment.mo_metrics import (
    generational_distance,
    hypervolume_2d,
    inverted_generational_distance,
    nondominated_vectors,
    spacing_metric,
)
from ga_lab.metrics import finite_or_none


def _feasible_objective_vectors(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
) -> list[list[float]]:
    return [
        [float(value) for value in objectives]
        for objectives, evaluation in zip(objective_vectors, evaluations, strict=True)
        if evaluation.feasible
    ]


def _infeasible_objective_vectors(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
) -> list[list[float]]:
    return [
        [float(value) for value in objectives]
        for objectives, evaluation in zip(objective_vectors, evaluations, strict=True)
        if not evaluation.feasible
    ]


def feasible_only_hypervolume_2d(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    *,
    directions: Sequence[bool],
    reference_point: Sequence[float] | None,
) -> float | None:
    feasible_vectors = _feasible_objective_vectors(objective_vectors, evaluations)
    if not feasible_vectors or reference_point is None:
        return None
    return finite_or_none(hypervolume_2d(feasible_vectors, directions, reference_point))


def feasible_only_reference_distance(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    *,
    directions: Sequence[bool],
    reference_front: Sequence[Sequence[float]] | None,
) -> float | None:
    feasible_vectors = _feasible_objective_vectors(objective_vectors, evaluations)
    if not feasible_vectors or not reference_front:
        return None
    return finite_or_none(generational_distance(feasible_vectors, reference_front, directions))


def feasible_only_igd(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    *,
    directions: Sequence[bool],
    reference_front: Sequence[Sequence[float]] | None,
) -> float | None:
    feasible_vectors = _feasible_objective_vectors(objective_vectors, evaluations)
    if not feasible_vectors or not reference_front:
        return None
    return finite_or_none(
        inverted_generational_distance(feasible_vectors, reference_front, directions)
    )


def feasible_only_spacing(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    *,
    directions: Sequence[bool],
) -> float | None:
    feasible_vectors = _feasible_objective_vectors(objective_vectors, evaluations)
    if not feasible_vectors:
        return None
    return finite_or_none(spacing_metric(feasible_vectors, directions))


def feasible_nondominated_count(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    *,
    directions: Sequence[bool],
) -> int:
    feasible_vectors = _feasible_objective_vectors(objective_vectors, evaluations)
    return len(nondominated_vectors(feasible_vectors, directions))


def infeasible_nondominated_count(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    *,
    directions: Sequence[bool],
) -> int:
    infeasible_vectors = _infeasible_objective_vectors(objective_vectors, evaluations)
    return len(nondominated_vectors(infeasible_vectors, directions))


def violation_summary_to_dict_mo(
    evaluations: Sequence[ConstraintEvaluation],
) -> dict[str, Any]:
    return summarize_constraint_violations(evaluations).to_dict()


def constrained_mo_summary_to_dict(
    summary: dict[str, Any],
) -> dict[str, Any]:
    return dict(summary)


def summarize_constrained_mo_population(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    *,
    directions: Sequence[bool],
    reference_point: Sequence[float] | None,
    reference_front: Sequence[Sequence[float]] | None,
    inequality_names: Sequence[str] | None = None,
    equality_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    violation_summary = summarize_constraint_violations(evaluations)
    per_constraint_summary = per_constraint_violation_summary(
        evaluations,
        inequality_names=inequality_names,
        equality_names=equality_names,
    )
    equality_rows = [
        row
        for row in per_constraint_summary
        if row.get("constraint_type") == "equality" and row.get("satisfaction_rate") is not None
    ]

    equality_satisfaction_rate = (
        sum(float(row["satisfaction_rate"]) for row in equality_rows) / len(equality_rows)
        if equality_rows
        else None
    )
    equality_mean_violation = (
        sum(float(row["mean_violation"]) for row in equality_rows) / len(equality_rows)
        if equality_rows
        else None
    )
    equality_max_violation = (
        max(float(row["max_violation"]) for row in equality_rows)
        if equality_rows
        else None
    )

    best_total_violation = min(
        (float(evaluation.total_violation) for evaluation in evaluations),
        default=None,
    )

    summary: dict[str, Any] = {
        "feasible_rate": violation_summary.feasible_rate,
        "feasible_count": violation_summary.feasible_count,
        "infeasible_count": violation_summary.infeasible_count,
        "best_total_violation": best_total_violation,
        "mean_total_violation": violation_summary.mean_total_violation,
        "median_total_violation": violation_summary.median_total_violation,
        "min_total_violation": violation_summary.min_total_violation,
        "max_total_violation": violation_summary.max_total_violation,
        "mean_max_violation": violation_summary.mean_max_violation,
        "violation_count_total": violation_summary.violation_count_total,
        "all_feasible": violation_summary.all_feasible,
        "any_feasible": violation_summary.any_feasible,
        "all_infeasible": violation_summary.all_infeasible,
        "feasible_nondominated_count": feasible_nondominated_count(
            objective_vectors,
            evaluations,
            directions=directions,
        ),
        "infeasible_nondominated_count": infeasible_nondominated_count(
            objective_vectors,
            evaluations,
            directions=directions,
        ),
        "feasible_only_HV": feasible_only_hypervolume_2d(
            objective_vectors,
            evaluations,
            directions=directions,
            reference_point=reference_point,
        ),
        "feasible_only_reference_distance": feasible_only_reference_distance(
            objective_vectors,
            evaluations,
            directions=directions,
            reference_front=reference_front,
        ),
        "feasible_only_IGD": feasible_only_igd(
            objective_vectors,
            evaluations,
            directions=directions,
            reference_front=reference_front,
        ),
        "spacing_feasible_only": feasible_only_spacing(
            objective_vectors,
            evaluations,
            directions=directions,
        ),
        "equality_satisfaction_rate": equality_satisfaction_rate,
        "equality_mean_violation": equality_mean_violation,
        "equality_max_violation": equality_max_violation,
        "per_constraint_violation_summary": per_constraint_summary,
        "constraint_summary": violation_summary.to_dict(),
    }
    return summary


__all__ = [
    "constrained_mo_summary_to_dict",
    "feasible_nondominated_count",
    "feasible_only_hypervolume_2d",
    "feasible_only_igd",
    "feasible_only_reference_distance",
    "feasible_only_spacing",
    "infeasible_nondominated_count",
    "summarize_constrained_mo_population",
    "violation_summary_to_dict_mo",
]
