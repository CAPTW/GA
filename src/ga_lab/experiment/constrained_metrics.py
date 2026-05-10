from __future__ import annotations

import math
from statistics import mean
from typing import Any, Sequence

from ga_lab.constraints import ConstraintEvaluation, summarize_constraint_violations
from ga_lab.experiment.mo_metrics import validate_objective_vector


def feasible_rate(evaluations: Sequence[ConstraintEvaluation]) -> float | None:
    return summarize_constraint_violations(evaluations).feasible_rate


def best_feasible_objective(
    objective_values: Sequence[float],
    evaluations: Sequence[ConstraintEvaluation],
    *,
    maximize: bool = False,
) -> float | None:
    if len(objective_values) != len(evaluations):
        raise ValueError("objective_values length must match evaluations length")

    feasible_values: list[float] = []
    for index, (value, evaluation) in enumerate(zip(objective_values, evaluations, strict=True)):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(
                "Non-finite objective detected in best_feasible_objective "
                f"(index={index}, value={value!r})"
            )
        if evaluation.feasible:
            feasible_values.append(normalized)

    if not feasible_values:
        return None
    return max(feasible_values) if maximize else min(feasible_values)


def best_feasible_objectives(
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    directions: Sequence[bool] | None = None,
) -> list[float] | None:
    if len(objective_vectors) != len(evaluations):
        raise ValueError("objective_vectors length must match evaluations length")
    if not objective_vectors:
        return None

    objective_count = len(objective_vectors[0])
    normalized_directions = list(directions) if directions is not None else [False] * objective_count
    if len(normalized_directions) != objective_count:
        raise ValueError("directions length must match objective vector length")

    feasible_vectors: list[list[float]] = []
    for index, (vector, evaluation) in enumerate(zip(objective_vectors, evaluations, strict=True)):
        if len(vector) != objective_count:
            raise ValueError("objective vector length must be consistent")
        validate_objective_vector(vector, label=f"best_feasible_objectives[{index}]")
        if evaluation.feasible:
            feasible_vectors.append([float(value) for value in vector])

    if not feasible_vectors:
        return None

    summary: list[float] = []
    for objective_index, maximize in enumerate(normalized_directions):
        values = [vector[objective_index] for vector in feasible_vectors]
        summary.append(max(values) if maximize else min(values))
    return summary


def violation_summary_to_dict(summary) -> dict[str, Any]:
    return summary.to_dict()


def summarize_constrained_population(
    evaluations: Sequence[ConstraintEvaluation],
) -> dict[str, Any]:
    return violation_summary_to_dict(summarize_constraint_violations(evaluations))


def per_constraint_violation_summary(
    evaluations: Sequence[ConstraintEvaluation],
    *,
    inequality_names: Sequence[str] | None = None,
    equality_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if not evaluations:
        return []

    inequality_count = len(evaluations[0].inequality_violations)
    equality_count = len(evaluations[0].equality_violations)
    for evaluation in evaluations[1:]:
        if len(evaluation.inequality_violations) != inequality_count:
            raise ValueError("inequality violation vector length must be consistent")
        if len(evaluation.equality_violations) != equality_count:
            raise ValueError("equality violation vector length must be consistent")

    inequality_labels = (
        list(inequality_names)
        if inequality_names is not None
        else [f"inequality_{index}" for index in range(inequality_count)]
    )
    equality_labels = (
        list(equality_names)
        if equality_names is not None
        else [f"equality_{index}" for index in range(equality_count)]
    )
    if len(inequality_labels) != inequality_count:
        raise ValueError("inequality_names length must match inequality violation length")
    if len(equality_labels) != equality_count:
        raise ValueError("equality_names length must match equality violation length")

    rows: list[dict[str, Any]] = []
    for index, label in enumerate(inequality_labels):
        values = [evaluation.inequality_violations[index] for evaluation in evaluations]
        rows.append(
            {
                "constraint": str(label),
                "constraint_type": "inequality",
                "index": index,
                "satisfaction_count": sum(1 for value in values if value <= 0.0),
                "satisfaction_rate": sum(1 for value in values if value <= 0.0) / len(values),
                "mean_violation": float(mean(values)),
                "max_violation": max(values),
            }
        )
    for index, label in enumerate(equality_labels):
        values = [evaluation.equality_violations[index] for evaluation in evaluations]
        rows.append(
            {
                "constraint": str(label),
                "constraint_type": "equality",
                "index": index,
                "satisfaction_count": sum(1 for value in values if value <= 0.0),
                "satisfaction_rate": sum(1 for value in values if value <= 0.0) / len(values),
                "mean_violation": float(mean(values)),
                "max_violation": max(values),
            }
        )
    return rows


__all__ = [
    "best_feasible_objective",
    "best_feasible_objectives",
    "feasible_rate",
    "per_constraint_violation_summary",
    "summarize_constrained_population",
    "violation_summary_to_dict",
]
