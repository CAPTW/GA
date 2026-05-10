from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Sequence


DEFAULT_EQUALITY_TOLERANCE = 1e-8


@dataclass(slots=True)
class ConstraintEvaluation:
    feasible: bool
    inequality_values: list[float]
    equality_values: list[float]
    inequality_violations: list[float]
    equality_violations: list[float]
    total_violation: float
    max_violation: float
    violation_count: int
    tolerance: dict[str, float | list[float]]
    metadata: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "inequality_values": list(self.inequality_values),
            "equality_values": list(self.equality_values),
            "inequality_violations": list(self.inequality_violations),
            "equality_violations": list(self.equality_violations),
            "total_violation": self.total_violation,
            "max_violation": self.max_violation,
            "violation_count": self.violation_count,
            "tolerance": dict(self.tolerance),
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ConstraintViolationSummary:
    feasible_count: int
    infeasible_count: int
    feasible_rate: float | None
    mean_total_violation: float | None
    median_total_violation: float | None
    min_total_violation: float | None
    max_total_violation: float | None
    mean_max_violation: float | None
    violation_count_total: int
    all_feasible: bool
    any_feasible: bool
    all_infeasible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible_count": self.feasible_count,
            "infeasible_count": self.infeasible_count,
            "feasible_rate": self.feasible_rate,
            "mean_total_violation": self.mean_total_violation,
            "median_total_violation": self.median_total_violation,
            "min_total_violation": self.min_total_violation,
            "max_total_violation": self.max_total_violation,
            "mean_max_violation": self.mean_max_violation,
            "violation_count_total": self.violation_count_total,
            "all_feasible": self.all_feasible,
            "any_feasible": self.any_feasible,
            "all_infeasible": self.all_infeasible,
        }


def _coerce_finite_float(value: object, *, label: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"Non-numeric constraint value detected in {label} "
            f"(constraint_index={index}, value={value!r})"
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(
            f"Non-finite constraint value detected in {label} "
            f"(constraint_index={index}, value={value!r})"
        )
    return normalized


def finite_float_list(values: Sequence[object], *, label: str) -> list[float]:
    return [
        _coerce_finite_float(value, label=label, index=index)
        for index, value in enumerate(values)
    ]


def validate_constraint_values(values: Sequence[object], *, label: str) -> list[float]:
    return finite_float_list(values, label=label)


def _normalize_equality_tolerances(
    equality_count: int,
    equality_tolerance: float | Sequence[object],
) -> tuple[list[float], float | list[float]]:
    if equality_count == 0:
        if isinstance(equality_tolerance, Sequence) and not isinstance(
            equality_tolerance, str | bytes
        ):
            normalized = finite_float_list(equality_tolerance, label="equality_tolerance")
            for index, value in enumerate(normalized):
                if value < 0.0:
                    raise ValueError(
                        "equality_tolerance must be >= 0 "
                        f"(constraint_index={index}, value={value!r})"
                    )
            return [], normalized
        scalar = _coerce_finite_float(
            equality_tolerance,
            label="equality_tolerance",
            index=0,
        )
        if scalar < 0.0:
            raise ValueError(f"equality_tolerance must be >= 0 (value={scalar!r})")
        return [], scalar

    if isinstance(equality_tolerance, Sequence) and not isinstance(
        equality_tolerance, str | bytes
    ):
        tolerances = finite_float_list(equality_tolerance, label="equality_tolerance")
        if len(tolerances) != equality_count:
            raise ValueError("equality_tolerance length must match equality_values length")
        for index, value in enumerate(tolerances):
            if value < 0.0:
                raise ValueError(
                    "equality_tolerance must be >= 0 "
                    f"(constraint_index={index}, value={value!r})"
                )
        return tolerances, tolerances

    scalar = _coerce_finite_float(
        equality_tolerance,
        label="equality_tolerance",
        index=0,
    )
    if scalar < 0.0:
        raise ValueError(f"equality_tolerance must be >= 0 (value={scalar!r})")
    return [scalar] * equality_count, scalar


def evaluate_constraint_violations(
    *,
    inequality_values: Sequence[object] | None = None,
    equality_values: Sequence[object] | None = None,
    equality_tolerance: float | Sequence[object] = DEFAULT_EQUALITY_TOLERANCE,
    metadata: dict[str, object] | None = None,
    warnings: Sequence[str] | None = None,
) -> ConstraintEvaluation:
    normalized_inequality_values = validate_constraint_values(
        [] if inequality_values is None else inequality_values,
        label="inequality_values",
    )
    normalized_equality_values = validate_constraint_values(
        [] if equality_values is None else equality_values,
        label="equality_values",
    )
    equality_tolerances, stored_tolerance = _normalize_equality_tolerances(
        len(normalized_equality_values),
        equality_tolerance,
    )

    inequality_violations = [max(0.0, value) for value in normalized_inequality_values]
    equality_violations = [
        max(0.0, abs(value) - tolerance)
        for value, tolerance in zip(
            normalized_equality_values,
            equality_tolerances,
            strict=True,
        )
    ]
    all_violations = [*inequality_violations, *equality_violations]
    total_violation = float(sum(all_violations))
    max_violation = max(all_violations) if all_violations else 0.0
    violation_count = sum(1 for value in all_violations if value > 0.0)

    return ConstraintEvaluation(
        feasible=violation_count == 0,
        inequality_values=normalized_inequality_values,
        equality_values=normalized_equality_values,
        inequality_violations=inequality_violations,
        equality_violations=equality_violations,
        total_violation=total_violation,
        max_violation=max_violation,
        violation_count=violation_count,
        tolerance={
            "inequality": 0.0,
            "equality": stored_tolerance,
        },
        metadata=dict(metadata or {}),
        warnings=[str(item) for item in (warnings or [])],
    )


def summarize_constraint_violations(
    evaluations: Sequence[ConstraintEvaluation],
) -> ConstraintViolationSummary:
    total = len(evaluations)
    if total == 0:
        return ConstraintViolationSummary(
            feasible_count=0,
            infeasible_count=0,
            feasible_rate=None,
            mean_total_violation=None,
            median_total_violation=None,
            min_total_violation=None,
            max_total_violation=None,
            mean_max_violation=None,
            violation_count_total=0,
            all_feasible=False,
            any_feasible=False,
            all_infeasible=False,
        )

    feasible_count = sum(1 for evaluation in evaluations if evaluation.feasible)
    infeasible_count = total - feasible_count
    total_violations = [evaluation.total_violation for evaluation in evaluations]
    max_violations = [evaluation.max_violation for evaluation in evaluations]

    return ConstraintViolationSummary(
        feasible_count=feasible_count,
        infeasible_count=infeasible_count,
        feasible_rate=feasible_count / total,
        mean_total_violation=mean(total_violations),
        median_total_violation=median(total_violations),
        min_total_violation=min(total_violations),
        max_total_violation=max(total_violations),
        mean_max_violation=mean(max_violations),
        violation_count_total=sum(
            evaluation.violation_count for evaluation in evaluations
        ),
        all_feasible=feasible_count == total,
        any_feasible=feasible_count > 0,
        all_infeasible=infeasible_count == total,
    )


def is_feasible(evaluation: ConstraintEvaluation) -> bool:
    return evaluation.feasible


__all__ = [
    "DEFAULT_EQUALITY_TOLERANCE",
    "ConstraintEvaluation",
    "ConstraintViolationSummary",
    "evaluate_constraint_violations",
    "finite_float_list",
    "is_feasible",
    "summarize_constraint_violations",
    "validate_constraint_values",
]
