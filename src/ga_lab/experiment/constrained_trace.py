from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Sequence

from ga_lab.constraints import ConstraintEvaluation
from ga_lab.experiment.constrained_protocol import ConstrainedCandidateRecord


@dataclass(slots=True)
class PerConstraintTraceRecord:
    strategy: str
    seed: int | None
    evaluation_index: int
    feasible: bool
    objective: float
    total_violation: float
    max_violation: float
    constraint_name: str
    constraint_type: str
    constraint_value: float
    constraint_violation: float
    satisfied: bool
    variant_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "strategy": self.strategy,
            "seed": self.seed,
            "evaluation_index": self.evaluation_index,
            "feasible": self.feasible,
            "objective": self.objective,
            "total_violation": self.total_violation,
            "max_violation": self.max_violation,
            "constraint_name": self.constraint_name,
            "constraint_type": self.constraint_type,
            "constraint_value": self.constraint_value,
            "constraint_violation": self.constraint_violation,
            "satisfied": self.satisfied,
            "metadata": dict(self.metadata),
        }
        if self.variant_name is not None:
            payload["variant_name"] = self.variant_name
        return payload


@dataclass(slots=True)
class PerConstraintAggregate:
    constraint_name: str
    constraint_type: str
    index: int
    satisfaction_rate: float
    mean_value: float
    mean_violation: float
    max_violation: float
    violation_count: int
    evaluated_count: int
    satisfaction_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint_name,
            "constraint_name": self.constraint_name,
            "constraint_type": self.constraint_type,
            "index": self.index,
            "satisfaction_count": self.satisfaction_count,
            "satisfaction_rate": self.satisfaction_rate,
            "mean_value": self.mean_value,
            "mean_violation": self.mean_violation,
            "max_violation": self.max_violation,
            "violation_count": self.violation_count,
            "evaluated_count": self.evaluated_count,
            "source": "constraint_evaluation_trace",
        }


@dataclass(slots=True)
class ConstraintTraceSummary:
    records_count: int
    constraints: list[str]
    aggregates: list[PerConstraintAggregate]
    records_sample: list[PerConstraintTraceRecord]
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_count": self.records_count,
            "constraints": list(self.constraints),
            "aggregates": [item.to_dict() for item in self.aggregates],
            "records_sample": [item.to_dict() for item in self.records_sample],
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
        }


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite numeric value, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return normalized


def _constraint_labels(
    *,
    constraint_names: Sequence[str] | None,
    inequality_count: int,
    equality_count: int,
) -> tuple[list[str], list[str], list[str]]:
    total_count = inequality_count + equality_count
    warnings: list[str] = []
    if constraint_names is None:
        warnings.append("constraint names missing; deterministic fallback names were used")
        names = [f"constraint_{index}" for index in range(total_count)]
    else:
        names = [str(item) for item in constraint_names]
        if len(names) != total_count:
            warnings.append(
                "constraint name count did not match evaluation vector length; "
                "deterministic fallback names were used"
            )
            names = [f"constraint_{index}" for index in range(total_count)]
    return names[:inequality_count], names[inequality_count:], warnings


def _validate_evaluation_shape(
    evaluation: ConstraintEvaluation,
    *,
    expected_inequality_count: int,
    expected_equality_count: int,
    evaluation_index: int,
) -> None:
    if len(evaluation.inequality_values) != expected_inequality_count:
        raise ValueError(
            "inequality value vector length must be consistent "
            f"(evaluation_index={evaluation_index})"
        )
    if len(evaluation.inequality_violations) != expected_inequality_count:
        raise ValueError(
            "inequality violation vector length must be consistent "
            f"(evaluation_index={evaluation_index})"
        )
    if len(evaluation.equality_values) != expected_equality_count:
        raise ValueError(
            "equality value vector length must be consistent "
            f"(evaluation_index={evaluation_index})"
        )
    if len(evaluation.equality_violations) != expected_equality_count:
        raise ValueError(
            "equality violation vector length must be consistent "
            f"(evaluation_index={evaluation_index})"
        )


def build_per_constraint_trace_records(
    records: Sequence[ConstrainedCandidateRecord],
    *,
    strategy: str,
    constraint_names: Sequence[str] | None = None,
    variant_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[PerConstraintTraceRecord], list[str]]:
    if not records:
        return [], []

    first_evaluation = records[0].constraint_evaluation
    inequality_count = len(first_evaluation.inequality_violations)
    equality_count = len(first_evaluation.equality_violations)
    inequality_names, equality_names, warnings = _constraint_labels(
        constraint_names=constraint_names,
        inequality_count=inequality_count,
        equality_count=equality_count,
    )
    base_metadata = dict(metadata or {})

    trace_records: list[PerConstraintTraceRecord] = []
    for record in records:
        evaluation = record.constraint_evaluation
        _validate_evaluation_shape(
            evaluation,
            expected_inequality_count=inequality_count,
            expected_equality_count=equality_count,
            evaluation_index=record.evaluation_index,
        )
        objective = _finite_float(record.objective, label="objective")
        total_violation = _finite_float(
            evaluation.total_violation,
            label="constraint_total_violation",
        )
        max_violation = _finite_float(
            evaluation.max_violation,
            label="constraint_max_violation",
        )
        row_metadata = {**base_metadata, **dict(record.metadata)}

        for index, (name, value, violation) in enumerate(
            zip(
                inequality_names,
                evaluation.inequality_values,
                evaluation.inequality_violations,
                strict=True,
            )
        ):
            normalized_violation = _finite_float(
                violation,
                label=f"inequality_violation[{index}]",
            )
            trace_records.append(
                PerConstraintTraceRecord(
                    strategy=strategy,
                    seed=record.seed,
                    evaluation_index=record.evaluation_index,
                    feasible=bool(evaluation.feasible),
                    objective=objective,
                    total_violation=total_violation,
                    max_violation=max_violation,
                    constraint_name=name,
                    constraint_type="inequality",
                    constraint_value=_finite_float(
                        value,
                        label=f"inequality_value[{index}]",
                    ),
                    constraint_violation=normalized_violation,
                    satisfied=normalized_violation <= 0.0,
                    variant_name=variant_name,
                    metadata=row_metadata,
                )
            )

        for index, (name, value, violation) in enumerate(
            zip(
                equality_names,
                evaluation.equality_values,
                evaluation.equality_violations,
                strict=True,
            )
        ):
            normalized_violation = _finite_float(
                violation,
                label=f"equality_violation[{index}]",
            )
            trace_records.append(
                PerConstraintTraceRecord(
                    strategy=strategy,
                    seed=record.seed,
                    evaluation_index=record.evaluation_index,
                    feasible=bool(evaluation.feasible),
                    objective=objective,
                    total_violation=total_violation,
                    max_violation=max_violation,
                    constraint_name=name,
                    constraint_type="equality",
                    constraint_value=_finite_float(
                        value,
                        label=f"equality_value[{index}]",
                    ),
                    constraint_violation=normalized_violation,
                    satisfied=normalized_violation <= 0.0,
                    variant_name=variant_name,
                    metadata=row_metadata,
                )
            )

    return trace_records, warnings


def summarize_per_constraint_trace(
    trace_records: Sequence[PerConstraintTraceRecord],
    *,
    warnings: Sequence[str] | None = None,
    sample_limit: int = 20,
) -> ConstraintTraceSummary:
    if sample_limit < 0:
        raise ValueError("sample_limit must be >= 0")

    buckets: dict[tuple[str, str], list[PerConstraintTraceRecord]] = {}
    order: list[tuple[str, str]] = []
    for record in trace_records:
        key = (record.constraint_type, record.constraint_name)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(record)

    aggregates: list[PerConstraintAggregate] = []
    for index, key in enumerate(order):
        entries = buckets[key]
        values = [entry.constraint_value for entry in entries]
        violations = [entry.constraint_violation for entry in entries]
        satisfaction_count = sum(1 for entry in entries if entry.satisfied)
        aggregates.append(
            PerConstraintAggregate(
                constraint_name=key[1],
                constraint_type=key[0],
                index=index,
                satisfaction_rate=satisfaction_count / len(entries),
                mean_value=float(mean(values)),
                mean_violation=float(mean(violations)),
                max_violation=max(violations),
                violation_count=sum(1 for value in violations if value > 0.0),
                evaluated_count=len(entries),
                satisfaction_count=satisfaction_count,
            )
        )

    limitations: list[str] = []
    if len(trace_records) > sample_limit:
        limitations.append(
            "raw per-constraint trace is sampled in artifact; aggregates use all records"
        )

    return ConstraintTraceSummary(
        records_count=len(trace_records),
        constraints=[item.constraint_name for item in aggregates],
        aggregates=aggregates,
        records_sample=list(trace_records[:sample_limit]),
        limitations=limitations,
        warnings=[str(item) for item in (warnings or [])],
    )


def per_constraint_summary_to_dict(summary: ConstraintTraceSummary) -> dict[str, Any]:
    return summary.to_dict()


def merge_per_constraint_summaries(
    summaries: Sequence[ConstraintTraceSummary | dict[str, Any]],
) -> dict[str, Any]:
    records_count = 0
    constraints: list[str] = []
    aggregates: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    limitations: list[str] = []
    warnings: list[str] = []

    for summary in summaries:
        payload = summary.to_dict() if isinstance(summary, ConstraintTraceSummary) else dict(summary)
        records_count += int(payload.get("records_count") or 0)
        for constraint in payload.get("constraints", []):
            value = str(constraint)
            if value not in constraints:
                constraints.append(value)
        aggregates.extend(dict(item) for item in payload.get("aggregates", []))
        samples.extend(dict(item) for item in payload.get("records_sample", []))
        limitations.extend(str(item) for item in payload.get("limitations", []))
        warnings.extend(str(item) for item in payload.get("warnings", []))

    return {
        "records_count": records_count,
        "constraints": constraints,
        "aggregates": aggregates,
        "records_sample": samples,
        "limitations": limitations,
        "warnings": warnings,
    }


__all__ = [
    "ConstraintTraceSummary",
    "PerConstraintAggregate",
    "PerConstraintTraceRecord",
    "build_per_constraint_trace_records",
    "merge_per_constraint_summaries",
    "per_constraint_summary_to_dict",
    "summarize_per_constraint_trace",
]
