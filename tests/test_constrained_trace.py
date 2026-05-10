from __future__ import annotations

import json
import math

import pytest

from ga_lab.constraints import ConstraintEvaluation, evaluate_constraint_violations
from ga_lab.experiment.constrained_protocol import ConstrainedCandidateRecord
from ga_lab.experiment.constrained_trace import (
    build_per_constraint_trace_records,
    per_constraint_summary_to_dict,
    summarize_per_constraint_trace,
)


def _record(
    *,
    objective: float,
    values: list[float],
    seed: int = 0,
    evaluation_index: int = 0,
) -> ConstrainedCandidateRecord:
    return ConstrainedCandidateRecord(
        solution=[0.0, 0.0],
        objective=objective,
        constraint_evaluation=evaluate_constraint_violations(
            inequality_values=values,
            equality_values=[],
        ),
        seed=seed,
        evaluation_index=evaluation_index,
        metadata={"fixture": "trace"},
    )


def test_per_constraint_trace_distinguishes_group_constraints_and_satisfaction() -> None:
    records = [_record(objective=1.0, values=[-1.0, 2.0])]

    trace, warnings = build_per_constraint_trace_records(
        records,
        strategy="constrained_ga_feasibility_first",
        constraint_names=["group1_budget", "group2_budget"],
    )

    assert warnings == []
    assert [item.constraint_name for item in trace] == ["group1_budget", "group2_budget"]
    assert [item.satisfied for item in trace] == [True, False]
    assert [item.constraint_violation for item in trace] == [0.0, 2.0]


def test_per_constraint_trace_aggregate_calculates_rates_and_violations() -> None:
    records = [
        _record(objective=1.0, values=[-1.0, 2.0], evaluation_index=0),
        _record(objective=2.0, values=[3.0, -2.0], evaluation_index=1),
    ]
    trace, warnings = build_per_constraint_trace_records(
        records,
        strategy="random_search_feasibility_first",
        constraint_names=["group1_budget", "group2_budget"],
    )

    summary = summarize_per_constraint_trace(trace, warnings=warnings, sample_limit=1)
    payload = per_constraint_summary_to_dict(summary)
    by_name = {item["constraint_name"]: item for item in payload["aggregates"]}

    assert payload["records_count"] == 4
    assert len(payload["records_sample"]) == 1
    assert payload["limitations"] == [
        "raw per-constraint trace is sampled in artifact; aggregates use all records"
    ]
    assert by_name["group1_budget"]["satisfaction_rate"] == 0.5
    assert by_name["group1_budget"]["mean_violation"] == 1.5
    assert by_name["group1_budget"]["max_violation"] == 3.0
    assert by_name["group2_budget"]["satisfaction_rate"] == 0.5
    assert by_name["group2_budget"]["mean_violation"] == 1.0
    assert by_name["group2_budget"]["max_violation"] == 2.0


def test_missing_constraint_names_use_deterministic_fallback_and_json_serializes() -> None:
    records = [_record(objective=1.0, values=[-1.0, 2.0])]

    trace, warnings = build_per_constraint_trace_records(
        records,
        strategy="constrained_ga_feasibility_first",
    )
    payload = per_constraint_summary_to_dict(
        summarize_per_constraint_trace(trace, warnings=warnings)
    )

    assert payload["constraints"] == ["constraint_0", "constraint_1"]
    assert payload["warnings"] == [
        "constraint names missing; deterministic fallback names were used"
    ]
    assert "NaN" not in json.dumps(payload, allow_nan=False)


def test_non_finite_constraint_trace_value_fails_fast() -> None:
    evaluation = ConstraintEvaluation(
        feasible=False,
        inequality_values=[math.inf],
        equality_values=[],
        inequality_violations=[math.inf],
        equality_violations=[],
        total_violation=math.inf,
        max_violation=math.inf,
        violation_count=1,
        tolerance={"inequality": 0.0, "equality": 1e-8},
    )
    record = ConstrainedCandidateRecord(
        solution=[0.0],
        objective=1.0,
        constraint_evaluation=evaluation,
        seed=0,
        evaluation_index=0,
    )

    with pytest.raises(ValueError, match="must be finite"):
        build_per_constraint_trace_records(
            [record],
            strategy="constrained_ga_feasibility_first",
            constraint_names=["group1_budget"],
        )
