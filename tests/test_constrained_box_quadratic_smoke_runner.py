from __future__ import annotations

import json
from pathlib import Path

from ga_lab.experiment.constrained_ga_smoke import (
    ConstrainedGASmokeConfig,
    run_constrained_ga_smoke,
)


def test_runner_supports_constrained_box_quadratic_and_creates_json_artifact(tmp_path: Path) -> None:
    artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            problem="constrained_box_quadratic",
            dimension=6,
            seeds=2,
            budget=20,
            population_size=5,
            artifact_suffix="constrained_box_quadratic_runner_json",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["problem"] == "constrained_box_quadratic"
    assert payload["rows"]
    assert payload["rows"][0]["constraint_count"] == 2
    assert payload["rows"][0]["inequality_count"] == 2
    assert payload["rows"][0]["equality_count"] == 0


def test_box_quadratic_artifact_includes_both_strategies_fairness_and_constraint_summaries(tmp_path: Path) -> None:
    artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            problem="constrained_box_quadratic",
            dimension=6,
            seeds=2,
            budget=20,
            population_size=5,
            artifact_suffix="constrained_box_quadratic_runner_summary",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    strategies = {row["strategy"] for row in payload["rows"]}
    assert strategies == {
        "random_search_feasibility_first",
        "constrained_ga_feasibility_first",
    }
    assert payload["fairness_summary"]["status"] == "pass"
    assert "constraint_summary" in payload["rows"][0]
    assert "per_constraint_violation_summary" in payload["rows"][0]
    assert payload["per_constraint_summaries"]
    constrained_row = next(
        row for row in payload["rows"] if row["strategy"] == "constrained_ga_feasibility_first"
    )
    constrained_summary = constrained_row["per_constraint_violation_summary"]
    assert {item["constraint_name"] for item in constrained_summary} == {
        "group1_budget",
        "group2_budget",
    }
    assert all(item["satisfaction_rate"] is not None for item in constrained_summary)
    assert constrained_row["per_constraint_trace_summary"]["records_count"] == 40
    assert constrained_row["per_constraint_trace_records_sample"]


def test_box_quadratic_artifact_respects_suffix_and_budget_and_keeps_default_unchanged(tmp_path: Path) -> None:
    suffix = "constrained_box_quadratic_runner_budget"
    artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            problem="constrained_box_quadratic",
            dimension=6,
            seeds=1,
            budget=30,
            population_size=6,
            artifact_suffix=suffix,
            output_dir=str(tmp_path),
        )
    )

    for row in artifact["rows"]:
        assert row["actual_evaluations"] == 30
    assert suffix in artifact["artifacts"]["json"]
    assert artifact["default_changed"] is False
    assert artifact["ga_default_changed"] is False
