from __future__ import annotations

import json
from pathlib import Path

from ga_lab.experiment.constrained_ga_smoke import (
    ConstrainedGASmokeConfig,
    EQUALITY_PLANE_SMOKE_TOLERANCE,
    run_constrained_ga_smoke,
)


def test_runner_supports_constrained_equality_plane_quadratic(tmp_path: Path) -> None:
    artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            problem="constrained_equality_plane_quadratic",
            dimension=6,
            seeds=2,
            budget=30,
            tolerance=0.1,
            population_size=6,
            artifact_suffix="equality_runner_json",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["problem"] == "constrained_equality_plane_quadratic"
    assert payload["configuration"]["tolerance"] == 0.1
    assert payload["rows"]
    assert {row["strategy"] for row in payload["rows"]} == {
        "random_search_feasibility_first",
        "constrained_ga_feasibility_first",
    }


def test_equality_artifact_contains_fairness_constraints_and_trace(tmp_path: Path) -> None:
    artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            problem="constrained_equality_plane_quadratic",
            dimension=6,
            seeds=2,
            budget=30,
            tolerance=0.1,
            population_size=6,
            artifact_suffix="equality_runner_summary",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["fairness_summary"]["status"] == "pass"
    assert payload["benchmark_contract"]["equality_count"] == 1
    assert payload["benchmark_contract"]["tolerance"] == 0.1
    assert payload["per_constraint_summaries"]
    assert any(
        item["constraint_type"] == "equality"
        and item["constraint_name"] == "plane_sum_target"
        and item["satisfaction_rate"] is not None
        for item in payload["per_constraint_summaries"]
    )
    for row in payload["rows"]:
        assert row["constraint_count"] == 2
        assert row["inequality_count"] == 1
        assert row["equality_count"] == 1
        assert row["tolerance"] == 0.1
        assert "constraint_summary" in row
        assert "per_constraint_violation_summary" in row
        assert row["actual_evaluations"] == row["requested_budget"]
        assert row["equality_satisfaction_rate"] is not None
        assert row["per_constraint_trace_summary"]["records_count"] == row["requested_budget"] * 2


def test_equality_artifact_suffix_default_status_and_no_nan_placeholders(tmp_path: Path) -> None:
    suffix = "equality_runner_suffix"
    artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            problem="constrained_equality_plane_quadratic",
            dimension=6,
            seeds=1,
            budget=24,
            tolerance=0.1,
            population_size=6,
            artifact_suffix=suffix,
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert suffix in artifact["artifacts"]["json"]
    assert artifact["default_changed"] is False
    assert artifact["ga_default_changed"] is False
    assert artifact["nsga2_constraint_domination_done"] is False
    assert "NaN" not in json.dumps(payload, allow_nan=False)


def test_equality_runner_uses_equality_smoke_tolerance_without_changing_sphere_default(
    tmp_path: Path,
) -> None:
    equality_artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            problem="constrained_equality_plane_quadratic",
            dimension=6,
            seeds=1,
            budget=24,
            population_size=6,
            artifact_suffix="equality_default_tolerance",
            output_dir=str(tmp_path),
        )
    )
    equality_payload = json.loads(
        Path(equality_artifact["artifacts"]["json"]).read_text(encoding="utf-8")
    )

    sphere_artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            problem="constrained_sphere",
            dimension=3,
            seeds=1,
            budget=24,
            population_size=6,
            artifact_suffix="sphere_default_tolerance",
            output_dir=str(tmp_path),
        )
    )
    sphere_payload = json.loads(
        Path(sphere_artifact["artifacts"]["json"]).read_text(encoding="utf-8")
    )

    assert equality_payload["configuration"]["tolerance"] == EQUALITY_PLANE_SMOKE_TOLERANCE
    assert sphere_payload["configuration"]["tolerance"] == 1e-8
