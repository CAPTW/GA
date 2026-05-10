from __future__ import annotations

import json
from pathlib import Path

from ga_lab.experiment.constrained_sphere_smoke import SmokeConfig, run_constrained_sphere_smoke


def test_runner_creates_json_artifact_and_includes_constraint_and_fairness_summary(tmp_path: Path) -> None:
    artifact = run_constrained_sphere_smoke(
        SmokeConfig(
            dimension=3,
            seeds=2,
            budget=20,
            artifact_suffix="runner_smoke_json",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    if not json_path.is_absolute():
        json_path = Path(tmp_path.parent) / json_path

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["rows"]
    assert "constraint_summary" in payload["rows"][0]
    assert "fairness_summary" in payload
    assert "feasible_rate" in payload["rows"][0]
    assert payload["default_changed"] is False


def test_runner_uses_null_for_missing_best_feasible_objective(tmp_path: Path) -> None:
    artifact = run_constrained_sphere_smoke(
        SmokeConfig(
            dimension=5,
            seeds=1,
            budget=20,
            constraint_budget=-30.0,
            artifact_suffix="runner_all_infeasible",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    if not json_path.is_absolute():
        json_path = Path(tmp_path.parent) / json_path
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["rows"][0]["best_feasible_objective"] is None
    assert payload["rows"][0]["all_infeasible"] is True


def test_runner_actual_evaluations_equal_budget_and_suffix_respected(tmp_path: Path) -> None:
    suffix = "runner_budget_check"
    artifact = run_constrained_sphere_smoke(
        SmokeConfig(
            dimension=4,
            seeds=1,
            budget=30,
            artifact_suffix=suffix,
            output_dir=str(tmp_path),
        )
    )

    assert artifact["rows"][0]["actual_evaluations"] == 30
    assert suffix in artifact["artifacts"]["json"]
    assert artifact["ga_integration_done"] is False
