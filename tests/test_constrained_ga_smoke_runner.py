from __future__ import annotations

import json
from pathlib import Path

from ga_lab.experiment.constrained_ga_smoke import (
    ConstrainedGASmokeConfig,
    run_constrained_ga_smoke,
)


def test_runner_creates_json_artifact_and_includes_both_strategies(tmp_path: Path) -> None:
    artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            dimension=3,
            seeds=2,
            budget=20,
            population_size=5,
            artifact_suffix="constrained_ga_runner_json",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    strategies = {row["strategy"] for row in payload["rows"]}
    assert "random_search_feasibility_first" in strategies
    assert "constrained_ga_feasibility_first" in strategies
    assert "fairness_summary" in payload
    assert "constraint_summary" in payload["rows"][0]
    assert payload["configuration"]["tolerance"] == 1e-8
    assert payload["default_changed"] is False
    assert payload["ga_default_changed"] is False


def test_runner_uses_null_for_missing_best_feasible_objective(tmp_path: Path) -> None:
    artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            dimension=5,
            seeds=1,
            budget=20,
            population_size=5,
            constraint_budget=-30.0,
            artifact_suffix="constrained_ga_all_infeasible",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    for row in payload["rows"]:
        assert row["best_feasible_objective"] is None


def test_runner_actual_evaluations_equal_budget_and_suffix_respected(tmp_path: Path) -> None:
    suffix = "constrained_ga_budget_check"
    artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            dimension=4,
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
    assert artifact["constrained_ga_opt_in_path_status"] == "implemented_explicit_opt_in_only"
