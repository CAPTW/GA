from __future__ import annotations

import json
from pathlib import Path

from ga_lab.experiment.constrained_nsga2_smoke import (
    ConstrainedNSGA2SmokeConfig,
    run_constrained_nsga2_smoke,
)


def test_runner_creates_json_artifact(tmp_path: Path) -> None:
    artifact = run_constrained_nsga2_smoke(
        ConstrainedNSGA2SmokeConfig(
            dimension=6,
            seeds=1,
            budget=20,
            population_size=4,
            artifact_suffix="constrained_nsga2_smoke_runner_json",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    assert json_path.exists()


def test_artifact_includes_fairness_constraint_summary_and_feasible_only_fields(tmp_path: Path) -> None:
    artifact = run_constrained_nsga2_smoke(
        ConstrainedNSGA2SmokeConfig(
            dimension=6,
            seeds=1,
            budget=20,
            population_size=4,
            artifact_suffix="constrained_nsga2_smoke_runner_summary",
            output_dir=str(tmp_path),
        )
    )

    payload = json.loads(Path(artifact["artifacts"]["json"]).read_text(encoding="utf-8"))

    assert payload["fairness_summary"]["status"] == "pass"
    assert payload["per_constraint_summaries"]
    assert {row["strategy"] for row in payload["rows"]} == {
        "constrained_nsga2_constraint_domination",
        "random_pareto_archive",
    }
    for row in payload["rows"]:
        assert "constraint_summary" in row
        assert "feasible_only_HV" in row
        assert "feasible_only_reference_distance" in row
        assert row["actual_evaluations"] == row["requested_budget"]


def test_artifact_suffix_and_no_default_path_contamination(tmp_path: Path) -> None:
    suffix = "constrained_nsga2_smoke_runner_suffix"
    artifact = run_constrained_nsga2_smoke(
        ConstrainedNSGA2SmokeConfig(
            dimension=6,
            seeds=1,
            budget=20,
            population_size=4,
            artifact_suffix=suffix,
            output_dir=str(tmp_path),
        )
    )

    payload = json.loads(Path(artifact["artifacts"]["json"]).read_text(encoding="utf-8"))

    assert suffix in artifact["artifacts"]["json"]
    assert payload["default_changed"] is False
    assert payload["nsga2_default_changed"] is False
    assert payload["nsga2_constraint_domination_done"] is True


def test_runner_supports_constrained_dtlz_box_toy(tmp_path: Path) -> None:
    suffix = "constrained_nsga2_dtlz_smoke_runner"
    artifact = run_constrained_nsga2_smoke(
        ConstrainedNSGA2SmokeConfig(
            problem="constrained_dtlz_box_toy",
            dimension=7,
            seeds=1,
            budget=20,
            population_size=4,
            artifact_suffix=suffix,
            output_dir=str(tmp_path),
        )
    )

    payload = json.loads(Path(artifact["artifacts"]["json"]).read_text(encoding="utf-8"))

    assert payload["problem"] == "constrained_dtlz_box_toy"
    assert Path(artifact["artifacts"]["json"]).name.startswith(
        "constrained_nsga2_dtlz_smoke_results_"
    )
    assert suffix in artifact["artifacts"]["json"]
    assert payload["fairness_summary"]["status"] == "pass"
    assert payload["per_constraint_summaries"]
    for row in payload["rows"]:
        assert row["problem"] == "constrained_dtlz_box_toy"
        assert "constraint_summary" in row
        assert "feasible_only_HV" in row
        assert "feasible_only_reference_distance" in row
        assert row["actual_evaluations"] == row["requested_budget"]
