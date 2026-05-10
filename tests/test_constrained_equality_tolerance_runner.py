from __future__ import annotations

import json
from pathlib import Path

from ga_lab.experiment.constrained_equality_tolerance_stress import (
    ConstrainedEqualityToleranceStressConfig,
    run_constrained_equality_tolerance_stress,
)
from ga_lab.problems.constrained_equality_plane_quadratic import (
    ConstrainedEqualityPlaneQuadraticProblem,
)


def _run_tiny_tolerance(tmp_path: Path, suffix: str = "equality_tolerance_test") -> dict:
    return run_constrained_equality_tolerance_stress(
        ConstrainedEqualityToleranceStressConfig(
            variants=("loose", "default", "strict"),
            dimension=6,
            seeds=2,
            budgets=(24, 30),
            population_size=6,
            artifact_suffix=suffix,
            output_dir=str(tmp_path),
        )
    )


def test_tolerance_runner_accepts_loose_default_strict_variants(tmp_path: Path) -> None:
    artifact = _run_tiny_tolerance(tmp_path)

    assert artifact["configuration"]["variants"] == ["loose", "default", "strict"]
    assert {row["variant_name"] for row in artifact["rows"]} == {"loose", "default", "strict"}


def test_tolerance_variant_metadata_includes_tolerance_and_default_matches_problem(
    tmp_path: Path,
) -> None:
    artifact = _run_tiny_tolerance(tmp_path, "equality_tolerance_metadata")
    default_problem = ConstrainedEqualityPlaneQuadraticProblem(dimension=6)
    default_rows = [row for row in artifact["rows"] if row["variant_name"] == "default"]

    assert default_rows
    assert {row["tolerance"] for row in default_rows} == {default_problem.equality_tolerance}
    for definition in artifact["variant_definitions"]:
        assert "tolerance" in definition


def test_tolerance_artifact_includes_summaries_pairing_and_fairness(
    tmp_path: Path,
) -> None:
    artifact = _run_tiny_tolerance(tmp_path, "equality_tolerance_artifact")
    payload = json.loads(Path(artifact["artifacts"]["json"]).read_text(encoding="utf-8"))

    assert payload["variant_budget_strategy_summaries"]
    assert payload["equality_summaries"]
    assert payload["per_constraint_summaries"]
    assert payload["paired_comparisons"]
    assert payload["fairness_summary"]["status"] == "pass"
    assert {row["strategy"] for row in payload["rows"]} == {
        "random_search_feasibility_first",
        "constrained_ga_feasibility_first",
    }
    assert any(
        item["constraint_name"] == "plane_sum_target"
        and item["constraint_type"] == "equality"
        and item["satisfaction_rate"] is not None
        for item in payload["per_constraint_summaries"]
    )


def test_tolerance_actual_evaluations_equal_requested_budget_for_each_run(
    tmp_path: Path,
) -> None:
    artifact = _run_tiny_tolerance(tmp_path, "equality_tolerance_budget")

    for row in artifact["rows"]:
        assert row["actual_evaluations"] == row["requested_budget"]


def test_tolerance_suffix_respected_and_default_not_changed(tmp_path: Path) -> None:
    suffix = "equality_tolerance_suffix"
    artifact = _run_tiny_tolerance(tmp_path, suffix)

    assert suffix in artifact["artifacts"]["json"]
    assert artifact["default_changed"] is False
    assert artifact["ga_default_changed"] is False
    assert artifact["nsga2_constraint_domination_done"] is False
    assert artifact["penalty_repair_done"] is False


def test_tolerance_artifact_has_no_nan_placeholders(tmp_path: Path) -> None:
    artifact = _run_tiny_tolerance(tmp_path, "equality_tolerance_no_nan")
    payload = json.loads(Path(artifact["artifacts"]["json"]).read_text(encoding="utf-8"))

    assert "NaN" not in json.dumps(payload, allow_nan=False)
