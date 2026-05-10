from __future__ import annotations

import json
from pathlib import Path

from ga_lab.experiment.constrained_box_quadratic_tightness import (
    ConstrainedBoxQuadraticTightnessConfig,
    run_constrained_box_quadratic_tightness_stress,
)
from ga_lab.problems.constrained_box_quadratic import ConstrainedBoxQuadraticProblem


def _run_tiny_tightness(tmp_path: Path, suffix: str = "tightness_runner_test") -> dict:
    return run_constrained_box_quadratic_tightness_stress(
        ConstrainedBoxQuadraticTightnessConfig(
            variants=("easy", "default", "strict"),
            dimension=6,
            seeds=2,
            budget=24,
            population_size=6,
            artifact_suffix=suffix,
            output_dir=str(tmp_path),
        )
    )


def test_tightness_runner_accepts_easy_default_strict_variants(tmp_path: Path) -> None:
    artifact = _run_tiny_tightness(tmp_path)

    assert artifact["configuration"]["variants"] == ["easy", "default", "strict"]
    assert {row["variant_name"] for row in artifact["rows"]} == {"easy", "default", "strict"}


def test_tightness_variant_metadata_includes_budgets_and_default_matches_problem(
    tmp_path: Path,
) -> None:
    artifact = _run_tiny_tightness(tmp_path, "tightness_runner_metadata")
    default_problem = ConstrainedBoxQuadraticProblem(dimension=6)
    default_rows = [row for row in artifact["rows"] if row["variant_name"] == "default"]

    assert default_rows
    assert {row["budget_1"] for row in default_rows} == {default_problem.group1_budget}
    assert {row["budget_2"] for row in default_rows} == {default_problem.group2_budget}
    for definition in artifact["variant_definitions"]:
        assert "budget_1" in definition
        assert "budget_2" in definition


def test_tightness_artifact_includes_summaries_pairing_and_fairness(
    tmp_path: Path,
) -> None:
    artifact = _run_tiny_tightness(tmp_path, "tightness_runner_artifact")
    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["variant_strategy_summaries"]
    assert payload["per_constraint_summaries"]
    assert payload["paired_comparisons"]
    assert payload["fairness_summary"]["status"] == "pass"
    assert {row["strategy"] for row in payload["rows"]} == {
        "random_search_feasibility_first",
        "constrained_ga_feasibility_first",
    }
    constrained_rows = [
        row for row in payload["rows"] if row["strategy"] == "constrained_ga_feasibility_first"
    ]
    assert constrained_rows
    for row in constrained_rows:
        assert row["per_constraint_summary_scope"] == "evaluation_trace"
        assert row["per_constraint_trace_summary"]["records_count"] == row["requested_budget"] * 2
    assert any(
        item["strategy"] == "constrained_ga_feasibility_first"
        and item["constraint_name"] == "group2_budget"
        and item["satisfaction_rate"] is not None
        for item in payload["per_constraint_summaries"]
    )


def test_tightness_actual_evaluations_equal_requested_budget_for_each_run(
    tmp_path: Path,
) -> None:
    artifact = _run_tiny_tightness(tmp_path, "tightness_runner_budget")

    for row in artifact["rows"]:
        assert row["actual_evaluations"] == row["requested_budget"]


def test_tightness_suffix_respected_and_default_not_changed(tmp_path: Path) -> None:
    suffix = "tightness_runner_suffix"
    artifact = _run_tiny_tightness(tmp_path, suffix)

    assert suffix in artifact["artifacts"]["json"]
    assert artifact["default_changed"] is False
    assert artifact["ga_default_changed"] is False
    assert artifact["nsga2_constraint_domination_done"] is False
    assert artifact["penalty_repair_done"] is False


def test_tightness_artifact_has_no_nan_placeholders(tmp_path: Path) -> None:
    artifact = _run_tiny_tightness(tmp_path, "tightness_runner_no_nan")
    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "NaN" not in json.dumps(payload, allow_nan=False)
