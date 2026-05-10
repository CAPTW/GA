from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_external_stress_runner_accepts_multiple_budgets(tmp_path: Path) -> None:
    suffix = "stress_contract"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_constrained_nsga2_external.py",
            "--problems",
            "constrained_zdt_box_toy,constrained_dtlz_box_toy",
            "--seeds",
            "1",
            "--budgets",
            "20,32",
            "--population-size",
            "4",
            "--artifact-suffix",
            suffix,
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_payload = json.loads(completed.stdout)
    json_path = Path(cli_payload["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.name == f"constrained_nsga2_external_stress_results_{suffix}.json"
    assert payload["configuration"]["budgets"] == [20, 32]
    assert payload["external_parity_established"] is False
    assert payload["constrained_nsga2_scope_change"] == "none"
    assert payload["default_nsga2_changed"] is False

    row_keys = {(row["problem"], row["requested_budget"], row["strategy"]) for row in payload["rows"]}
    for problem in ("constrained_zdt_box_toy", "constrained_dtlz_box_toy"):
        for budget in (20, 32):
            for strategy in (
                "constrained_nsga2_constraint_domination",
                "pymoo_constrained_nsga2",
                "random_pareto_archive",
            ):
                assert (problem, budget, strategy) in row_keys

    summary_keys = {
        (row["benchmark"], row["budget"], row["algorithm"])
        for row in payload["summaries"]
    }
    for problem in ("constrained_zdt_box_toy", "constrained_dtlz_box_toy"):
        for budget in (20, 32):
            assert (problem, budget, "constrained_nsga2_constraint_domination") in summary_keys
            assert (problem, budget, "pymoo_constrained_nsga2") in summary_keys

    assert payload["paired_comparisons"]
    assert "fairness_summary" in payload
    for row in payload["rows"]:
        if row["status"] == "success":
            assert row["actual_evaluations"] == row["requested_budget"]
            assert "feasible_only_HV" in row


def test_external_stress_runner_resume_skips_completed_rows(tmp_path: Path) -> None:
    source_suffix = "resume_phase0_source"
    source_run = subprocess.run(
        [
            sys.executable,
            "scripts/compare_constrained_nsga2_external.py",
            "--problems",
            "constrained_zdt_box_toy",
            "--seeds",
            "1",
            "--budgets",
            "20",
            "--population-size",
            "4",
            "--artifact-suffix",
            source_suffix,
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    source_payload = json.loads(
        Path(json.loads(source_run.stdout)["artifacts"]["json"]).read_text(encoding="utf-8")
    )
    source_artifact = Path(source_payload["artifacts"]["json"])

    resume_suffix = "resume_phase0_target"
    resume_run = subprocess.run(
        [
            sys.executable,
            "scripts/compare_constrained_nsga2_external.py",
            "--problems",
            "constrained_zdt_box_toy",
            "--seeds",
            "2",
            "--budgets",
            "20",
            "--population-size",
            "4",
            "--artifact-suffix",
            resume_suffix,
            "--output-dir",
            str(tmp_path),
            "--resume-from",
            str(source_artifact),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    resume_payload = json.loads(
        Path(json.loads(resume_run.stdout)["artifacts"]["json"]).read_text(encoding="utf-8")
    )

    assert resume_payload["resume_enabled"] is True
    assert resume_payload["resume_source_artifact"] == str(source_artifact)
    assert resume_payload["resume_summary"]["total_planned"] == 6
    assert resume_payload["resume_summary"]["skipped_existing"] == 3
    assert resume_payload["resume_summary"]["newly_executed"] == 3
    assert source_artifact.exists()
    assert Path(resume_payload["artifacts"]["json"]).name.endswith(f"{resume_suffix}.json")
    assert {row["row_origin"] for row in resume_payload["rows"]} == {
        "source_artifact",
        "newly_executed",
    }
    assert all("resume_key" in row for row in resume_payload["rows"])
