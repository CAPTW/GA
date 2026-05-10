from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.experiment.constrained_nsga2_stress import (
    ConstrainedNSGA2StressConfig,
    run_constrained_nsga2_stress,
)


def test_stress_runner_accepts_both_problems_and_multiple_budgets(tmp_path: Path) -> None:
    suffix = "stress_runner_multi"
    artifact = run_constrained_nsga2_stress(
        ConstrainedNSGA2StressConfig(
            problems=("constrained_zdt_box_toy", "constrained_dtlz_box_toy"),
            dimensions={
                "constrained_zdt_box_toy": 6,
                "constrained_dtlz_box_toy": 7,
            },
            seeds=1,
            budgets=(20, 32),
            population_size=4,
            artifact_suffix=suffix,
            output_dir=str(tmp_path),
        )
    )

    payload = json.loads(Path(artifact["artifacts"]["json"]).read_text(encoding="utf-8"))

    assert suffix in artifact["artifacts"]["json"]
    assert {row["benchmark"] for row in payload["rows"]} == {
        "constrained_zdt_box_toy",
        "constrained_dtlz_box_toy",
    }
    assert {row["requested_budget"] for row in payload["rows"]} == {20, 32}
    assert payload["benchmark_budget_summaries"]
    assert payload["paired_comparisons"]


def test_stress_artifact_contains_metrics_fairness_and_no_default_contamination(
    tmp_path: Path,
) -> None:
    artifact = run_constrained_nsga2_stress(
        ConstrainedNSGA2StressConfig(
            problems=("constrained_zdt_box_toy",),
            dimensions={"constrained_zdt_box_toy": 6},
            seeds=1,
            budgets=(20,),
            population_size=4,
            artifact_suffix="stress_runner_contract",
            output_dir=str(tmp_path),
        )
    )

    payload = json.loads(Path(artifact["artifacts"]["json"]).read_text(encoding="utf-8"))

    assert payload["fairness_summary"]["status"] == "pass"
    assert payload["fairness_summary"]["by_benchmark_budget"]
    assert payload["default_changed"] is False
    assert payload["nsga2_default_changed"] is False
    for row in payload["rows"]:
        assert row["actual_evaluations"] == row["requested_budget"]
        assert "feasible_only_HV" in row
        assert "feasible_only_reference_distance" in row
        assert "spacing_feasible_only" in row
        assert "per_constraint_satisfaction_rate" in row
        assert "per_constraint_mean_violation" in row
        assert "per_constraint_max_violation" in row


def test_stress_null_policy_is_used_when_feasible_front_unavailable(tmp_path: Path) -> None:
    artifact = run_constrained_nsga2_stress(
        ConstrainedNSGA2StressConfig(
            problems=("constrained_zdt_box_toy",),
            dimensions={"constrained_zdt_box_toy": 6},
            seeds=(0,),
            budgets=(20,),
            population_size=4,
            artifact_suffix="stress_runner_null_policy",
            output_dir=str(tmp_path),
            strategies=("random_pareto_archive",),
            constraint_overrides={
                "constrained_zdt_box_toy": {
                    "first_pair_budget": -1.0,
                    "second_half_budget": -1.0,
                }
            },
        )
    )

    payload = json.loads(Path(artifact["artifacts"]["json"]).read_text(encoding="utf-8"))

    assert payload["rows"][0]["feasible_rate"] == 0.0
    assert payload["rows"][0]["feasible_only_HV"] is None
    assert payload["rows"][0]["feasible_only_reference_distance"] is None
    assert payload["rows"][0]["spacing_feasible_only"] is None


def test_stress_runner_resume_skips_completed_rows(tmp_path: Path) -> None:
    source_suffix = "resume_phase0_expand_source"
    source_run = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_nsga2_stress.py",
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

    resume_suffix = "resume_phase0_expand_target"
    resume_run = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_nsga2_stress.py",
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
    assert resume_payload["resume_summary"]["total_planned"] == 4
    assert resume_payload["resume_summary"]["completed_from_source"] == 2
    assert resume_payload["resume_summary"]["skipped_existing"] == 2
    assert resume_payload["resume_summary"]["newly_executed"] == 2
    assert resume_payload["resume_summary"]["failed_existing"] == 0
    assert source_artifact.exists()
    assert Path(resume_payload["artifacts"]["json"]).name.endswith(f"{resume_suffix}.json")
    assert {row["row_origin"] for row in resume_payload["rows"]} == {
        "source_artifact",
        "newly_executed",
    }
    assert all("resume_key" in row for row in resume_payload["rows"])


def test_runner_accepts_serial_parallel_options_and_records_summary(tmp_path: Path) -> None:
    run = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_nsga2_stress.py",
            "--problems",
            "constrained_zdt_box_toy",
            "--seeds",
            "1",
            "--budgets",
            "20",
            "--population-size",
            "4",
            "--artifact-suffix",
            "runner_parallel_serial_cli",
            "--output-dir",
            str(tmp_path),
            "--row-execution-backend",
            "serial",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(Path(json.loads(run.stdout)["artifacts"]["json"]).read_text(encoding="utf-8"))

    assert payload["parallel_enabled"] is False
    assert payload["parallel_backend"] == "serial"
    assert payload["parallel_summary"]["deterministic_merge_order"] is True
    assert all(row["row_execution_origin"] == "serial" for row in payload["rows"])
    assert [row["row_order_index"] for row in payload["rows"]] == list(range(len(payload["rows"])))


def test_thread_parallel_rows_match_serial_order_and_values(tmp_path: Path) -> None:
    common_args = [
        "--problems",
        "constrained_zdt_box_toy",
        "--seeds",
        "2",
        "--budgets",
        "20",
        "--population-size",
        "4",
        "--output-dir",
        str(tmp_path),
    ]
    serial = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_nsga2_stress.py",
            *common_args,
            "--artifact-suffix",
            "runner_parallel_serial_compare",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    thread = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_nsga2_stress.py",
            *common_args,
            "--artifact-suffix",
            "runner_parallel_thread_compare",
            "--row-execution-backend",
            "thread",
            "--workers",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    serial_payload = json.loads(Path(json.loads(serial.stdout)["artifacts"]["json"]).read_text(encoding="utf-8"))
    thread_payload = json.loads(Path(json.loads(thread.stdout)["artifacts"]["json"]).read_text(encoding="utf-8"))

    serial_keys = [
        (row["benchmark"], row["strategy"], row["seed"], row["requested_budget"])
        for row in serial_payload["rows"]
    ]
    thread_keys = [
        (row["benchmark"], row["strategy"], row["seed"], row["requested_budget"])
        for row in thread_payload["rows"]
    ]
    assert thread_keys == serial_keys
    assert [row["actual_evaluations"] for row in thread_payload["rows"]] == [
        row["actual_evaluations"] for row in serial_payload["rows"]
    ]
    assert [row["feasible_rate"] for row in thread_payload["rows"]] == [
        row["feasible_rate"] for row in serial_payload["rows"]
    ]
    assert thread_payload["parallel_enabled"] is True
    assert thread_payload["parallel_summary"]["success_count"] == len(thread_payload["rows"])


def test_resume_from_source_skips_completed_before_thread_parallel(tmp_path: Path) -> None:
    source = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_nsga2_stress.py",
            "--problems",
            "constrained_zdt_box_toy",
            "--seeds",
            "1",
            "--budgets",
            "20",
            "--population-size",
            "4",
            "--artifact-suffix",
            "runner_parallel_resume_source",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    source_artifact = Path(json.loads(source.stdout)["artifacts"]["json"])

    resumed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_nsga2_stress.py",
            "--problems",
            "constrained_zdt_box_toy",
            "--seeds",
            "2",
            "--budgets",
            "20",
            "--population-size",
            "4",
            "--artifact-suffix",
            "runner_parallel_resume_thread",
            "--output-dir",
            str(tmp_path),
            "--resume-from",
            str(source_artifact),
            "--row-execution-backend",
            "thread",
            "--workers",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(Path(json.loads(resumed.stdout)["artifacts"]["json"]).read_text(encoding="utf-8"))

    assert payload["resume_summary"]["completed_from_source"] == 2
    assert payload["resume_summary"]["newly_executed"] == 2
    assert payload["parallel_summary"]["skipped_before_parallel"] == 2
    assert payload["parallel_summary"]["total_submitted"] == 2
    assert {row["row_origin"] for row in payload["rows"]} == {"source_artifact", "newly_executed"}
    assert source_artifact.exists()
