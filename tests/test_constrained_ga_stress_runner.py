from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.experiment.constrained_ga_stress import (
    ConstrainedGAStressConfig,
    run_constrained_ga_stress,
)


def _stress_row_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["problem"],
        row["strategy"],
        row["seed"],
        row["requested_budget"],
        row["dimension"],
        row.get("tolerance"),
    )


def test_stress_runner_accepts_multiple_budgets_and_seed_count(tmp_path: Path) -> None:
    artifact = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=3,
            seeds=2,
            budgets=(12, 20),
            population_size=4,
            artifact_suffix="constrained_ga_stress_runner_multi",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    observed_budgets = {row["budget"] for row in payload["rows"]}
    assert observed_budgets == {12, 20}
    assert payload["configuration"]["seed_list"] == [0, 1]


def test_stress_artifact_includes_budget_summaries_and_paired_comparisons(tmp_path: Path) -> None:
    artifact = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=3,
            seeds=2,
            budgets=(12, 20),
            population_size=4,
            artifact_suffix="constrained_ga_stress_runner_summary",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    strategies = {row["strategy"] for row in payload["rows"]}
    assert "random_search_feasibility_first" in strategies
    assert "constrained_ga_feasibility_first" in strategies
    assert payload["budget_strategy_summaries"]
    assert payload["paired_comparisons"]
    assert payload["fairness_summary"]["status"] == "pass"


def test_stress_actual_evaluations_equal_requested_budget_for_each_run(tmp_path: Path) -> None:
    artifact = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=4,
            seeds=2,
            budgets=(15, 24),
            population_size=5,
            artifact_suffix="constrained_ga_stress_budget_match",
            output_dir=str(tmp_path),
        )
    )

    for row in artifact["rows"]:
        assert row["actual_evaluations"] == row["requested_budget"]


def test_stress_artifact_suffix_respected_and_default_not_changed(tmp_path: Path) -> None:
    suffix = "constrained_ga_seed_budget_suffix"
    artifact = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=3,
            seeds=1,
            budgets=(12, 18),
            population_size=4,
            artifact_suffix=suffix,
            output_dir=str(tmp_path),
        )
    )

    assert suffix in artifact["artifacts"]["json"]
    assert artifact["default_changed"] is False
    assert artifact["ga_default_changed"] is False
    assert artifact["nsga2_constraint_domination_done"] is False


def test_stress_artifact_uses_null_for_missing_best_feasible_metric(tmp_path: Path) -> None:
    artifact = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=5,
            seeds=1,
            budgets=(10, 12),
            constraint_budget=-30.0,
            population_size=4,
            artifact_suffix="constrained_ga_stress_all_infeasible",
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    for row in payload["rows"]:
        assert row["best_feasible_objective"] is None
    serialized = json.dumps(payload)
    assert "NaN" not in serialized


def test_stress_runner_supports_constrained_box_quadratic(tmp_path: Path) -> None:
    suffix = "constrained_box_quadratic_stress_runner"
    artifact = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            problem="constrained_box_quadratic",
            dimension=6,
            seeds=1,
            budgets=(12, 16),
            population_size=4,
            artifact_suffix=suffix,
            output_dir=str(tmp_path),
        )
    )

    json_path = Path(artifact["artifacts"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "constrained_box_quadratic_stress_results" in json_path.name
    assert payload["problem"] == "constrained_box_quadratic"
    assert payload["configuration"]["problem"] == "constrained_box_quadratic"
    assert {row["strategy"] for row in payload["rows"]} == {
        "random_search_feasibility_first",
        "constrained_ga_feasibility_first",
    }
    assert payload["per_constraint_summaries"]
    assert payload["paired_comparisons"]
    assert payload["fairness_summary"]["status"] == "pass"
    for row in payload["rows"]:
        assert row["actual_evaluations"] == row["requested_budget"]
        assert "per_constraint_mean_violation" in row
    constrained_rows = [
        row for row in payload["rows"] if row["strategy"] == "constrained_ga_feasibility_first"
    ]
    assert constrained_rows
    for row in constrained_rows:
        assert row["per_constraint_summary_scope"] == "evaluation_trace"
        assert row["per_constraint_trace_summary"]["records_count"] == row["requested_budget"] * 2
        assert {item["constraint_name"] for item in row["per_constraint_violation_summary"]} == {
            "group1_budget",
            "group2_budget",
        }
    assert any(
        item["strategy"] == "constrained_ga_feasibility_first"
        and item["constraint_name"] == "group1_budget"
        and item["satisfaction_rate"] is not None
        for item in payload["per_constraint_summaries"]
    )
    assert payload["default_changed"] is False
    assert payload["ga_default_changed"] is False
    assert payload["nsga2_constraint_domination_done"] is False


def test_constrained_ga_stress_runner_resume_skips_completed_rows(tmp_path: Path) -> None:
    source_suffix = "runner_resume_phase0_cga_source"
    source_run = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_ga_stress.py",
            "--problem",
            "constrained_sphere",
            "--dimension",
            "3",
            "--seeds",
            "1",
            "--budgets",
            "12",
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
    source_payload = json.loads(source_run.stdout)
    source_artifact = Path(source_payload["artifacts"]["json"])

    resume_suffix = "runner_resume_phase0_cga_resume"
    resume_run = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_ga_stress.py",
            "--problem",
            "constrained_sphere",
            "--dimension",
            "3",
            "--seeds",
            "2",
            "--budgets",
            "12",
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
    resume_payload = json.loads(resume_run.stdout)

    assert resume_payload["resume_enabled"] is True
    assert resume_payload["resume_source_artifact"] == str(source_artifact)
    assert resume_payload["resume_summary"]["total_planned"] == 4
    assert resume_payload["resume_summary"]["completed_from_source"] == 2
    assert resume_payload["resume_summary"]["skipped_existing"] == 2
    assert resume_payload["resume_summary"]["newly_executed"] == 2
    assert resume_payload["resume_summary"]["failed_existing"] == 0
    assert source_artifact.exists()
    assert Path(resume_payload["artifacts"]["json"]).name.endswith(f"{resume_suffix}.json")
    assert resume_payload["artifacts"]["resume_markdown"]
    assert {row["row_origin"] for row in resume_payload["rows"]} == {
        "source_artifact",
        "newly_executed",
    }
    assert all("resume_key" in row for row in resume_payload["rows"])


def test_constrained_ga_stress_runner_accepts_serial_row_execution_backend(tmp_path: Path) -> None:
    artifact = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=3,
            seeds=1,
            budgets=(12,),
            population_size=4,
            artifact_suffix="runner_seed_parallelism_cga_serial_test",
            output_dir=str(tmp_path),
            row_execution_backend="serial",
        )
    )

    assert artifact["parallel_enabled"] is False
    assert artifact["parallel_backend"] == "serial"
    assert artifact["parallel_summary"]["parallel_enabled"] is False
    assert artifact["parallel_summary"]["total_submitted"] == 2
    assert all(row["row_execution_origin"] == "serial" for row in artifact["rows"])


def test_constrained_ga_stress_runner_thread_backend_matches_serial_order_and_metrics(
    tmp_path: Path,
) -> None:
    common = dict(
        dimension=3,
        seeds=2,
        budgets=(20,),
        population_size=4,
        output_dir=str(tmp_path),
    )
    serial = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            **common,
            artifact_suffix="runner_seed_parallelism_cga_serial_compare",
            row_execution_backend="serial",
        )
    )
    thread = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            **common,
            artifact_suffix="runner_seed_parallelism_cga_thread_compare",
            row_execution_backend="thread",
            row_execution_workers=2,
        )
    )

    assert thread["parallel_enabled"] is True
    assert thread["parallel_backend"] == "thread"
    assert thread["parallel_summary"]["success_count"] == len(thread["rows"])
    assert thread["parallel_summary"]["failure_count"] == 0
    assert [_stress_row_key(row) for row in thread["rows"]] == [
        _stress_row_key(row) for row in serial["rows"]
    ]
    assert [row["actual_evaluations"] for row in thread["rows"]] == [
        row["actual_evaluations"] for row in serial["rows"]
    ]
    assert [row["feasible_rate"] for row in thread["rows"]] == [
        row["feasible_rate"] for row in serial["rows"]
    ]
    assert [row["row_order_index"] for row in thread["rows"]] == list(range(len(thread["rows"])))
    assert all(row["row_execution_origin"] == "parallel_thread" for row in thread["rows"])


def test_constrained_ga_stress_runner_resume_thread_skips_completed_before_parallel(
    tmp_path: Path,
) -> None:
    source = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=3,
            seeds=1,
            budgets=(20,),
            population_size=4,
            artifact_suffix="runner_seed_parallelism_cga_source_seed1_test",
            output_dir=str(tmp_path),
            row_execution_backend="serial",
        )
    )
    source_artifact = Path(source["artifacts"]["json"])

    resumed = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=3,
            seeds=2,
            budgets=(20,),
            population_size=4,
            artifact_suffix="runner_seed_parallelism_cga_resume_thread_test",
            output_dir=str(tmp_path),
            resume_from=source_artifact,
            row_execution_backend="thread",
            row_execution_workers=2,
        )
    )

    assert source_artifact.exists()
    assert resumed["resume_enabled"] is True
    assert resumed["resume_summary"]["total_planned"] == 4
    assert resumed["resume_summary"]["completed_from_source"] == 2
    assert resumed["resume_summary"]["newly_executed"] == 2
    assert resumed["parallel_summary"]["skipped_before_parallel"] == 2
    assert resumed["parallel_summary"]["total_submitted"] == 2
    assert resumed["parallel_summary"]["success_count"] == 2
    assert [row["row_order_index"] for row in resumed["rows"]] == list(range(4))
    origins = {row["row_execution_origin"] for row in resumed["rows"]}
    assert origins == {"resume_source_artifact", "parallel_thread"}


def test_constrained_ga_stress_cli_accepts_thread_row_execution_backend(
    tmp_path: Path,
) -> None:
    run = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_ga_stress.py",
            "--problem",
            "constrained_sphere",
            "--dimension",
            "3",
            "--seeds",
            "1",
            "--budgets",
            "12",
            "--population-size",
            "4",
            "--artifact-suffix",
            "runner_seed_parallelism_cga_cli_thread",
            "--output-dir",
            str(tmp_path),
            "--row-execution-backend",
            "thread",
            "--workers",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(run.stdout)

    assert payload["parallel_enabled"] is True
    assert payload["parallel_backend"] == "thread"
    assert payload["parallel_workers"] == 2
    assert payload["parallel_summary"]["success_count"] == 2
    assert payload["parallel_summary"]["failure_count"] == 0


def test_constrained_ga_stress_cli_process_backend_requires_allow_flag(
    tmp_path: Path,
) -> None:
    run = subprocess.run(
        [
            sys.executable,
            "scripts/validate_constrained_ga_stress.py",
            "--problem",
            "constrained_sphere",
            "--dimension",
            "3",
            "--seeds",
            "1",
            "--budgets",
            "12",
            "--population-size",
            "4",
            "--artifact-suffix",
            "process_backend_requires_allow",
            "--output-dir",
            str(tmp_path),
            "--row-execution-backend",
            "process",
            "--workers",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0
    payload = json.loads(run.stdout)
    assert payload["parallel_enabled"] is True
    assert payload["parallel_backend"] == "process"
    assert payload["parallel_summary"]["success_count"] == 0
    assert payload["parallel_summary"]["failure_count"] == 2
    assert payload["failures"]
    assert "allow_process_backend=True" in payload["failures"][0]["message"]


def test_constrained_ga_stress_process_backend_tiny_smoke(
    tmp_path: Path,
) -> None:
    artifact = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=3,
            seeds=2,
            budgets=(20,),
            population_size=4,
            artifact_suffix="process_backend_smoke_test",
            output_dir=str(tmp_path),
            row_execution_backend="process",
            row_execution_workers=2,
            allow_process_backend=True,
        )
    )

    assert artifact["parallel_enabled"] is True
    assert artifact["parallel_backend"] == "process"
    assert artifact["parallel_workers"] == 2
    assert artifact["parallel_summary"]["success_count"] == 4
    assert artifact["parallel_summary"]["failure_count"] == 0
    assert artifact["parallel_summary"]["deterministic_merge_order"] is True
    assert [row["row_order_index"] for row in artifact["rows"]] == list(range(4))
    assert all(row["row_execution_origin"] == "parallel_process" for row in artifact["rows"])
    assert all(row["actual_evaluations"] == row["requested_budget"] for row in artifact["rows"])


def test_constrained_ga_stress_resume_process_skips_completed_before_submission(
    tmp_path: Path,
) -> None:
    source = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=3,
            seeds=1,
            budgets=(20,),
            population_size=4,
            artifact_suffix="process_backend_resume_source_test",
            output_dir=str(tmp_path),
            row_execution_backend="serial",
        )
    )
    source_artifact = Path(source["artifacts"]["json"])
    source_mtime = source_artifact.stat().st_mtime_ns

    resumed = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            dimension=3,
            seeds=2,
            budgets=(20,),
            population_size=4,
            artifact_suffix="process_backend_resume_smoke_test",
            output_dir=str(tmp_path),
            resume_from=source_artifact,
            row_execution_backend="process",
            row_execution_workers=2,
            allow_process_backend=True,
        )
    )

    assert source_artifact.exists()
    assert source_artifact.stat().st_mtime_ns == source_mtime
    assert resumed["resume_enabled"] is True
    assert resumed["resume_summary"]["completed_from_source"] == 2
    assert resumed["resume_summary"]["newly_executed"] == 2
    assert resumed["parallel_summary"]["skipped_before_parallel"] == 2
    assert resumed["parallel_summary"]["total_submitted"] == 2
    assert resumed["parallel_summary"]["success_count"] == 2
    assert resumed["parallel_summary"]["failure_count"] == 0
    assert [row["row_order_index"] for row in resumed["rows"]] == list(range(4))
    assert {row["row_execution_origin"] for row in resumed["rows"]} == {
        "resume_source_artifact",
        "parallel_process",
    }
