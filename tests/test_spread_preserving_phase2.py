from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_spread_preserving_phase2_runner_writes_candidate_o_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "spread-phase2-artifacts"
    output_root = tmp_path / "spread-phase2-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_nsga2_spread_preserving_phase2.py",
            "--problems",
            "zdt1,zdt2,zdt3",
            "--zdt1-seeds",
            "1",
            "--other-seeds",
            "1",
            "--zdt1-seed-start",
            "31101",
            "--other-seed-start",
            "31201",
            "--budget",
            "300",
            "--artifact-suffix",
            "candidate_o_phase2_test",
            "--artifact-root",
            str(artifact_root),
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    results_json = Path(str(payload["results_json"]))
    report_md = Path(str(payload["report_md"]))
    result_payload = json.loads(results_json.read_text(encoding="utf-8"))

    assert results_json.name == "nsga2_spread_preserving_phase2_results_candidate_o_phase2_test.json"
    assert report_md.name == "nsga2_spread_preserving_phase2_report_candidate_o_phase2_test.md"
    assert result_payload["selected_problems"] == ["zdt1", "zdt2", "zdt3"]
    assert len(result_payload["problem_seed_map"]["zdt1"]) == 1
    assert len(result_payload["problem_seed_map"]["zdt2"]) == 1
    assert len(result_payload["problem_seed_map"]["zdt3"]) == 1
    assert "fairness_summary" in result_payload
    assert result_payload["fairness"]["status"] in {"pass", "warning", "fail"}
    assert result_payload["spread_rows"]
    assert result_payload["zdt1_component_rows"]
    assert all(row["problem"] == "zdt1" for row in result_payload["zdt1_component_rows"])
    assert result_payload["operator_supply_rows"]

    candidate_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"] == "candidate_o_spread_preserving_variation_light"
    ]
    assert candidate_rows
    for row in candidate_rows:
        assert row["metadata"]["candidate_id"] == "candidate_o_spread_preserving_variation_light"
        assert row["metadata"]["base_candidate_id"] == "candidate_n_low_g_tail_mutation_light"
        assert row["metadata"]["default_changed"] is False
        assert row["metadata"]["promotion_status"] == "phase2_validation"
        assert row["requested_budget"] == 300
        assert row["actual_evaluations"] == 300
        assert row["spread_parity_diagnostics_success"] is True
        assert row["success"] is True
        if row["problem"] == "zdt1":
            assert row["zdt1_component_diagnostics_success"] is True

    default_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert default_rows
    for row in default_rows:
        assert "candidate_id" not in row["metadata"]
        assert "default_changed" not in row["metadata"]

    paired_rows = result_payload["paired_rows"]
    assert any(row["comparison"] == "candidate_o vs candidate_n" for row in paired_rows)
    assert any(row["comparison"] == "candidate_o vs pymoo" for row in paired_rows)


def test_spread_preserving_phase2_runner_rejects_non_zdt_problem(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "spread-phase2-invalid-artifacts"
    output_root = tmp_path / "spread-phase2-invalid-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_nsga2_spread_preserving_phase2.py",
            "--problems",
            "zdt1,dtlz2",
            "--zdt1-seeds",
            "1",
            "--other-seeds",
            "1",
            "--budget",
            "300",
            "--artifact-root",
            str(artifact_root),
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "zdt1,zdt2,zdt3" in (completed.stderr + completed.stdout).lower()
