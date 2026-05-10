from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_survivor_pressure_diagnostics_runner_writes_deep_trace_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "diagnostics-artifacts"
    output_root = tmp_path / "diagnostics-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_nsga2_survivor_pressure.py",
            "--problem",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            "15001",
            "--budget",
            "80",
            "--artifact-suffix",
            "diag_deep_test",
            "--artifact-root",
            str(artifact_root),
            "--output-root",
            str(output_root),
            "--deep",
            "--reference-algorithm",
            "candidate_j_h_lite_retry2",
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

    assert (
        results_json.name
        == "nsga2_survivor_pressure_deep_diagnostics_results_diag_deep_test.json"
    )
    assert (
        report_md.name
        == "nsga2_survivor_pressure_deep_diagnostics_report_diag_deep_test.md"
    )
    assert result_payload["selected_problems"] == ["zdt1"]
    assert result_payload["deep_trace_enabled"] is True
    assert result_payload["lineage_trace_enabled"] is False
    assert result_payload["reference_algorithm"] == "candidate_j_h_lite_retry2"

    algorithms = {row["algorithm"] for row in result_payload["raw_rows"]}
    assert algorithms == {
        "internal_nsga2",
        "candidate_j_h_lite_retry2",
        "candidate_l_sparse_parent_bias_light",
        "candidate_m_boundary_preservation_light",
    }

    internal_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert len(internal_rows) == 1
    assert "candidate_id" not in internal_rows[0]["metadata"]
    assert "nsga2_diagnostics" in internal_rows[0]["metadata"]
    assert (
        internal_rows[0]["metadata"]["nsga2_diagnostics"]["trace_config"]["deep_trace_enabled"]
        is True
    )

    candidate_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] != "internal_nsga2"
    ]
    for row in candidate_rows:
        assert row["metadata"]["candidate_id"] == row["algorithm"]
        assert row["metadata"]["default_changed"] is False
        assert "nsga2_diagnostics" in row["metadata"]
        assert row["actual_evaluations"] == 80

    assert result_payload["deep_parent_rows"]
    assert result_payload["deep_offspring_rows"]
    assert result_payload["deep_survivor_diff_rows"]
    assert result_payload["deep_boundary_rows"]
    assert result_payload["deep_segment_rows"]

    paired_rows = result_payload["paired_rows"]
    assert any(
        row["comparison"] == "candidate_l vs candidate_j" and row["problem"] == "zdt1"
        for row in paired_rows
    )
    assert any(
        row["comparison"] == "candidate_m vs candidate_l" and row["problem"] == "zdt1"
        for row in paired_rows
    )


def test_survivor_pressure_diagnostics_runner_writes_lineage_trace_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "diagnostics-artifacts"
    output_root = tmp_path / "diagnostics-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_nsga2_survivor_pressure.py",
            "--problem",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            "15101",
            "--budget",
            "80",
            "--artifact-suffix",
            "diag_lineage_test",
            "--artifact-root",
            str(artifact_root),
            "--output-root",
            str(output_root),
            "--deep",
            "--lineage",
            "--reference-algorithm",
            "candidate_j_h_lite_retry2",
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

    assert (
        results_json.name
        == "nsga2_survivor_pressure_lineage_diagnostics_results_diag_lineage_test.json"
    )
    assert (
        report_md.name
        == "nsga2_survivor_pressure_lineage_diagnostics_report_diag_lineage_test.md"
    )
    assert result_payload["selected_problems"] == ["zdt1"]
    assert result_payload["deep_trace_enabled"] is True
    assert result_payload["lineage_trace_enabled"] is True
    assert result_payload["lineage_funnel_rows"]
    assert result_payload["lineage_sparse_rows"]
    assert result_payload["lineage_divergence_summary_rows"]
    assert result_payload["lineage_segment0_rows"]
    assert result_payload["lineage_duplicate_rows"]
    assert result_payload["lineage_boundary_rows"]

    internal_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert len(internal_rows) == 1
    assert "candidate_id" not in internal_rows[0]["metadata"]
    assert "nsga2_diagnostics" in internal_rows[0]["metadata"]
    assert (
        internal_rows[0]["metadata"]["nsga2_diagnostics"]["trace_config"]["lineage_trace_enabled"]
        is True
    )

    candidate_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] != "internal_nsga2"
    ]
    for row in candidate_rows:
        assert row["metadata"]["candidate_id"] == row["algorithm"]
        assert row["metadata"]["default_changed"] is False
        assert row["actual_evaluations"] == 80

    assert any(
        row["comparison"] == "candidate_m vs candidate_j" and row["problem"] == "zdt1"
        for row in result_payload["paired_rows"]
    )


def test_survivor_pressure_diagnostics_runner_writes_operator_supply_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "diagnostics-artifacts"
    output_root = tmp_path / "diagnostics-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_nsga2_survivor_pressure.py",
            "--problem",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            "15201",
            "--budget",
            "80",
            "--artifact-suffix",
            "diag_operator_supply_test",
            "--artifact-root",
            str(artifact_root),
            "--output-root",
            str(output_root),
            "--operator-supply",
            "--reference-algorithm",
            "candidate_j_h_lite_retry2",
            "--segment-count",
            "6",
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

    assert (
        results_json.name
        == "nsga2_operator_supply_diagnostics_results_diag_operator_supply_test.json"
    )
    assert (
        report_md.name
        == "nsga2_operator_supply_diagnostics_report_diag_operator_supply_test.md"
    )
    assert result_payload["operator_supply_trace_enabled"] is True
    assert result_payload["segment_count"] == 6
    assert result_payload["selected_problems"] == ["zdt1"]

    internal_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert len(internal_rows) == 1
    assert "candidate_id" not in internal_rows[0]["metadata"]
    assert (
        internal_rows[0]["metadata"]["nsga2_diagnostics"]["trace_config"][
            "operator_supply_trace_enabled"
        ]
        is True
    )

    candidate_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"]
        in {
            "candidate_j_h_lite_retry2",
            "candidate_l_sparse_parent_bias_light",
            "candidate_m_boundary_preservation_light",
        }
    ]
    for row in candidate_rows:
        assert row["metadata"]["candidate_id"] == row["algorithm"]
        assert row["metadata"]["default_changed"] is False
        assert row["actual_evaluations"] == 80

    assert result_payload["operator_initialization_rows"]
    assert result_payload["operator_transition_rows"]
    assert result_payload["operator_offspring_quality_rows"]
    assert result_payload["operator_retry_rows"]
    assert result_payload["operator_supply_funnel_rows"]
    assert "operator_supply_trace_enabled" not in internal_rows[0]["metadata"]
    assert any(
        row["comparison"] == "candidate_l vs candidate_j" and row["problem"] == "zdt1"
        for row in result_payload["paired_rows"]
    )
    assert any(
        row["comparison"] == "candidate_m vs candidate_l" and row["problem"] == "zdt1"
        for row in result_payload["paired_rows"]
    )


def test_survivor_pressure_diagnostics_runner_writes_zdt1_component_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "diagnostics-artifacts"
    output_root = tmp_path / "diagnostics-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_nsga2_survivor_pressure.py",
            "--problem",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            "15301",
            "--budget",
            "80",
            "--artifact-suffix",
            "diag_zdt1_component_test",
            "--artifact-root",
            str(artifact_root),
            "--output-root",
            str(output_root),
            "--zdt1-components",
            "--operator-supply",
            "--reference-algorithm",
            "candidate_j_h_lite_retry2",
            "--segment-count",
            "6",
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

    assert (
        results_json.name
        == "nsga2_zdt1_component_diagnostics_results_diag_zdt1_component_test.json"
    )
    assert (
        report_md.name
        == "nsga2_zdt1_component_diagnostics_report_diag_zdt1_component_test.md"
    )
    assert result_payload["zdt1_component_trace_enabled"] is True
    assert result_payload["operator_supply_trace_enabled"] is True
    assert result_payload["segment_count"] == 6
    assert result_payload["selected_problems"] == ["zdt1"]

    internal_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert len(internal_rows) == 1
    assert "candidate_id" not in internal_rows[0]["metadata"]
    assert (
        internal_rows[0]["metadata"]["nsga2_diagnostics"]["trace_config"][
            "zdt1_component_trace_enabled"
        ]
        is True
    )

    candidate_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"]
        in {
            "candidate_j_h_lite_retry2",
            "candidate_l_sparse_parent_bias_light",
            "candidate_m_boundary_preservation_light",
        }
    ]
    for row in candidate_rows:
        assert row["metadata"]["candidate_id"] == row["algorithm"]
        assert row["metadata"]["default_changed"] is False
        assert row["actual_evaluations"] == 80

    assert result_payload["zdt1_initial_component_rows"]
    assert result_payload["zdt1_offspring_component_rows"]
    assert result_payload["zdt1_parent_child_delta_rows"]
    assert result_payload["zdt1_retry_component_rows"]
    assert result_payload["zdt1_segment0_funnel_rows"]
    assert result_payload["zdt1_internal_external_rows"]
    assert any(
        row["comparison"] == "candidate_l vs candidate_j" and row["problem"] == "zdt1"
        for row in result_payload["paired_rows"]
    )
