from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_spread_preserving_phase1_runner_writes_candidate_o_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "spread-phase1-artifacts"
    output_root = tmp_path / "spread-phase1-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_nsga2_spread_preserving_phase1.py",
            "--problem",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            "28101",
            "--budget",
            "760",
            "--artifact-suffix",
            "candidate_o_phase1_test",
            "--artifact-root",
            str(artifact_root),
            "--output-root",
            str(output_root),
            "--skip-random-archive",
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

    assert results_json.name == "nsga2_spread_preserving_phase1_results_candidate_o_phase1_test.json"
    assert report_md.name == "nsga2_spread_preserving_phase1_report_candidate_o_phase1_test.md"
    assert result_payload["selected_problems"] == ["zdt1"]
    assert "fairness" in result_payload
    assert "fairness_summary" in result_payload
    assert result_payload["fairness"]["status"] in {"pass", "warning", "fail"}
    assert result_payload["spread_rows"]
    assert result_payload["zdt1_component_rows"]
    assert result_payload["operator_supply_rows"]

    candidate_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"] == "candidate_o_spread_preserving_variation_light"
    ]
    assert len(candidate_rows) == 1
    for row in candidate_rows:
        assert row["metadata"]["candidate_id"] == "candidate_o_spread_preserving_variation_light"
        assert row["metadata"]["base_candidate_id"] == "candidate_n_low_g_tail_mutation_light"
        assert row["metadata"]["default_changed"] is False
        assert row["metadata"]["promotion_status"] in {"phase0_sanity", "phase1_validation"}
        assert row["requested_budget"] == 760
        assert row["actual_evaluations"] == 760
        assert row["spread_parity_diagnostics_success"] is True
        assert row["zdt1_component_diagnostics_success"] is True

    default_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert len(default_rows) == 1
    for row in default_rows:
        assert "candidate_id" not in row["metadata"]
        assert "default_changed" not in row["metadata"]

    paired_rows = result_payload["paired_rows"]
    assert any(row["comparison"] == "candidate_o vs candidate_n" for row in paired_rows)
    assert any(row["comparison"] == "candidate_o vs pymoo" for row in paired_rows)
    assert "phase1_decision" in result_payload
