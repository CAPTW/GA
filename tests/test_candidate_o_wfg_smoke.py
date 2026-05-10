from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def test_candidate_o_wfg_smoke_runner_writes_artifacts_or_skips(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "wfg-artifacts"
    output_root = tmp_path / "wfg-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_nsga2_candidate_o_wfg_smoke.py",
            "--problems",
            "wfg1,wfg2",
            "--seeds",
            "1",
            "--seed-start",
            "46101",
            "--budget",
            "300",
            "--artifact-suffix",
            "wfg_smoke_test",
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
    fairness_md = Path(str(payload["fairness_md"]))
    result_payload = json.loads(results_json.read_text(encoding="utf-8"))

    assert results_json.name == "nsga2_candidate_o_wfg_smoke_results_wfg_smoke_test.json"
    assert report_md.name == "nsga2_candidate_o_wfg_smoke_report_wfg_smoke_test.md"
    assert fairness_md.name == "nsga2_candidate_o_wfg_smoke_fairness_report_wfg_smoke_test.md"
    assert result_payload["selected_problems"] == ["wfg1", "wfg2"]
    assert result_payload["budget"] == 300
    assert "fairness_summary" in result_payload
    assert result_payload["metric_limitations"]
    assert result_payload["wfg_problem_rows"]
    assert any(row["comparison"] == "candidate_o vs candidate_j" for row in result_payload["paired_rows"])

    default_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert len(default_rows) == 2
    for row in default_rows:
        assert "candidate_id" not in row["metadata"]
        assert "default_changed" not in row["metadata"]
        assert "diagnostics_enabled" not in row["metadata"]

    candidate_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"] == "candidate_o_spread_preserving_variation_light"
    ]
    assert len(candidate_rows) == 2

    if importlib.util.find_spec("pymoo") is None:
        assert all(row["status"] == "skipped" for row in result_payload["raw_rows"])
        assert any("pymoo" in str(row.get("error_message", "")).lower() for row in candidate_rows)
    else:
        for row in candidate_rows:
            assert row["metadata"]["candidate_id"] == "candidate_o_spread_preserving_variation_light"
            assert row["metadata"]["base_candidate_id"] == "candidate_n_low_g_tail_mutation_light"
            assert row["metadata"]["default_changed"] is False
            assert row["metadata"]["approval_status"] == "approved_restricted_opt_in"
            assert row["requested_budget"] == 300
            assert row["actual_evaluations"] == 300
            assert row["spread_proxy_diagnostics_success"] is True

        assert any(row["comparison"] == "candidate_o vs candidate_n" for row in result_payload["paired_rows"])
        assert any(row["comparison"] == "candidate_o vs pymoo" for row in result_payload["paired_rows"])


def test_candidate_o_wfg_smoke_runner_rejects_non_wfg_problem(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "wfg-invalid-artifacts"
    output_root = tmp_path / "wfg-invalid-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_nsga2_candidate_o_wfg_smoke.py",
            "--problems",
            "wfg1,dtlz2",
            "--seeds",
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
    assert "wfg1,wfg2" in (completed.stderr + completed.stdout).lower()
