from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_candidate_o_non_zdt_scope_is_consistent() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_payload = json.loads(
        (
            repo_root
            / "configs"
            / "candidates"
            / "nsga2_spread_preserving_variation_candidate_o.json"
        ).read_text(encoding="utf-8")
    )
    usage_guide = (
        repo_root / "docs" / "candidates" / "nsga2_candidate_o_opt_in_usage.md"
    ).read_text(encoding="utf-8")

    assert config_payload["candidate_id"] == "candidate_o_spread_preserving_variation_light"
    assert config_payload["approval_status"] == "approved_restricted_opt_in"
    assert config_payload["default_changed"] is False
    assert "approved_restricted_opt_in" in usage_guide or "restricted opt-in experimental profile" in usage_guide.lower()
    assert "default NSGA-II replacement" in usage_guide
    assert "broader non-ZDT approval claim or scope expansion without a separate review" in usage_guide
    assert "WFG status: completed_positive" in usage_guide
    assert "WFG decision: restricted opt-in scope maintained, WFG smoke positive" in usage_guide


def test_candidate_o_non_zdt_smoke_runner_writes_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "non-zdt-artifacts"
    output_root = tmp_path / "non-zdt-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_nsga2_candidate_o_non_zdt_smoke.py",
            "--problems",
            "dtlz2,dtlz3",
            "--seeds",
            "1",
            "--seed-start",
            "42101",
            "--budget",
            "300",
            "--artifact-suffix",
            "dtlz_smoke_test",
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

    assert results_json.name == "nsga2_candidate_o_non_zdt_smoke_results_dtlz_smoke_test.json"
    assert report_md.name == "nsga2_candidate_o_non_zdt_smoke_report_dtlz_smoke_test.md"
    assert fairness_md.name == "nsga2_candidate_o_non_zdt_smoke_fairness_report_dtlz_smoke_test.md"
    assert result_payload["selected_problems"] == ["dtlz2", "dtlz3"]
    assert result_payload["budget"] == 300
    assert "fairness_summary" in result_payload
    assert result_payload["fairness"]["status"] in {"pass", "warning", "fail"}
    assert result_payload["metric_limitations"]

    candidate_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"] == "candidate_o_spread_preserving_variation_light"
    ]
    assert len(candidate_rows) == 2
    for row in candidate_rows:
        assert row["metadata"]["candidate_id"] == "candidate_o_spread_preserving_variation_light"
        assert row["metadata"]["base_candidate_id"] == "candidate_n_low_g_tail_mutation_light"
        assert row["metadata"]["default_changed"] is False
        assert row["metadata"]["approval_status"] == "approved_restricted_opt_in"
        assert row["metadata"]["allowed_use"]
        assert row["requested_budget"] == 300
        assert row["actual_evaluations"] == 300
        assert row["spread_proxy_diagnostics_success"] is True

    default_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert len(default_rows) == 2
    for row in default_rows:
        assert "candidate_id" not in row["metadata"]
        assert "default_changed" not in row["metadata"]
        assert "diagnostics_enabled" not in row["metadata"]

    paired_rows = result_payload["paired_rows"]
    assert any(row["comparison"] == "candidate_o vs candidate_j" for row in paired_rows)
    assert any(row["comparison"] == "candidate_o vs candidate_n" for row in paired_rows)
    assert any(row["comparison"] == "candidate_o vs pymoo" for row in paired_rows)


def test_candidate_o_non_zdt_smoke_runner_rejects_non_dtlz_problem(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "non-zdt-invalid-artifacts"
    output_root = tmp_path / "non-zdt-invalid-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_nsga2_candidate_o_non_zdt_smoke.py",
            "--problems",
            "dtlz2,zdt1",
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
    assert "dtlz2,dtlz3" in (completed.stderr + completed.stdout).lower()
