from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_external_comparison_runner_creates_expected_artifacts(tmp_path: Path) -> None:
    suffix = "runner_contract"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_constrained_nsga2_external.py",
            "--problems",
            "constrained_zdt_box_toy,constrained_dtlz_box_toy",
            "--seeds",
            "1",
            "--budget",
            "20",
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

    assert suffix in str(json_path)
    assert Path(payload["artifacts"]["csv"]).exists()
    assert Path(payload["artifacts"]["results_markdown"]).exists()
    assert Path(payload["artifacts"]["report_markdown"]).exists()
    assert Path(payload["artifacts"]["fairness_markdown"]).exists()

    strategies = {row["strategy"] for row in payload["rows"]}
    assert "constrained_nsga2_constraint_domination" in strategies
    assert "pymoo_constrained_nsga2" in strategies
    assert "random_pareto_archive" in strategies
    assert "fairness_summary" in payload
    assert "paired_comparisons" in payload
    assert payload["default_nsga2_changed"] is False
    assert payload["constrained_nsga2_scope_change"] == "none"

    for row in payload["rows"]:
        if row["status"] == "success":
            assert row["actual_evaluations"] == row["requested_budget"]
        else:
            assert row.get("skip_reason") or row.get("failure_reason")


def test_external_comparison_runner_dependency_skip_does_not_crash(tmp_path: Path) -> None:
    suffix = "skip_contract"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_constrained_nsga2_external.py",
            "--problems",
            "constrained_zdt_box_toy",
            "--seeds",
            "1",
            "--budget",
            "20",
            "--population-size",
            "4",
            "--artifact-suffix",
            suffix,
            "--output-dir",
            str(tmp_path),
            "--force-pymoo-skip",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        Path(json.loads(completed.stdout)["artifacts"]["json"]).read_text(encoding="utf-8")
    )

    pymoo_rows = [row for row in payload["rows"] if row["strategy"] == "pymoo_constrained_nsga2"]
    assert pymoo_rows
    assert all(row["status"] == "skipped" for row in pymoo_rows)
    assert payload["fairness_summary"]["status"] in {"pass", "warning"}
