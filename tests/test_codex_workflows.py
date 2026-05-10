from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_repo_local_skills_exist_for_standard_experiment_tasks() -> None:
    project_root = _project_root()
    skills_dir = project_root / "skills"

    expected = {
        "baseline-regression": [
            "tests/test_baseline_regression.py",
            "configs/ci/baseline_smoke.json",
            "configs/baselines/manifest.json",
        ],
        "add-problem": [
            "src/ga_lab/problems/",
            "tests/test_factory.py",
            "README.md",
        ],
        "summarize-results": [
            "scripts/summarize_results.py",
            "SUMMARY.md",
            "RUNS.csv",
        ],
    }

    for skill_name, snippets in expected.items():
        skill_dir = skills_dir / skill_name
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "agents" / "openai.yaml").exists()
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in content


def test_worktree_and_nightly_docs_capture_operating_rules() -> None:
    project_root = _project_root()
    worktree_rules = (project_root / "docs" / "worktree_rules.md").read_text(encoding="utf-8")
    nightly = (project_root / "docs" / "nightly_automation.md").read_text(encoding="utf-8")

    assert "codex/<role>/<task>" in worktree_rules
    assert "outputs/nightly/<timestamp>_nightly_regression/" in worktree_rules
    assert "make nightly" in nightly
    assert "configs/comparisons/onemax_operator_compare_10seeds.json" in nightly


def test_run_nightly_dry_run_writes_a_plan_and_status(tmp_path) -> None:
    project_root = _project_root()
    timestamp = "20260308T120000Z"

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_nightly.py"),
            "--output-root",
            str(tmp_path),
            "--timestamp",
            timestamp,
            "--dry-run",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = tmp_path / f"{timestamp}_nightly_regression"
    plan = json.loads((run_dir / "nightly_plan.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "nightly_status.json").read_text(encoding="utf-8"))

    assert plan["collection_name"] == "nightly_regression"
    assert plan["baseline_manifest"] == "configs/ci/baseline_smoke.json"
    assert plan["comparison_manifest"] == "configs/comparisons/onemax_operator_compare_10seeds.json"
    assert [step["name"] for step in plan["steps"]] == [
        "regression_tests",
        "baseline_smoke",
        "seed_sweep",
        "collection_summary",
    ]
    assert status["dry_run"] is True
    assert status["success"] is True
    assert {step["status"] for step in status["steps"]} == {"dry_run"}
