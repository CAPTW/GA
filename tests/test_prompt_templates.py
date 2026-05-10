from __future__ import annotations

from pathlib import Path


def test_prompt_templates_exist_for_standard_codex_tasks() -> None:
    project_root = Path(__file__).resolve().parents[1]
    prompts_dir = project_root / "docs" / "prompts"

    expected_files = {
        "README.md",
        "add_selection.md",
        "add_problem.md",
        "baseline_compare.md",
        "review.md",
        "ci_failure_analysis.md",
        "baseline_regression.md",
        "update_readme.md",
    }
    assert expected_files.issubset({path.name for path in prompts_dir.iterdir()})


def test_add_problem_template_mentions_tests_and_readme() -> None:
    project_root = Path(__file__).resolve().parents[1]
    prompt = (project_root / "docs" / "prompts" / "add_problem.md").read_text(encoding="utf-8")

    assert "tests/test_problems.py" in prompt
    assert "README.md" in prompt
    assert "configs/baselines/manifest.json" in prompt


def test_add_selection_template_mentions_registry_and_baseline_compare() -> None:
    project_root = Path(__file__).resolve().parents[1]
    prompt = (project_root / "docs" / "prompts" / "add_selection.md").read_text(encoding="utf-8")

    assert "src/ga_lab/core/selection.py" in prompt
    assert "python scripts/run_baselines.py" in prompt
    assert "README.md" in prompt


def test_review_prompt_mentions_regressions_and_artifacts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    prompt = (project_root / "docs" / "prompts" / "review.md").read_text(encoding="utf-8")

    assert "reproducibility" in prompt
    assert "artifacts" in prompt
    assert "findings" in prompt


def test_ci_failure_prompt_mentions_logs_and_root_cause() -> None:
    project_root = Path(__file__).resolve().parents[1]
    prompt = (project_root / "docs" / "prompts" / "ci_failure_analysis.md").read_text(
        encoding="utf-8"
    )

    assert "root cause" in prompt
    assert "*.log" in prompt
    assert "failing command" in prompt


def test_baseline_regression_prompt_mentions_summary_artifacts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    prompt = (project_root / "docs" / "prompts" / "baseline_regression.md").read_text(
        encoding="utf-8"
    )

    assert "suite_summary.json" in prompt
    assert "RUNS.csv" in prompt
    assert "tests/test_baseline_regression.py" in prompt
