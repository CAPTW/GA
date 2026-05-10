from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import ga_lab


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_package_import_smoke() -> None:
    assert ga_lab.__version__
    assert ga_lab.GAConfig is not None
    assert ga_lab.run_experiment is not None


def test_pyproject_exposes_console_scripts() -> None:
    pyproject = tomllib.loads((_project_root() / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    package_data = pyproject["tool"]["setuptools"]["package-data"]["ga_lab"]

    assert "ga-lab-run" in scripts
    assert "ga-lab-recommend-solver" in scripts
    assert "ga-lab-render-release-artifacts" in scripts
    assert "ga-lab-demo" in scripts
    assert "builtin_resources/presets/*.json" in package_data


def test_demo_runner_lists_available_demos() -> None:
    completed = subprocess.run(
        [sys.executable, str(_project_root() / "scripts" / "run_demo_suite.py"), "--list"],
        cwd=_project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    demo_names = {row["demo"] for row in payload["available_demos"]}
    assert {"baseline", "pure-ga", "hybrid", "nsga2"} <= demo_names


def test_baseline_demo_smoke(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(_project_root() / "scripts" / "run_demo_suite.py"),
            "--demo",
            "baseline",
            "--output-root",
            str(tmp_path),
        ],
        cwd=_project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["baseline_label"] == "hill_climb"
    assert Path(payload["summary_path"]).exists()
    assert Path(tmp_path / "baseline" / "demo_result.json").exists()


def test_pure_ga_demo_smoke(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(_project_root() / "scripts" / "run_demo_suite.py"),
            "--demo",
            "pure-ga",
            "--output-root",
            str(tmp_path),
        ],
        cwd=_project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["demo"] == "pure-ga"
    assert Path(payload["summary_path"]).exists()


def test_docs_and_examples_reference_real_files() -> None:
    project_root = _project_root()
    assert (project_root / "docs" / "install.md").exists()
    assert (project_root / "docs" / "faq.md").exists()
    assert (project_root / "docs" / "python_api.md").exists()
    assert (project_root / "docs" / "api_stability.md").exists()
    assert (project_root / "examples" / "README.md").exists()
    assert (project_root / "examples" / "minimal_baseline" / "README.md").exists()
    assert (project_root / "examples" / "minimal_ga" / "README.md").exists()
    assert (project_root / "examples" / "minimal_hybrid" / "README.md").exists()
    assert (project_root / "examples" / "minimal_nsga2" / "README.md").exists()

    quickstart = (project_root / "docs" / "quickstart.md").read_text(encoding="utf-8")
    install = (project_root / "docs" / "install.md").read_text(encoding="utf-8")
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    faq = (project_root / "docs" / "faq.md").read_text(encoding="utf-8")
    python_api = (project_root / "docs" / "python_api.md").read_text(encoding="utf-8")
    api_stability = (project_root / "docs" / "api_stability.md").read_text(encoding="utf-8")

    assert "ga-lab-run --preset onemax_small" in quickstart
    assert "ga-lab-demo baseline" in quickstart
    assert "ga-lab-check-claims" in quickstart
    assert "from ga_lab.api import recommend_solver, run_preset" in quickstart
    assert "portable consumer path" in quickstart.lower()
    assert "installed consumer install" in install.lower()
    assert "from ga_lab.api import list_presets, recommend_solver, run_preset" in install
    assert "repo-only" in install.lower()
    assert "installed consumer mode" in readme.lower()
    assert "Stable Python API" in readme
    assert "repo-only" in faq.lower()
    assert "ga_lab.api" in python_api
    assert "public_api_snapshot.json" in api_stability


def test_release_workflow_smoke_exists() -> None:
    workflow = (_project_root() / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "python -m build" in workflow
    assert "twine check" in workflow
    assert "package_portability_smoke.py" in workflow
    assert "check_public_api.py" in workflow
    assert "scripts/check_claim_drift.py" in workflow


def test_ci_workflow_has_cross_platform_portable_smoke_matrix() -> None:
    workflow = (_project_root() / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "portable-smoke" in workflow
    assert "windows-latest" in workflow
    assert "ubuntu-latest" in workflow
    assert "macos-latest" in workflow
    assert "package_portability_smoke.py" in workflow
    assert "check_public_api.py" in workflow
