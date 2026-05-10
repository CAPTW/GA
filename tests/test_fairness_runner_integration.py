from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_runner(
    repo_root: Path,
    tmp_path: Path,
    script_name: str,
    *,
    artifact_dir_name: str,
    output_dir_name: str,
    seed_start: int,
    budget: int = 760,
) -> dict[str, object]:
    artifact_root = tmp_path / artifact_dir_name
    output_root = tmp_path / output_dir_name
    completed = subprocess.run(
        [
            sys.executable,
            script_name,
            "--problems",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            str(seed_start),
            "--budget",
            str(budget),
            "--artifact-suffix",
            "fairness_test",
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
    assert completed.returncode == 0
    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    if lines:
        stdout_payload = json.loads(lines[-1])
        results_json = Path(str(stdout_payload["results_json"]))
    else:
        default_result_names = {
            "scripts/validate_nsga2_candidate_suite.py": "nsga2_candidate_suite_validation_results_fairness_test.json",
            "scripts/validate_nsga2_diversity_candidates.py": "nsga2_diversity_candidate_results_fairness_test.json",
            "scripts/validate_nsga2_h_lite_candidates.py": "nsga2_h_lite_candidate_results_fairness_test.json",
            "scripts/validate_nsga2_boundary_phase0.py": "nsga2_boundary_preservation_phase0_results_fairness_test.json",
        }
        results_json = artifact_root / default_result_names[script_name]
    return json.loads(results_json.read_text(encoding="utf-8"))


def test_compare_mo_baselines_writes_fairness_payload(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_mo_baselines.py",
            "--seeds",
            "1",
            "--seed-start",
            "6301",
            "--budget",
            "80",
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

    assert completed.returncode == 0
    payload = json.loads((artifact_root / "mo_baseline_comparison_results.json").read_text(encoding="utf-8"))
    assert "fairness" in payload
    assert payload["fairness"]["status"] in {"pass", "warning", "fail"}
    assert payload["fairness"]["issues"]


def test_candidate_suite_runner_writes_fairness_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = _run_runner(
        repo_root,
        tmp_path,
        "scripts/validate_nsga2_candidate_suite.py",
        artifact_dir_name="candidate-suite-artifacts",
        output_dir_name="candidate-suite-outputs",
        seed_start=8801,
    )

    assert "fairness" in payload
    assert "fairness_summary" in payload
    assert payload["fairness"]["status"] in {"pass", "warning", "fail"}
    assert isinstance(payload["fairness_summary"]["warning"], int)


def test_diversity_runner_writes_fairness_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = _run_runner(
        repo_root,
        tmp_path,
        "scripts/validate_nsga2_diversity_candidates.py",
        artifact_dir_name="diversity-artifacts",
        output_dir_name="diversity-outputs",
        seed_start=8901,
    )

    assert "fairness" in payload
    assert "fairness_summary" in payload
    assert payload["fairness"]["issues"]


def test_h_lite_runner_writes_fairness_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = _run_runner(
        repo_root,
        tmp_path,
        "scripts/validate_nsga2_h_lite_candidates.py",
        artifact_dir_name="h-lite-artifacts",
        output_dir_name="h-lite-outputs",
        seed_start=9001,
    )

    assert "fairness" in payload
    assert "fairness_summary" in payload
    assert payload["fairness"]["status"] in {"pass", "warning", "fail"}


def test_boundary_phase0_runner_writes_fairness_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = _run_runner(
        repo_root,
        tmp_path,
        "scripts/validate_nsga2_boundary_phase0.py",
        artifact_dir_name="boundary-phase0-artifacts",
        output_dir_name="boundary-phase0-outputs",
        seed_start=9101,
        budget=300,
    )

    assert "fairness" in payload
    assert "fairness_summary" in payload
    assert payload["fairness"]["status"] in {"pass", "warning", "fail"}
