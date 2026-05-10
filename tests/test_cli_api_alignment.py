from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.api import recommend_preset, recommend_solver, run_preset


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_output_dir(stdout: str) -> Path:
    for line in reversed(stdout.splitlines()):
        if line.startswith("Output directory: "):
            return Path(line.split(": ", 1)[1].strip())
    raise ValueError("missing output directory line")


def test_recommend_preset_cli_matches_public_api() -> None:
    api_payload = recommend_preset("zdt1", 50, "hv", format="dict")
    completed = subprocess.run(
        [
            sys.executable,
            str(_project_root() / "scripts" / "recommend_preset.py"),
            "--problem",
            "zdt1",
            "--size",
            "50",
            "--priority",
            "hv",
        ],
        cwd=_project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    cli_payload = json.loads(completed.stdout)

    assert cli_payload["preset_name"] == api_payload["preset_name"]
    assert cli_payload["resource_uri"] == api_payload["resource_uri"]


def test_recommend_solver_cli_matches_public_api() -> None:
    api_payload = recommend_solver("tsp", 20, "quality", format="dict")
    completed = subprocess.run(
        [
            sys.executable,
            str(_project_root() / "scripts" / "recommend_solver.py"),
            "--problem",
            "tsp",
            "--size",
            "20",
            "--priority",
            "quality",
        ],
        cwd=_project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    cli_payload = json.loads(completed.stdout)

    assert cli_payload["solver_family"] == api_payload["solver_family"]
    assert cli_payload["config_name"] == api_payload["config_name"]


def test_run_cli_matches_public_api(tmp_path) -> None:
    api_result = run_preset("onemax_small", output_root=tmp_path / "api")
    completed = subprocess.run(
        [
            sys.executable,
            str(_project_root() / "scripts" / "run_experiment.py"),
            "--preset",
            "onemax_small",
            "--output-root",
            str(tmp_path / "cli"),
        ],
        cwd=_project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    cli_output_dir = _parse_output_dir(completed.stdout)
    cli_summary = json.loads((cli_output_dir / "summary.json").read_text(encoding="utf-8"))

    assert api_result.metrics["best_fitness"] == cli_summary["best_fitness"]
    assert api_result.raw_summary["problem"] == cli_summary["problem"]
