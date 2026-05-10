from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_recommend_preset_onemax_large_robust() -> None:
    project_root = _project_root()
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "recommend_preset.py"),
            "--problem",
            "onemax",
            "--size",
            "128",
            "--priority",
            "robust",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["tier"] == "large"
    assert payload["preset_path"] == "configs/presets/onemax_large.json"
    assert payload["priority"] == "robust"


def test_recommend_preset_zdt1_large_coverage() -> None:
    project_root = _project_root()
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "recommend_preset.py"),
            "--problem",
            "zdt1",
            "--size",
            "50",
            "--priority",
            "coverage",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["tier"] == "large"
    assert payload["preset_path"] == "configs/presets/zdt1_large.json"
    assert payload["priority"] == "coverage"


def test_recommend_preset_knapsack_large_default() -> None:
    project_root = _project_root()
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "recommend_preset.py"),
            "--problem",
            "knapsack",
            "--size",
            "80",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["preset_path"] == "configs/presets/knapsack_large.json"


def test_recommend_preset_rejects_unvalidated_size() -> None:
    project_root = _project_root()
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "recommend_preset.py"),
            "--problem",
            "tsp",
            "--size",
            "60",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Validated sizes" in completed.stderr
