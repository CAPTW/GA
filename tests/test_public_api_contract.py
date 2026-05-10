from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_public_api_snapshot_check_passes() -> None:
    completed = subprocess.run(
        [sys.executable, str(_project_root() / "scripts" / "check_public_api.py")],
        cwd=_project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "passed" in completed.stdout.lower()


def test_public_api_snapshot_check_detects_drift(tmp_path) -> None:
    snapshot_path = tmp_path / "public_api_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "module_path": "ga_lab.api",
                "status": "stable_public_api",
                "exports": [],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(_project_root() / "scripts" / "check_public_api.py"),
            "--snapshot",
            str(snapshot_path),
        ],
        cwd=_project_root(),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "drift" in completed.stderr.lower()
