from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ga_lab.local_change_request import build_local_baseline_change_request


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _zdt1_report_path() -> Path:
    return (
        _project_root()
        / "outputs"
        / "local_candidates"
        / "20260418T124918775291Z_example_zdt1_spread_candidate_note_only"
        / "candidate_report.json"
    )


def _tsp_report_path() -> Path:
    return (
        _project_root()
        / "outputs"
        / "local_candidates"
        / "20260418T124930243344Z_example_tsp_pg_contour_rejected"
        / "candidate_report.json"
    )


def _run_json_command(*args: str, cwd: Path | None = None) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=cwd or _project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_change_request_pack_generation_for_passed_candidate(tmp_path: Path) -> None:
    passed_report = json.loads(_zdt1_report_path().read_text(encoding="utf-8"))
    passed_report["decision_label"] = "candidate_passes_local_guard"
    passed_report_path = tmp_path / "passed_candidate_report.json"
    passed_report_path.write_text(
        json.dumps(passed_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = build_local_baseline_change_request(
        passed_report_path,
        project_root=_project_root(),
        output_root=tmp_path / "change_requests",
    )

    output_paths = result["output_paths"]
    assert Path(output_paths["change_request_json"]).exists()
    payload = json.loads(Path(output_paths["change_request_json"]).read_text(encoding="utf-8"))
    assert payload["snapshot_update_required"] is True
    assert payload["docs_update_required"] is True
    assert payload["draft_only"] is False


def test_forced_draft_change_request_smoke(tmp_path: Path) -> None:
    result = _run_json_command(
        "scripts/build_local_baseline_change_request.py",
        "--candidate-report",
        str(_zdt1_report_path()),
        "--output-root",
        str(tmp_path / "change_requests"),
        "--force-draft",
    )
    assert result["draft_only"] is True
    assert Path(result["output_paths"]["change_request_json"]).exists()


def test_blocked_change_request_without_force_for_rejected_candidate(tmp_path: Path) -> None:
    rejected_report = json.loads(_tsp_report_path().read_text(encoding="utf-8"))
    rejected_report["decision_label"] = "reject_no_material_gain"
    rejected_report_path = tmp_path / "rejected_candidate_report.json"
    rejected_report_path.write_text(
        json.dumps(rejected_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        build_local_baseline_change_request(
            rejected_report_path,
            project_root=_project_root(),
            output_root=tmp_path / "change_requests",
        )
