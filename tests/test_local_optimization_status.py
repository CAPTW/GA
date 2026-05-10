from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.local_optimization_status import (
    build_local_candidate_backlog_closeout,
    build_local_optimization_closeout,
    build_local_optimization_status,
    build_local_reopen_criteria,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_json_command(*args: str, cwd: Path | None = None) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=cwd or _project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_local_optimization_status_generation(tmp_path: Path) -> None:
    status = build_local_optimization_status(
        project_root=_project_root(),
        output_json_path=tmp_path / "local_optimization_status.json",
        output_md_path=tmp_path / "local_optimization_status.md",
    )

    assert status["cycle_state"] == "local_optimization_cycle_1_frozen"
    assert status["baseline_check_status"] == "PASS"
    assert status["no_candidate_ready_to_change_baseline"] is True
    assert "tsp_fast_anti_case_tail" in status["frozen_targets"]
    assert "zdt1_fast_joint_safety_fail" in status["monitor_only_targets"]


def test_local_reopen_criteria_schema(tmp_path: Path) -> None:
    criteria = build_local_reopen_criteria(
        project_root=_project_root(),
        output_json_path=tmp_path / "local_reopen_criteria.json",
        output_md_path=tmp_path / "local_reopen_criteria.md",
    )

    tsp = criteria["problems"]["tsp"]
    zdt1 = criteria["problems"]["zdt1"]
    assert "new mechanism hypothesis" in tsp["reopen_trigger"][0].lower()
    assert "candidate_passes_local_guard" in tsp["required_candidate_label"]
    assert any("stable" in item.lower() for item in zdt1["required_evidence"])


def test_local_candidate_backlog_closeout_generation(tmp_path: Path) -> None:
    backlog = build_local_candidate_backlog_closeout(
        project_root=_project_root(),
        output_json_path=tmp_path / "local_candidate_backlog_closeout.json",
        output_md_path=tmp_path / "local_candidate_backlog_closeout.md",
    )

    assert backlog["total_candidates"] >= 3
    assert backlog["no_candidate_is_ready_to_change_baseline"] is True
    assert any(
        row["candidate_id"] == "example_zdt1_spread_candidate_note_only"
        for row in backlog["note_only_candidates"]
    )
    assert any(
        row["candidate_id"] == "example_tsp_pg_contour_rejected"
        for row in backlog["requires_new_mechanism_candidates"]
    )


def test_summarize_local_optimization_status_script_smoke(tmp_path: Path) -> None:
    status = _run_json_command(
        "scripts/summarize_local_optimization_status.py",
        "--artifacts-dir",
        str(tmp_path / "artifacts"),
        "--docs-dir",
        str(tmp_path / "docs"),
    )

    assert status["baseline_check_status"] == "PASS"
    assert status["no_profile_changes_pending"] is True


def test_build_local_optimization_closeout_outputs_all_files(tmp_path: Path) -> None:
    result = build_local_optimization_closeout(
        project_root=_project_root(),
        status_json_path=tmp_path / "artifacts" / "local_optimization_status.json",
        status_md_path=tmp_path / "artifacts" / "local_optimization_status.md",
        reopen_json_path=tmp_path / "artifacts" / "local_reopen_criteria.json",
        reopen_md_path=tmp_path / "docs" / "local_reopen_criteria.md",
        backlog_json_path=tmp_path / "artifacts" / "local_candidate_backlog_closeout.json",
        backlog_md_path=tmp_path / "artifacts" / "local_candidate_backlog_closeout.md",
    )

    assert result["status"]["cycle_state"] == "local_optimization_cycle_1_frozen"
    assert result["candidate_backlog_closeout"]["closeout_read"]["zdt1_spread_candidate"] == (
        "note_only_stress_slice"
    )
    assert (tmp_path / "docs" / "local_reopen_criteria.md").exists()


def test_local_optimization_status_docs_reference_real_commands() -> None:
    project_root = _project_root()
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    protocol_guide = (project_root / "docs" / "local_protocol_guide.md").read_text(
        encoding="utf-8"
    )
    workflow_doc = (project_root / "docs" / "local_candidate_workflow.md").read_text(
        encoding="utf-8"
    )
    change_control_doc = (project_root / "docs" / "local_change_control.md").read_text(
        encoding="utf-8"
    )
    experiment_guide = (project_root / "docs" / "local_experiment_guide.md").read_text(
        encoding="utf-8"
    )
    reopen_doc = (project_root / "docs" / "local_reopen_criteria.md").read_text(
        encoding="utf-8"
    )
    examples = (project_root / "examples" / "README.md").read_text(encoding="utf-8")

    assert "python scripts/summarize_local_optimization_status.py" in readme
    assert "python scripts/summarize_local_optimization_status.py" in protocol_guide
    assert "python scripts/summarize_local_optimization_status.py" in workflow_doc
    assert "python scripts/summarize_local_optimization_status.py" in change_control_doc
    assert "python scripts/summarize_local_optimization_status.py" in experiment_guide
    assert "candidate manifest" in reopen_doc.lower()
    assert "python scripts/summarize_local_optimization_status.py" in examples
