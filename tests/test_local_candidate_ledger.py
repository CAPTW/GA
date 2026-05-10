from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.local_candidate_ledger import (
    build_candidate_ledger,
    lifecycle_state_for_decision_label,
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


def test_decision_label_to_lifecycle_state_mapping() -> None:
    assert lifecycle_state_for_decision_label("reject_regression") == "rejected"
    assert lifecycle_state_for_decision_label("reject_no_material_gain") == "rejected"
    assert lifecycle_state_for_decision_label("note_only_stress_slice") == "note_only"
    assert lifecycle_state_for_decision_label("monitor_only") == "monitor"
    assert (
        lifecycle_state_for_decision_label("candidate_promising_needs_confirm")
        == "promising_needs_confirm"
    )
    assert lifecycle_state_for_decision_label("candidate_passes_local_guard") == "passed_local_guard"
    assert (
        lifecycle_state_for_decision_label("candidate_requires_new_mechanism_hypothesis")
        == "requires_new_mechanism"
    )
    assert lifecycle_state_for_decision_label("baseline_drift_detected") == "blocked_by_baseline_drift"
    assert (
        lifecycle_state_for_decision_label("intentional_baseline_change_required")
        == "ready_for_change_request"
    )


def test_candidate_ledger_generation(tmp_path: Path) -> None:
    result = build_candidate_ledger(
        project_root=_project_root(),
        ledger_json_path=tmp_path / "local_candidate_ledger.json",
        ledger_csv_path=tmp_path / "local_candidate_ledger.csv",
        ledger_md_path=tmp_path / "local_candidate_ledger.md",
        summary_json_path=tmp_path / "local_candidate_summary.json",
        summary_md_path=tmp_path / "local_candidate_summary.md",
    )

    ledger_rows = result["ledger"]["rows"]
    candidate_ids = {row["candidate_id"] for row in ledger_rows}
    assert "example_zdt1_spread_candidate_note_only" in candidate_ids
    assert "example_tsp_pg_contour_rejected" in candidate_ids
    assert "example_knapsack_repair_note" in candidate_ids

    zdt1_row = next(
        row for row in ledger_rows if row["candidate_id"] == "example_zdt1_spread_candidate_note_only"
    )
    assert zdt1_row["decision_label"] == "note_only_stress_slice"
    assert zdt1_row["lifecycle_state"] == "note_only"
    assert zdt1_row["baseline_change_candidate"] is False


def test_candidate_summary_generation(tmp_path: Path) -> None:
    result = build_candidate_ledger(
        project_root=_project_root(),
        ledger_json_path=tmp_path / "local_candidate_ledger.json",
        ledger_csv_path=tmp_path / "local_candidate_ledger.csv",
        ledger_md_path=tmp_path / "local_candidate_ledger.md",
        summary_json_path=tmp_path / "local_candidate_summary.json",
        summary_md_path=tmp_path / "local_candidate_summary.md",
    )
    summary = result["summary"]

    assert summary["counts_by_problem"]["tsp"] >= 1
    assert summary["counts_by_problem"]["zdt1"] >= 1
    assert summary["counts_by_problem"]["knapsack"] >= 1
    assert summary["counts_by_decision_label"]["note_only_stress_slice"] >= 2
    assert summary["counts_by_lifecycle_state"]["requires_new_mechanism"] >= 1


def test_summarize_local_candidates_script_smoke(tmp_path: Path) -> None:
    summary = _run_json_command(
        "scripts/summarize_local_candidates.py",
        "--artifacts-dir",
        str(tmp_path / "artifacts"),
    )
    assert summary["total_candidates"] >= 3


def test_local_candidate_ledger_docs_reference_real_commands() -> None:
    project_root = _project_root()
    workflow_doc = (project_root / "docs" / "local_candidate_workflow.md").read_text(
        encoding="utf-8"
    )
    change_control_doc = (project_root / "docs" / "local_change_control.md").read_text(
        encoding="utf-8"
    )
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    examples = (project_root / "examples" / "README.md").read_text(encoding="utf-8")

    assert "python scripts/summarize_local_candidates.py" in workflow_doc
    assert "python scripts/build_local_baseline_change_request.py" in workflow_doc
    assert "python scripts/build_local_baseline_change_request.py" in change_control_doc
    assert "python scripts/summarize_local_candidates.py" in readme
    assert "python scripts/build_local_baseline_change_request.py" in examples
