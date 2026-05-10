from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.local_maintenance_audit import (
    build_local_maintenance_audit,
    build_local_reopen_criteria_check,
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


def test_local_reopen_criteria_check_generation(tmp_path: Path) -> None:
    report = build_local_reopen_criteria_check(
        project_root=_project_root(),
        output_json_path=tmp_path / "local_reopen_criteria_check.json",
        output_md_path=tmp_path / "local_reopen_criteria_check.md",
    )

    assert report["status"] == "PASS"
    assert any(
        check["name"].startswith("artifact:tsp:reopen_only_if")
        and check["status"] == "PASS"
        for check in report["checks"]
    )
    assert any(
        check["name"].startswith("doc:zdt1:")
        and check["status"] == "PASS"
        for check in report["checks"]
    )


def test_local_maintenance_audit_generation(tmp_path: Path) -> None:
    reopen_path = tmp_path / "local_reopen_criteria_check.json"
    build_local_reopen_criteria_check(
        project_root=_project_root(),
        output_json_path=reopen_path,
        output_md_path=tmp_path / "local_reopen_criteria_check.md",
    )

    audit = build_local_maintenance_audit(
        project_root=_project_root(),
        reopen_criteria_check_path=reopen_path,
        output_json_path=tmp_path / "local_maintenance_audit.json",
        output_md_path=tmp_path / "local_maintenance_audit.md",
    )

    assert audit["baseline_check_status"] == "PASS"
    assert audit["candidate_backlog_status"] == "PASS"
    assert audit["reopen_criteria_status"] == "PASS"
    assert audit["protocol_docs_status"] == "PASS"
    assert audit["final_maintenance_decision"] == "no_op_pass"


def test_run_local_maintenance_audit_script_smoke(tmp_path: Path) -> None:
    result = _run_json_command(
        "scripts/run_local_maintenance_audit.py",
        "--artifacts-dir",
        str(tmp_path / "artifacts"),
    )

    assert result["reopen_criteria_check"] == "PASS"
    assert result["maintenance_audit"]["final_maintenance_decision"] == "no_op_pass"


def test_local_maintenance_audit_docs_reference_real_commands() -> None:
    project_root = _project_root()
    reopen_doc = (project_root / "docs" / "local_reopen_criteria.md").read_text(
        encoding="utf-8"
    )
    workflow_doc = (project_root / "docs" / "local_candidate_workflow.md").read_text(
        encoding="utf-8"
    )
    protocol_doc = (project_root / "docs" / "local_protocol_guide.md").read_text(
        encoding="utf-8"
    )
    experiment_doc = (project_root / "docs" / "local_experiment_guide.md").read_text(
        encoding="utf-8"
    )
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    examples = (project_root / "examples" / "README.md").read_text(encoding="utf-8")

    assert "candidate manifest" in reopen_doc.lower()
    assert "python scripts/check_local_baseline.py" in reopen_doc
    assert "baseline drift" in workflow_doc.lower()
    assert "candidate improvement" in workflow_doc.lower()
    assert "anti-case / corridor suspicion or quality-sensitive final still goes" in protocol_doc
    assert "final safety still belongs to `Q`" in experiment_doc
    assert "future work must enter through a candidate manifest" in readme
    assert (
        "python scripts/run_local_protocol.py --problem tsp --mode final --anti-case-suspected"
        in examples
    )
