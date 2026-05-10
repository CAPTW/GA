from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ALLOWED_RECOMMENDED_STATUS = {
    "ready_for_planned_implementation",
    "usable_with_version_guards",
    "dependency_missing",
    "api_import_failed",
    "hold_for_manual_review",
    "not_recommended_as_primary",
}


def _run_inspection(tmp_path: Path, suffix: str = "test_ext_api") -> dict[str, object]:
    script = Path("scripts/inspect_constrained_external_dependencies.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-suffix",
            suffix,
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    json_path = Path(payload["artifacts"]["json"])
    assert json_path.exists()
    return json.loads(json_path.read_text(encoding="utf-8"))


def test_inspection_script_handles_optional_dependencies_without_uncaught_errors(
    tmp_path: Path,
) -> None:
    payload = _run_inspection(tmp_path)

    assert "dependencies" in payload
    assert "inspection_decision" in payload
    assert set(payload["dependencies"]) >= {"pymoo", "deap"}

    for dependency_name in ("pymoo", "deap"):
        dependency = payload["dependencies"][dependency_name]
        assert dependency["recommended_status"] in ALLOWED_RECOMMENDED_STATUS
        if dependency["installed"]:
            assert dependency["import_status"] in {"imported", "api_import_failed"}
            assert dependency["version"] is not None
        else:
            assert dependency["import_status"] in {"dependency_missing", "skipped"}
            assert dependency["recommended_status"] == "dependency_missing"
            assert dependency["failures"] == []


def test_inspection_artifacts_respect_suffix_and_markdown_is_created(tmp_path: Path) -> None:
    suffix = "suffix_respected"
    payload = _run_inspection(tmp_path, suffix=suffix)

    artifacts = payload["artifacts"]
    assert suffix in artifacts["json"]
    assert suffix in artifacts["markdown"]
    assert Path(artifacts["markdown"]).exists()


def test_inspection_artifact_records_no_benchmark_or_default_change(tmp_path: Path) -> None:
    payload = _run_inspection(tmp_path, suffix="no_benchmark_marker")

    assert payload["benchmark_execution"] == "not_executed"
    assert payload["optimizer_execution"] == "not_executed"
    assert payload["default_nsga2_changed"] is False
    assert payload["constrained_nsga2_scope_change"] == "none"
    assert payload["external_comparator_implemented"] is False


def test_recommended_status_values_are_allowed(tmp_path: Path) -> None:
    payload = _run_inspection(tmp_path, suffix="allowed_status")

    for dependency in payload["dependencies"].values():
        assert dependency["recommended_status"] in ALLOWED_RECOMMENDED_STATUS

