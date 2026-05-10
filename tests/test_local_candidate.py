from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.local_candidate import (
    evaluate_candidate_manifest,
    load_candidate_manifest,
    validate_candidate_manifest,
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


def test_candidate_schema_loads() -> None:
    schema_path = _project_root() / "configs" / "local_candidates" / "candidate_schema.json"
    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    assert payload["title"] == "Local Candidate Experiment Manifest"
    assert "candidate_id" in payload["required"]
    assert "expected_decision_label" in payload["properties"]


def test_candidate_manifest_validation() -> None:
    manifest_path = (
        _project_root() / "configs" / "local_candidates" / "example_zdt1_candidate.json"
    )
    manifest = load_candidate_manifest(manifest_path)
    validated = validate_candidate_manifest(manifest)

    assert manifest["candidate_id"] == "example_zdt1_spread_candidate_note_only"
    assert validated["problem"] == "zdt1"
    assert validated["expected_decision_label"] == "note_only_stress_slice"


def test_baseline_snapshot_mismatch_warns_before_candidate_comparison(tmp_path: Path) -> None:
    project_root = _project_root()
    snapshot_path = project_root / "artifacts" / "local_baseline_snapshot.json"
    mutated_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    mutated_snapshot["candidate_schema_hash"] = "not-the-current-schema"
    mismatch_snapshot_path = tmp_path / "mismatch_snapshot.json"
    mismatch_snapshot_path.write_text(
        json.dumps(mutated_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = load_candidate_manifest(
        project_root / "configs" / "local_candidates" / "example_zdt1_candidate.json"
    )
    manifest["baseline_snapshot_path"] = str(mismatch_snapshot_path)

    report = evaluate_candidate_manifest(
        manifest,
        project_root=project_root,
        output_root=tmp_path / "candidate_reports",
        use_existing_output=True,
    )

    assert report["baseline_status"] == "FAIL"
    assert report["decision_label"] == "baseline_drift_detected"


def test_candidate_report_generation_from_existing_output(tmp_path: Path) -> None:
    report = evaluate_candidate_manifest(
        _project_root() / "configs" / "local_candidates" / "example_zdt1_candidate.json",
        project_root=_project_root(),
        output_root=tmp_path / "candidate_reports",
        use_existing_output=True,
    )

    output_paths = report["output_paths"]
    assert Path(output_paths["candidate_report_json"]).exists()
    assert Path(output_paths["candidate_report_md"]).exists()
    assert Path(output_paths["candidate_summary_csv"]).exists()
    assert Path(output_paths["candidate_vs_baseline_csv"]).exists()


def test_expected_decision_labels_match_examples(tmp_path: Path) -> None:
    project_root = _project_root()
    zdt1_report = evaluate_candidate_manifest(
        project_root / "configs" / "local_candidates" / "example_zdt1_candidate.json",
        project_root=project_root,
        output_root=tmp_path / "zdt1",
        use_existing_output=True,
    )
    tsp_report = evaluate_candidate_manifest(
        project_root / "configs" / "local_candidates" / "example_tsp_candidate.json",
        project_root=project_root,
        output_root=tmp_path / "tsp",
        no_execute=True,
    )
    knapsack_report = evaluate_candidate_manifest(
        project_root / "configs" / "local_candidates" / "example_knapsack_candidate.json",
        project_root=project_root,
        output_root=tmp_path / "knapsack",
        use_existing_output=True,
    )

    assert zdt1_report["decision_label"] == "note_only_stress_slice"
    assert tsp_report["decision_label"] == "candidate_requires_new_mechanism_hypothesis"
    assert knapsack_report["decision_label"] == "note_only_stress_slice"


def test_run_local_candidate_script_smoke(tmp_path: Path) -> None:
    report = _run_json_command(
        "scripts/run_local_candidate.py",
        "--candidate",
        "configs/local_candidates/example_zdt1_candidate.json",
        "--output-root",
        str(tmp_path / "candidate_reports"),
        "--use-existing-output",
    )
    assert report["decision_label"] == "note_only_stress_slice"


def test_local_candidate_docs_reference_real_commands() -> None:
    project_root = _project_root()
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    experiment_guide = (project_root / "docs" / "local_experiment_guide.md").read_text(
        encoding="utf-8"
    )
    protocol_guide = (project_root / "docs" / "local_protocol_guide.md").read_text(
        encoding="utf-8"
    )
    workflow_doc = (project_root / "docs" / "local_candidate_workflow.md").read_text(
        encoding="utf-8"
    )
    examples = (project_root / "examples" / "README.md").read_text(encoding="utf-8")

    assert "python scripts/check_local_baseline.py" in readme
    assert (
        "python scripts/run_local_candidate.py --candidate configs/local_candidates/example_zdt1_candidate.json"
        in readme
    )
    assert "candidate_requires_new_mechanism_hypothesis" in experiment_guide
    assert "note_only_stress_slice" in experiment_guide
    assert "python scripts/run_local_candidate.py --candidate configs/local_candidates/example_tsp_candidate.json --no-execute" in workflow_doc
    assert "baseline_drift_detected" in workflow_doc
    assert "spread_pg_pop41_gen88" in protocol_guide
    assert (
        "python scripts/run_local_candidate.py --candidate configs/local_candidates/example_knapsack_candidate.json --use-existing-output"
        in examples
    )
