from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_render_release_artifacts_smoke(tmp_path) -> None:
    project_root = _project_root()
    docs_dir = tmp_path / "docs"
    artifacts_dir = tmp_path / "artifacts"
    docs_dir.mkdir()
    artifacts_dir.mkdir()

    benchmark_card = docs_dir / "benchmark_card.md"
    benchmark_card.write_text(
        "# Benchmark Card\n\n"
        "<!-- BEGIN AUTO-GENERATED: benchmark-card -->\nold\n"
        "<!-- END AUTO-GENERATED: benchmark-card -->\n",
        encoding="utf-8",
    )
    solver_matrix = docs_dir / "solver_matrix.md"
    solver_matrix.write_text(
        "# Solver Matrix\n\n"
        "<!-- BEGIN AUTO-GENERATED: solver-matrix -->\nold\n"
        "<!-- END AUTO-GENERATED: solver-matrix -->\n",
        encoding="utf-8",
    )
    release_status = docs_dir / "release_status.md"
    release_status.write_text(
        "# Release Status\n\n"
        "<!-- BEGIN AUTO-GENERATED: release-status -->\nold\n"
        "<!-- END AUTO-GENERATED: release-status -->\n",
        encoding="utf-8",
    )
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "# Temp README\n\n"
        "<!-- BEGIN AUTO-GENERATED: solver-matrix-preview -->\nold\n"
        "<!-- END AUTO-GENERATED: solver-matrix-preview -->\n\n"
        "<!-- BEGIN AUTO-GENERATED: evidence-snapshot -->\nold\n"
        "<!-- END AUTO-GENERATED: evidence-snapshot -->\n\n"
        "<!-- BEGIN AUTO-GENERATED: release-status -->\nold\n"
        "<!-- END AUTO-GENERATED: release-status -->\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "render_release_artifacts.py"),
            "--manifest",
            str(project_root / "configs" / "release" / "release_artifacts_manifest.json"),
            "--registry",
            str(project_root / "claims" / "claim_registry.json"),
            "--drift-report",
            str(project_root / "outputs" / "benchmark_summary" / "claim_drift_report.json"),
            "--docs-dir",
            str(docs_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--readme-path",
            str(readme_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )

    claim_matrix = json.loads((artifacts_dir / "claim_matrix.json").read_text(encoding="utf-8"))
    solver_matrix_payload = json.loads(
        (artifacts_dir / "solver_matrix.json").read_text(encoding="utf-8")
    )
    governance_badge = json.loads(
        (artifacts_dir / "badges" / "governance_status.json").read_text(encoding="utf-8")
    )
    release_snapshot = json.loads(
        (artifacts_dir / "release_snapshot.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (project_root / "claims" / "claim_registry.json").read_text(encoding="utf-8")
    )
    registry_ids = {row["claim_id"] for row in registry["claims"]}

    assert claim_matrix["rows"]
    assert solver_matrix_payload["rows"]
    assert governance_badge["message"] == "pass"
    assert release_snapshot["drift_overall_status"] == "PASS"
    assert {row["claim_id"] for row in claim_matrix["rows"]}.issubset(registry_ids)

    readme_text = readme_path.read_text(encoding="utf-8")
    assert "old" not in readme_text
    assert "drift-governed ci status" in readme_text
    assert "`nearest_neighbor_2opt`" in readme_text

    benchmark_card_text = benchmark_card.read_text(encoding="utf-8")
    assert "## Benchmark Fairness" in benchmark_card_text
    assert "`configured_budget`" in benchmark_card_text

    solver_matrix_text = solver_matrix.read_text(encoding="utf-8")
    assert "problem_family" in solver_matrix_text
    assert "`configs/presets/tsp_medium_hybrid.json`" in solver_matrix_text

    release_status_text = release_status.read_text(encoding="utf-8")
    assert "## Current Drift Status" in release_status_text
    assert "`PASS`" in release_status_text
