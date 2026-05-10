from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.local_baseline import (
    build_local_baseline_snapshot,
    check_local_baseline,
    freeze_target_registry_statuses,
)
from ga_lab.local_experiments import load_local_study


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


def test_local_baseline_guard_studies_load() -> None:
    tsp = load_local_study("local_baseline_guard_tsp")
    zdt1 = load_local_study("local_baseline_guard_zdt1")
    knapsack = load_local_study("local_baseline_guard_knapsack")
    onemax = load_local_study("local_baseline_guard_onemax")

    assert tsp.problem == "tsp"
    assert tsp.analysis["tsp_tail_freeze"]["fast_variant"] == "current_fast"
    assert zdt1.problem == "zdt1"
    assert zdt1.analysis["zdt1_spread_candidate_boundary"]["candidate_variant"] == (
        "spread_pg_pop41_gen88"
    )
    assert knapsack.problem == "knapsack"
    assert "repair_only" in knapsack.variant_overrides
    assert onemax.problem == "onemax"
    assert "early_stop_reference" in onemax.variant_overrides


def test_local_baseline_snapshot_generation(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "local_baseline_snapshot.json"
    markdown_path = tmp_path / "local_baseline_snapshot.md"

    snapshot = build_local_baseline_snapshot(
        project_root=_project_root(),
        snapshot_path=snapshot_path,
        markdown_path=markdown_path,
    )

    assert snapshot_path.exists()
    assert markdown_path.exists()
    assert snapshot["problem_operating_decisions"]["tsp"]["target_status"] == (
        "tsp_fast_anti_case_tail = freeze_as_protocol_limitation"
    )
    assert (
        snapshot["frozen_target_decisions"]["zdt1_fast_spread_safety_fail"]["latest_decision"]
        == "note_only_stress_slice"
    )


def test_local_baseline_check_pass(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "local_baseline_snapshot.json"
    markdown_path = tmp_path / "local_baseline_snapshot.md"
    output_json = tmp_path / "local_baseline_check.json"
    output_md = tmp_path / "local_baseline_check.md"

    build_local_baseline_snapshot(
        project_root=_project_root(),
        snapshot_path=snapshot_path,
        markdown_path=markdown_path,
    )
    report = check_local_baseline(
        project_root=_project_root(),
        snapshot_path=snapshot_path,
        output_json_path=output_json,
        output_md_path=output_md,
    )

    assert report["status"] == "PASS"
    assert output_json.exists()
    assert output_md.exists()


def test_target_registry_statuses_match_baseline_freeze() -> None:
    rows = freeze_target_registry_statuses(_project_root())
    row_map = {row["target_id"]: row for row in rows}

    assert row_map["tsp_fast_anti_case_tail"]["latest_decision"] == (
        "freeze_as_protocol_limitation"
    )
    assert row_map["tsp_rescue_target_ambiguity"]["latest_decision"] == (
        "secondary_regression_slice"
    )
    assert row_map["zdt1_fast_spread_safety_fail"]["next_recommended_action"] == (
        "only_validate_if_candidate_generalizes_beyond_stress_slice"
    )
    assert row_map["zdt1_fast_joint_safety_fail"]["next_recommended_action"] == (
        "use_Q_for_final_safety"
    )
    assert row_map["knapsack_repair_boundary_subset_sum_tight_capacity"]["latest_decision"] == (
        "narrow_note_only"
    )
    assert row_map["onemax_no_active_target"]["latest_decision"] == "no_active_target"


def test_check_local_baseline_script_smoke(tmp_path: Path) -> None:
    del tmp_path
    report = _run_json_command(
        "scripts/check_local_baseline.py",
        "--write-snapshot",
        "--snapshot",
        "artifacts/test_local_baseline_snapshot.json",
        "--output-dir",
        "artifacts/test_local_baseline_smoke",
    )
    assert report["status"] == "PASS"


def test_local_baseline_docs_reference_real_commands() -> None:
    project_root = _project_root()
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    experiment_guide = (project_root / "docs" / "local_experiment_guide.md").read_text(
        encoding="utf-8"
    )
    protocol_guide = (project_root / "docs" / "local_protocol_guide.md").read_text(
        encoding="utf-8"
    )
    examples = (project_root / "examples" / "README.md").read_text(encoding="utf-8")

    assert "python scripts/check_local_baseline.py --write-snapshot" in readme
    assert "python scripts/run_local_sweep.py --study local_baseline_guard_tsp" in readme
    assert "python scripts/check_local_baseline.py" in experiment_guide
    assert "tsp_fast_anti_case_tail is frozen as a protocol limitation" in experiment_guide
    assert "zdt1_fast_joint_safety_fail" in protocol_guide
    assert "python scripts/run_local_sweep.py --study local_baseline_guard_zdt1" in protocol_guide
    assert "python scripts/check_local_baseline.py" in examples
    assert "python scripts/run_local_protocol.py --problem zdt1 --mode final" in examples
