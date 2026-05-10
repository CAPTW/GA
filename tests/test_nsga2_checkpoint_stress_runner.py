from __future__ import annotations

import json
from pathlib import Path

import pytest

from ga_lab.experiment.nsga2_checkpoint_stress import (
    Nsga2CheckpointStressConfig,
    run_nsga2_checkpoint_stress,
)


def _config(
    tmp_path: Path,
    *,
    suffix: str = "nsga2_checkpoint_stress_test",
) -> Nsga2CheckpointStressConfig:
    return Nsga2CheckpointStressConfig(
        problem="zdt1",
        seeds=1,
        budgets=[168],
        checkpoint_interval=1,
        interruption_policy="midpoint",
        artifact_suffix=suffix,
        output_dir=tmp_path,
        run_compatibility_negative_tests=True,
    )


def test_stress_runner_creates_json_artifact(tmp_path) -> None:
    artifact = run_nsga2_checkpoint_stress(_config(tmp_path))

    json_path = tmp_path / "nsga2_checkpoint_stress_results_nsga2_checkpoint_stress_test.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["exact_equivalence_results"]
    assert payload["compatibility_negative_tests"]
    assert payload["aggregate_summary"]["total_cases"] == 1
    assert artifact["artifact_paths"]["json"] == str(json_path)


def test_one_small_seed_budget_exact_equivalence_passes(tmp_path) -> None:
    artifact = run_nsga2_checkpoint_stress(_config(tmp_path))

    row = artifact["exact_equivalence_results"][0]
    assert row["actual_evaluations_match"] is True
    assert row["objective_matrix_match"] is True
    assert row["nondominated_set_match"] is True
    assert row["metrics_match"] is True
    assert row["history_match"] is True
    assert row["exact_or_equivalence_pass"] is True


def test_negative_tests_pass_as_expected(tmp_path) -> None:
    artifact = run_nsga2_checkpoint_stress(_config(tmp_path))
    negative = {row["test_name"]: row for row in artifact["compatibility_negative_tests"]}

    assert negative["config_hash_mismatch"]["pass"] is True
    assert negative["objective_count_mismatch"]["pass"] is True
    assert negative["budget_smaller_than_actual_evaluations"]["pass"] is True
    assert negative["corrupted_checkpoint_file"]["pass"] is True
    assert all(row["pass"] for row in negative.values())


def test_artifact_suffix_respected_and_overwrite_protected(tmp_path) -> None:
    config = _config(tmp_path, suffix="unique_nsga2_suffix")
    run_nsga2_checkpoint_stress(config)

    expected = tmp_path / "nsga2_checkpoint_stress_results_unique_nsga2_suffix.json"
    assert expected.exists()
    with pytest.raises(FileExistsError):
        run_nsga2_checkpoint_stress(config)


def test_checkpoint_config_none_default_path_unaffected(tmp_path) -> None:
    artifact = run_nsga2_checkpoint_stress(_config(tmp_path))

    assert artifact["default_changed"] is False
    assert artifact["aggregate_summary"]["default_path_unchanged"] is True


def test_no_constrained_external_or_production_metadata_created(tmp_path) -> None:
    artifact = run_nsga2_checkpoint_stress(_config(tmp_path))

    assert artifact["constrained_checkpoint_done"] is False
    assert artifact["external_comparator_checkpoint_done"] is False
    assert artifact["production_reliability_claim"] is False
