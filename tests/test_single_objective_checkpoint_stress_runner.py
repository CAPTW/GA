from __future__ import annotations

import json
from pathlib import Path

import pytest

from ga_lab.experiment.single_objective_checkpoint_stress import (
    SingleObjectiveCheckpointStressConfig,
    run_single_objective_checkpoint_stress,
)


def _config(tmp_path: Path, *, suffix: str = "checkpoint_stress_test") -> SingleObjectiveCheckpointStressConfig:
    return SingleObjectiveCheckpointStressConfig(
        problem="onemax",
        seeds=1,
        generations=4,
        population_size=6,
        genome_length=8,
        checkpoint_interval=1,
        interrupt_generation=1,
        artifact_suffix=suffix,
        output_dir=tmp_path,
        run_compatibility_negative_tests=True,
    )


def test_stress_runner_creates_json_artifact(tmp_path) -> None:
    artifact = run_single_objective_checkpoint_stress(_config(tmp_path))

    json_path = tmp_path / "single_objective_checkpoint_stress_results_checkpoint_stress_test.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["exact_match_results"]
    assert payload["compatibility_negative_tests"]
    assert payload["aggregate_summary"]["total_seeds"] == 1
    assert artifact["artifact_paths"]["json"] == str(json_path)


def test_one_small_seed_exact_match_passes(tmp_path) -> None:
    artifact = run_single_objective_checkpoint_stress(_config(tmp_path))

    row = artifact["exact_match_results"][0]
    assert row["best_fitness_match"] is True
    assert row["best_genome_match"] is True
    assert row["actual_evaluations_match"] is True
    assert row["history_match"] is True
    assert row["exact_match_pass"] is True


def test_negative_tests_pass_as_expected(tmp_path) -> None:
    artifact = run_single_objective_checkpoint_stress(_config(tmp_path))
    negative = {row["test_name"]: row for row in artifact["compatibility_negative_tests"]}

    assert negative["config_hash_mismatch"]["pass"] is True
    assert negative["budget_smaller_than_actual_evaluations"]["pass"] is True
    assert negative["corrupted_checkpoint_file"]["pass"] is True
    assert negative["missing_rng_state_warning"]["actual_result"] == "warning"
    assert all(row["pass"] for row in negative.values())


def test_artifact_suffix_respected_and_overwrite_protected(tmp_path) -> None:
    config = _config(tmp_path, suffix="unique_suffix")
    run_single_objective_checkpoint_stress(config)

    expected = tmp_path / "single_objective_checkpoint_stress_results_unique_suffix.json"
    assert expected.exists()
    with pytest.raises(FileExistsError):
        run_single_objective_checkpoint_stress(config)


def test_checkpoint_disabled_default_path_unaffected(tmp_path) -> None:
    artifact = run_single_objective_checkpoint_stress(_config(tmp_path))

    assert artifact["default_changed"] is False
    assert artifact["aggregate_summary"]["default_path_unchanged"] is True


def test_no_nsga2_or_constrained_checkpoint_metadata_created(tmp_path) -> None:
    artifact = run_single_objective_checkpoint_stress(_config(tmp_path))

    assert artifact["nsga2_checkpoint_done"] is False
    assert artifact["constrained_checkpoint_done"] is False
    assert artifact["production_reliability_claim"] is False
