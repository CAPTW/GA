from __future__ import annotations

import inspect
import json
import math
import random
from dataclasses import replace

import pytest

from ga_lab.algorithms.nsga2 import run_nsga2
from ga_lab.config import GAConfig
from ga_lab.experiment.nsga2_checkpoint import (
    NSGA2_ALGORITHM_NAME,
    NSGA2_CHECKPOINT_TYPE,
    NSGA2_SCHEMA_VERSION,
    Nsga2CheckpointMetadata,
    Nsga2CheckpointState,
    Nsga2EvaluationBudgetState,
    Nsga2PopulationCheckpoint,
    Nsga2RNGCheckpoint,
    build_nsga2_config_hash,
    capture_nsga2_rng_state,
    checkpoint_from_dict,
    checkpoint_to_dict,
    load_nsga2_checkpoint,
    summarize_nsga2_operator_signature,
    summarize_nsga2_problem_signature,
    validate_cached_rank_crowding,
    validate_nsga2_resume_compatibility,
    write_nsga2_checkpoint_atomic,
)


def _config(**overrides) -> GAConfig:
    payload = {
        "run_name": "nsga2_checkpoint_schema_test",
        "problem": "zdt1",
        "algorithm": "nsga2",
        "representation": "real",
        "selection": "tournament",
        "crossover": "arithmetic",
        "mutation": "gaussian",
        "population_size": 4,
        "genome_length": 3,
        "generations": 4,
        "crossover_rate": 0.9,
        "mutation_rate": 0.1,
        "elitism": 1,
        "tournament_size": 2,
        "seed": 7,
        "maximize": False,
        "objective_directions": [False, False],
        "representation_options": {"low": 0.0, "high": 1.0},
        "mutation_options": {"sigma": 0.05},
    }
    payload.update(overrides)
    return GAConfig(**payload)


def _checkpoint(config: GAConfig | None = None) -> Nsga2CheckpointState:
    config = config or _config()
    objective_directions = [False, False]
    objective_count = 2
    metadata = Nsga2CheckpointMetadata(
        schema_version=NSGA2_SCHEMA_VERSION,
        checkpoint_type=NSGA2_CHECKPOINT_TYPE,
        algorithm=NSGA2_ALGORITHM_NAME,
        problem="zdt1",
        seed=7,
        created_at="2026-04-29T00:00:00Z",
        generation_index=2,
        actual_evaluations=40,
        requested_budget=100,
        config_hash=build_nsga2_config_hash(config),
        operator_signature_summary=summarize_nsga2_operator_signature(config),
        problem_signature_summary=summarize_nsga2_problem_signature(
            config,
            objective_count=objective_count,
            objective_directions=objective_directions,
        ),
        objective_count=objective_count,
        objective_directions=objective_directions,
        completed=False,
    )
    population = Nsga2PopulationCheckpoint(
        decision_vectors=[
            [0.0, 0.2, 0.4],
            [0.1, 0.3, 0.5],
            [0.6, 0.7, 0.8],
            [0.9, 0.4, 0.2],
        ],
        objective_values=[
            [0.0, 1.0],
            [0.1, 0.8],
            [0.6, 0.4],
            [0.9, 0.2],
        ],
        population_size=4,
        objective_count=objective_count,
        ranks=[0, 0, 1, 1],
        crowding_distances=[math.inf, 1.0, 0.5, math.inf],
        front_indices=[[0, 1], [2, 3]],
        finite_validation_status="pass",
    )
    return Nsga2CheckpointState(
        metadata=metadata,
        population=population,
        rng=capture_nsga2_rng_state(random.Random(7)),
        budget_state=Nsga2EvaluationBudgetState(
            requested_budget=100,
            actual_evaluations=40,
            remaining_budget=60,
            generation_index=2,
            evaluations_per_generation=None,
        ),
    )


def test_nsga2_checkpoint_metadata_and_population_are_json_serializable() -> None:
    payload = checkpoint_to_dict(_checkpoint())

    text = json.dumps(payload, allow_nan=False, sort_keys=True)
    restored = checkpoint_from_dict(json.loads(text))

    assert restored.metadata.schema_version == NSGA2_SCHEMA_VERSION
    assert restored.metadata.objective_count == 2
    assert restored.metadata.objective_directions == [False, False]
    assert restored.population.ranks == [0, 0, 1, 1]
    assert restored.population.crowding_distances[0] == "inf"


def test_same_config_compatibility_passes() -> None:
    config = _config()
    report = validate_nsga2_resume_compatibility(
        _checkpoint(config),
        config=config,
        objective_count=2,
        objective_directions=[False, False],
        requested_budget=100,
    )

    assert report.decision == "pass"
    assert report.failures == []


@pytest.mark.parametrize(
    ("changed_config", "expected_failure"),
    [
        (_config(mutation_rate=0.2), "config_hash mismatch"),
        (_config(mutation="uniform_reset"), "operator_signature mismatch"),
        (_config(population_size=5), "population_size mismatch"),
    ],
)
def test_compatibility_failures_for_config_operator_and_population(
    changed_config: GAConfig,
    expected_failure: str,
) -> None:
    report = validate_nsga2_resume_compatibility(
        _checkpoint(_config()),
        config=changed_config,
        objective_count=2,
        objective_directions=[False, False],
        requested_budget=100,
    )

    assert report.decision == "fail"
    assert expected_failure in report.failures


def test_objective_count_and_direction_mismatch_fail() -> None:
    count_report = validate_nsga2_resume_compatibility(
        _checkpoint(),
        config=_config(),
        objective_count=3,
        objective_directions=[False, False, False],
    )
    direction_report = validate_nsga2_resume_compatibility(
        _checkpoint(),
        config=_config(),
        objective_count=2,
        objective_directions=[True, False],
    )

    assert count_report.decision == "fail"
    assert "objective_count mismatch" in count_report.failures
    assert direction_report.decision == "fail"
    assert "objective_directions mismatch" in direction_report.failures


def test_budget_smaller_than_actual_evaluations_fails() -> None:
    report = validate_nsga2_resume_compatibility(
        _checkpoint(),
        config=_config(),
        objective_count=2,
        objective_directions=[False, False],
        requested_budget=39,
    )

    assert report.decision == "fail"
    assert "budget smaller than checkpoint actual_evaluations" in report.failures


def test_checkpoint_with_nan_objective_fails() -> None:
    checkpoint = _checkpoint()
    checkpoint.population.objective_values[0][1] = float("nan")

    with pytest.raises(ValueError, match="non-finite objective"):
        checkpoint_to_dict(checkpoint)


def test_corrupted_checkpoint_load_fails_fast(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupted NSGA-II checkpoint"):
        load_nsga2_checkpoint(path)


def test_atomic_write_and_overwrite_protection(tmp_path) -> None:
    checkpoint = _checkpoint()
    written = write_nsga2_checkpoint_atomic(
        checkpoint,
        output_dir=tmp_path,
        run_id="phase2a_safe_run",
    )

    assert written.exists()
    assert load_nsga2_checkpoint(written).metadata.algorithm == "nsga2"
    with pytest.raises(FileExistsError):
        write_nsga2_checkpoint_atomic(
            checkpoint,
            output_dir=tmp_path,
            run_id="phase2a_safe_run",
        )


def test_missing_rng_state_fails_deterministic_resume() -> None:
    checkpoint = _checkpoint()
    checkpoint.rng = Nsga2RNGCheckpoint(
        python_random_state=None,
        rng_capture_complete=False,
        rng_warning="RNG state missing; deterministic true resume unavailable",
    )

    report = validate_nsga2_resume_compatibility(
        checkpoint,
        config=_config(),
        objective_count=2,
        objective_directions=[False, False],
    )

    assert report.decision == "fail"
    assert "RNG state missing; deterministic true resume unavailable" in report.failures


def test_cached_rank_crowding_shape_mismatch_fails() -> None:
    report = validate_cached_rank_crowding(
        objective_values=_checkpoint().population.objective_values,
        cached_ranks=[0, 1],
        cached_crowding_distances=[0.0, 1.0, 2.0, math.inf],
        objective_directions=[False, False],
    )

    assert report.decision == "fail"
    assert "rank length mismatch" in report.failures


def test_cached_rank_crowding_allows_boundary_infinity() -> None:
    report = validate_cached_rank_crowding(
        objective_values=_checkpoint().population.objective_values,
        cached_ranks=[0, 0, 1, 1],
        cached_crowding_distances=[math.inf, 1.0, 0.5, math.inf],
        objective_directions=[False, False],
    )

    assert report.decision == "pass"
    assert report.failures == []


def test_run_nsga2_signature_has_optional_checkpoint_config() -> None:
    signature = inspect.signature(run_nsga2)

    assert "checkpoint_config" in signature.parameters
    assert signature.parameters["checkpoint_config"].default is None
