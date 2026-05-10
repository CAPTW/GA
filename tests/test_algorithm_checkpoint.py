from __future__ import annotations

import json
import random

import pytest

from ga_lab.config import GAConfig
from ga_lab.experiment.algorithm_checkpoint import (
    AlgorithmCheckpointState,
    CheckpointConfig,
    CheckpointMetadata,
    EvaluationBudgetState,
    PopulationCheckpoint,
    RNGCheckpoint,
    build_config_hash,
    capture_rng_state,
    checkpoint_from_dict,
    checkpoint_to_dict,
    load_checkpoint,
    restore_rng_state,
    summarize_operator_signature,
    summarize_problem_signature,
    validate_resume_compatibility,
    write_checkpoint_atomic,
)


def _config(**overrides) -> GAConfig:
    payload = {
        "run_name": "checkpoint_test",
        "problem": "onemax",
        "population_size": 6,
        "genome_length": 8,
        "generations": 4,
        "crossover_rate": 0.8,
        "mutation_rate": 0.05,
        "elitism": 1,
        "tournament_size": 3,
        "seed": 17,
        "maximize": True,
        "log_every": 1,
    }
    payload.update(overrides)
    return GAConfig(**payload)


class _Problem:
    name = "onemax"

    def fitness(self, genome):
        return sum(genome)


def _checkpoint(config: GAConfig | None = None) -> AlgorithmCheckpointState:
    config = config or _config()
    rng = random.Random(config.seed)
    return AlgorithmCheckpointState(
        metadata=CheckpointMetadata(
            schema_version="algorithm_checkpoint_phase1_v1",
            checkpoint_type="single_objective_ga",
            algorithm="single_objective_ga",
            problem=config.problem,
            seed=config.seed,
            created_at="2026-04-28T00:00:00Z",
            generation_index=1,
            actual_evaluations=12,
            requested_budget=36,
            config_hash=build_config_hash(config),
            operator_signature_summary=summarize_operator_signature(config),
            problem_signature_summary=summarize_problem_signature(config, _Problem()),
            completed=False,
            warnings=[],
        ),
        population=PopulationCheckpoint(
            decision_vectors=[[1.0, 0.0] for _ in range(config.population_size)],
            fitness_values=[1.0 for _ in range(config.population_size)],
            population_size=config.population_size,
            best_index=0,
            best_fitness=1.0,
            objective_direction=True,
            finite_validation_status="pass",
        ),
        rng=RNGCheckpoint(
            python_random_state=capture_rng_state(rng).python_random_state,
            numpy_random_state=None,
            algorithm_rng_state=None,
            rng_capture_complete=True,
            rng_warning=None,
        ),
        budget_state=EvaluationBudgetState(
            requested_budget=36,
            actual_evaluations=12,
            remaining_budget=24,
            generation_index=1,
            evaluations_per_generation=6,
            budget_policy="configured_evaluation_budget",
        ),
        resume_generation=2,
        history=[{"generation": 0, "best_fitness": 1.0}],
    )


def test_checkpoint_state_is_json_serializable() -> None:
    payload = checkpoint_to_dict(_checkpoint())
    encoded = json.dumps(payload, sort_keys=True)
    decoded = checkpoint_from_dict(json.loads(encoded))

    assert decoded.metadata.checkpoint_type == "single_objective_ga"
    assert decoded.population.fitness_values == [1.0] * _config().population_size


def test_config_hash_and_operator_signature_are_deterministic() -> None:
    config = _config(selection_options={"tournament_size": 3})

    assert build_config_hash(config) == build_config_hash(config)
    assert summarize_operator_signature(config) == summarize_operator_signature(config)


def test_same_config_compatibility_passes() -> None:
    config = _config()

    report = validate_resume_compatibility(
        _checkpoint(config),
        config=config,
        problem=_Problem(),
        requested_budget=36,
    )

    assert report.decision == "pass"
    assert report.failures == []


def test_config_hash_mismatch_fails() -> None:
    checkpoint = _checkpoint(_config())

    report = validate_resume_compatibility(
        checkpoint,
        config=_config(mutation_rate=0.2),
        problem=_Problem(),
        requested_budget=36,
    )

    assert report.decision == "fail"
    assert any("config_hash" in failure for failure in report.failures)


def test_operator_signature_mismatch_fails() -> None:
    checkpoint = _checkpoint(_config())

    report = validate_resume_compatibility(
        checkpoint,
        config=_config(mutation="scramble"),
        problem=_Problem(),
        requested_budget=36,
    )

    assert report.decision == "fail"
    assert any("operator_signature" in failure for failure in report.failures)


def test_budget_smaller_than_checkpoint_actual_evaluations_fails() -> None:
    report = validate_resume_compatibility(
        _checkpoint(),
        config=_config(),
        problem=_Problem(),
        requested_budget=8,
    )

    assert report.decision == "fail"
    assert any("budget" in failure for failure in report.failures)


def test_checkpoint_with_nan_fitness_fails_compatibility() -> None:
    checkpoint = _checkpoint()
    checkpoint.population.fitness_values = [1.0, float("nan")]

    report = validate_resume_compatibility(
        checkpoint,
        config=_config(),
        problem=_Problem(),
        requested_budget=36,
    )

    assert report.decision == "fail"
    assert any("finite" in failure for failure in report.failures)


def test_corrupted_checkpoint_load_fails_fast(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupted checkpoint"):
        load_checkpoint(path)


def test_atomic_write_creates_file_and_prevents_overwrite(tmp_path) -> None:
    config = CheckpointConfig(
        enabled=True,
        output_dir=tmp_path,
        run_id="safe_run",
        interval_generations=1,
    )
    checkpoint = _checkpoint()

    path = write_checkpoint_atomic(checkpoint, config)

    assert path.exists()
    with pytest.raises(FileExistsError):
        write_checkpoint_atomic(checkpoint, config)


def test_missing_rng_state_is_warning_not_failure() -> None:
    checkpoint = _checkpoint()
    checkpoint.rng.python_random_state = None
    checkpoint.rng.rng_capture_complete = False

    report = validate_resume_compatibility(
        checkpoint,
        config=_config(),
        problem=_Problem(),
        requested_budget=36,
    )

    assert report.decision == "warning"
    assert any("RNG" in warning for warning in report.warnings)


def test_python_rng_state_round_trip() -> None:
    source = random.Random(123)
    _ = [source.random() for _ in range(4)]
    checkpoint_rng = capture_rng_state(source)
    expected = [source.random() for _ in range(5)]

    restored = random.Random(999)
    restore_rng_state(restored, checkpoint_rng)

    assert [restored.random() for _ in range(5)] == expected
