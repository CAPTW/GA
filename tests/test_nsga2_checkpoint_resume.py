from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ga_lab.algorithms.nsga2 import run_nsga2
from ga_lab.config import GAConfig
from ga_lab.experiment.algorithm_checkpoint import CheckpointConfig
from ga_lab.experiment.nsga2_checkpoint import (
    checkpoint_to_dict,
    load_nsga2_checkpoint,
    validate_nsga2_resume_compatibility,
    write_nsga2_checkpoint_atomic,
)
from ga_lab.factory import build_runtime_context
from ga_lab.utils.seed import make_rng


def _config() -> GAConfig:
    return GAConfig(
        run_name="nsga2_checkpoint_resume_test",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=12,
        genome_length=6,
        generations=4,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        tournament_size=2,
        seed=17,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.05},
        log_every=1,
        algorithm_options={"_return_final_population": True},
    )


def _run(config: GAConfig, *, checkpoint_config: CheckpointConfig | None = None):
    runtime = build_runtime_context(config)
    rng = make_rng(config.seed)
    return run_nsga2(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=rng,
        checkpoint_config=checkpoint_config,
    )


def _source_checkpoint(tmp_path, config: GAConfig, *, generation: int = 1):
    source = CheckpointConfig(
        enabled=True,
        output_dir=tmp_path,
        run_id="nsga2_phase2c_source",
        interval_generations=1,
        stop_after_checkpoint_generation=generation,
        artifact_suffix="nsga2_checkpoint_phase2c",
    )
    _run(config, checkpoint_config=source)
    return tmp_path / "nsga2" / source.run_id / f"checkpoint_gen_{generation}.json"


def _strip_checkpoint_metadata(summary: dict) -> dict:
    cleaned = dict(summary)
    cleaned.pop("checkpointing", None)
    cleaned.pop("resume_metadata", None)
    return cleaned


def test_resume_from_checkpoint_matches_uninterrupted_small_zdt1(tmp_path) -> None:
    config = _config()
    uninterrupted_summary, uninterrupted_history = _run(config)
    checkpoint_path = _source_checkpoint(tmp_path, config, generation=1)

    resumed_summary, resumed_history = _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="nsga2_phase2c_resumed",
            interval_generations=1,
            resume_from=checkpoint_path,
            artifact_suffix="nsga2_checkpoint_phase2c",
        ),
    )

    assert _strip_checkpoint_metadata(resumed_summary) == uninterrupted_summary
    assert resumed_history == uninterrupted_history
    assert resumed_summary["actual_evaluations_used"] == uninterrupted_summary["actual_evaluations_used"]
    assert resumed_summary["pareto_front_vectors"] == uninterrupted_summary["pareto_front_vectors"]
    assert resumed_summary["final_population"] == uninterrupted_summary["final_population"]
    assert resumed_summary["resume_metadata"]["enabled"] is True
    assert resumed_summary["resume_metadata"]["resumed_from_generation"] == 1
    assert resumed_summary["resume_metadata"]["compatibility_decision"] == "pass"


def test_checkpoint_config_none_default_exact_match() -> None:
    config = _config()
    baseline_summary, baseline_history = _run(config)
    default_summary, default_history = _run(config, checkpoint_config=None)

    assert default_summary == baseline_summary
    assert default_history == baseline_history
    assert "resume_metadata" not in default_summary


def test_write_only_checkpoint_still_works(tmp_path) -> None:
    config = _config()
    summary, _ = _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="nsga2_phase2c_write_only",
            interval_generations=1,
        ),
    )

    assert summary["checkpointing"]["mode"] == "write_only"
    assert (tmp_path / "nsga2" / "nsga2_phase2c_write_only" / "checkpoint_gen_1.json").exists()


def test_resume_config_hash_mismatch_fails(tmp_path) -> None:
    config = _config()
    checkpoint_path = _source_checkpoint(tmp_path, config)
    changed_config = replace(config, mutation_rate=0.2)

    with pytest.raises(ValueError, match="config_hash mismatch"):
        _run(
            changed_config,
            checkpoint_config=CheckpointConfig(
                enabled=True,
                output_dir=tmp_path,
                run_id="nsga2_phase2c_config_mismatch",
                resume_from=checkpoint_path,
            ),
        )


def test_resume_budget_smaller_than_actual_evaluations_fails(tmp_path) -> None:
    config = _config()
    checkpoint_path = _source_checkpoint(tmp_path, config)
    smaller_budget_config = replace(config, generations=0)

    with pytest.raises(ValueError, match="budget smaller than checkpoint actual_evaluations"):
        _run(
            smaller_budget_config,
            checkpoint_config=CheckpointConfig(
                enabled=True,
                output_dir=tmp_path,
                run_id="nsga2_phase2c_budget_mismatch",
                resume_from=checkpoint_path,
            ),
        )


def test_resume_objective_count_mismatch_fails(tmp_path) -> None:
    config = _config()
    checkpoint_path = _source_checkpoint(tmp_path, config)
    checkpoint = load_nsga2_checkpoint(checkpoint_path)
    report = validate_nsga2_resume_compatibility(
        checkpoint,
        config=config,
        objective_count=3,
        objective_directions=[False, False, False],
        requested_budget=checkpoint.metadata.requested_budget,
    )

    assert report.decision == "fail"
    assert "objective_count mismatch" in report.failures


def test_resume_objective_direction_mismatch_fails(tmp_path) -> None:
    config = _config()
    checkpoint_path = _source_checkpoint(tmp_path, config)
    checkpoint = load_nsga2_checkpoint(checkpoint_path)
    report = validate_nsga2_resume_compatibility(
        checkpoint,
        config=config,
        objective_count=2,
        objective_directions=[True, False],
        requested_budget=checkpoint.metadata.requested_budget,
    )

    assert report.decision == "fail"
    assert "objective_directions mismatch" in report.failures


def test_resume_population_size_mismatch_fails(tmp_path) -> None:
    config = _config()
    checkpoint_path = _source_checkpoint(tmp_path, config)
    changed_config = replace(config, population_size=10)

    with pytest.raises(ValueError, match="population_size mismatch"):
        _run(
            changed_config,
            checkpoint_config=CheckpointConfig(
                enabled=True,
                output_dir=tmp_path,
                run_id="nsga2_phase2c_population_mismatch",
                resume_from=checkpoint_path,
            ),
        )


def test_resume_missing_rng_state_fails_for_deterministic_resume(tmp_path) -> None:
    config = _config()
    checkpoint_path = _source_checkpoint(tmp_path, config)
    checkpoint = load_nsga2_checkpoint(checkpoint_path)
    checkpoint.rng.python_random_state = None
    modified_path = write_nsga2_checkpoint_atomic(
        checkpoint,
        output_dir=tmp_path,
        run_id="nsga2_phase2c_missing_rng",
        allow_overwrite=False,
    )

    with pytest.raises(ValueError, match="RNG state missing"):
        _run(
            config,
            checkpoint_config=CheckpointConfig(
                enabled=True,
                output_dir=tmp_path,
                run_id="nsga2_phase2c_missing_rng_resume",
                resume_from=modified_path,
            ),
        )


def test_resume_nan_objective_checkpoint_fails(tmp_path) -> None:
    config = _config()
    checkpoint_path = _source_checkpoint(tmp_path, config)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["population"]["objective_values"][0][0] = "nan"
    broken_path = tmp_path / "nan_objective_checkpoint.json"
    broken_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="corrupted NSGA-II checkpoint"):
        _run(
            config,
            checkpoint_config=CheckpointConfig(
                enabled=True,
                output_dir=tmp_path,
                run_id="nsga2_phase2c_nan_resume",
                resume_from=broken_path,
            ),
        )


def test_resume_cached_rank_shape_mismatch_fails(tmp_path) -> None:
    config = _config()
    checkpoint_path = _source_checkpoint(tmp_path, config)
    checkpoint = load_nsga2_checkpoint(checkpoint_path)
    checkpoint.population.ranks = [0]

    report = validate_nsga2_resume_compatibility(
        checkpoint,
        config=config,
        objective_count=2,
        objective_directions=[False, False],
        requested_budget=checkpoint.metadata.requested_budget,
    )

    assert report.decision == "fail"
    assert any("rank length mismatch" in failure for failure in report.failures)
