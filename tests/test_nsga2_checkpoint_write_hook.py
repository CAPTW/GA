from __future__ import annotations

import pytest

from ga_lab.algorithms.nsga2 import run_nsga2
from ga_lab.config import GAConfig
from ga_lab.experiment.algorithm_checkpoint import CheckpointConfig
from ga_lab.experiment.nsga2_checkpoint import (
    load_nsga2_checkpoint,
    validate_nsga2_resume_compatibility,
)
from ga_lab.factory import build_runtime_context
from ga_lab.utils.seed import make_rng


def _config() -> GAConfig:
    return GAConfig(
        run_name="nsga2_checkpoint_write_hook_test",
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
        seed=13,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.05},
        log_every=1,
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


def test_checkpoint_config_none_preserves_default_summary_and_history() -> None:
    config = _config()
    baseline_summary, baseline_history = _run(config)
    default_summary, default_history = _run(config, checkpoint_config=None)

    assert default_summary == baseline_summary
    assert default_history == baseline_history
    assert "checkpointing" not in default_summary


def test_checkpoint_enabled_writes_generation_boundary_state(tmp_path) -> None:
    config = _config()
    checkpoint_config = CheckpointConfig(
        enabled=True,
        output_dir=tmp_path,
        run_id="nsga2_phase2b_write_hook",
        interval_generations=1,
        artifact_suffix="nsga2_checkpoint_phase2b",
    )

    summary, _ = _run(config, checkpoint_config=checkpoint_config)
    checkpoint_path = tmp_path / "nsga2" / "nsga2_phase2b_write_hook" / "checkpoint_gen_1.json"
    checkpoint = load_nsga2_checkpoint(checkpoint_path)
    report = validate_nsga2_resume_compatibility(
        checkpoint,
        config=config,
        objective_count=2,
        objective_directions=[False, False],
        requested_budget=checkpoint.metadata.requested_budget,
    )

    assert checkpoint_path.exists()
    assert checkpoint.population.decision_vectors
    assert checkpoint.population.objective_values
    assert checkpoint.population.ranks is not None
    assert checkpoint.population.crowding_distances is not None
    assert checkpoint.population.front_indices is not None
    assert checkpoint.metadata.actual_evaluations > 0
    assert checkpoint.metadata.generation_index == 1
    assert checkpoint.rng.python_random_state is not None
    assert report.decision == "pass"
    assert summary["checkpointing"]["enabled"] is True
    assert str(checkpoint_path) in summary["checkpointing"]["checkpoint_paths"]


def test_checkpoint_enabled_full_run_does_not_change_final_result(tmp_path) -> None:
    config = _config()
    baseline_summary, baseline_history = _run(config)
    checkpoint_summary, checkpoint_history = _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="nsga2_phase2b_full_observational",
            interval_generations=2,
        ),
    )
    checkpoint_summary_without_metadata = dict(checkpoint_summary)
    checkpoint_summary_without_metadata.pop("checkpointing", None)

    assert checkpoint_summary_without_metadata == baseline_summary
    assert checkpoint_history == baseline_history


def test_checkpoint_overwrite_protection_raises(tmp_path) -> None:
    config = _config()
    checkpoint_config = CheckpointConfig(
        enabled=True,
        output_dir=tmp_path,
        run_id="nsga2_phase2b_overwrite",
        interval_generations=1,
    )

    _run(config, checkpoint_config=checkpoint_config)
    with pytest.raises(FileExistsError):
        _run(config, checkpoint_config=checkpoint_config)


def test_stop_after_checkpoint_generation_debug_stop(tmp_path) -> None:
    config = _config()
    summary, history = _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="nsga2_phase2b_debug_stop",
            interval_generations=1,
            stop_after_checkpoint_generation=1,
        ),
    )

    assert summary["stop_reason"] == "checkpoint_debug_stop"
    assert summary["checkpointing"]["debug_only"] is True
    assert summary["checkpointing"]["debug_stop_after_checkpoint_generation"] == 1
    assert (tmp_path / "nsga2" / "nsga2_phase2b_debug_stop" / "checkpoint_gen_1.json").exists()
    assert history[-1]["generation"] == 1


def test_resume_from_missing_checkpoint_fails_fast(tmp_path) -> None:
    config = _config()
    with pytest.raises(ValueError, match="corrupted NSGA-II checkpoint"):
        _run(
            config,
            checkpoint_config=CheckpointConfig(
                enabled=True,
                output_dir=tmp_path,
                run_id="nsga2_phase2b_resume_rejected",
                resume_from=tmp_path / "missing.json",
            ),
        )
