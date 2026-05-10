from __future__ import annotations

from ga_lab.config import GAConfig
from ga_lab.experiment.grid import build_grid_summary
from ga_lab.runner import run_experiment


def test_nsga2_runs_multiobjective(tmp_path) -> None:
    config = GAConfig(
        run_name="test_nsga2",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=30,
        genome_length=10,
        generations=15,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        tournament_size=3,
        seed=3,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.05},
        log_every=5,
    )
    result = run_experiment(config, output_root=tmp_path)
    summary = result.summary
    assert summary["is_nsga2"] is True
    assert summary["pareto_front_size"] >= 1
    assert isinstance(summary["best_fitness_vector"], list)
    assert len(summary["best_fitness_vector"]) == 2
    assert "mean_fitness" in summary
    assert "worst_fitness" in summary
    assert summary["hypervolume"] is not None
    assert summary["hypervolume"] > 0.0
    assert "spread" in summary
    assert 0.0 < summary["pareto_ratio"] <= 1.0
    assert "mean_convergence_speed" in summary
    assert "normalized_hypervolume" in result.history[0]
    assert "convergence_speed" in result.history[0]
    assert "pareto_ratio" in result.history[0]
    assert summary["hypervolume_reference_point_preset"] == [1.1, 11.0]
    assert summary["hypervolume_reference_point_override"] is None
    assert summary["hypervolume_reference_point_source"] == "problem_preset"


def test_nsga2_respects_configured_hypervolume_reference_point(tmp_path) -> None:
    config = GAConfig(
        run_name="test_nsga2_reference_point",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=20,
        genome_length=6,
        generations=8,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        tournament_size=2,
        seed=9,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.05},
        algorithm_options={"hypervolume_reference_point": [1.25, 12.5]},
        log_every=4,
    )
    result = run_experiment(config, output_root=tmp_path)

    assert result.summary["hypervolume_reference_point"] == [1.25, 12.5]
    assert result.summary["hypervolume_reference_point_preset"] == [1.1, 11.0]
    assert result.summary["hypervolume_reference_point_override"] == [1.25, 12.5]
    assert result.summary["hypervolume_reference_point_source"] == "config_override"


def test_run_grid_aggregates_multiobjective_progress(tmp_path) -> None:
    base_config = GAConfig(
        run_name="test_nsga2_grid",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=24,
        genome_length=8,
        generations=12,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        tournament_size=2,
        seed=4,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.05},
        log_every=4,
    )
    run_results = []
    for seed in (4, 5):
        config = GAConfig(**base_config.to_dict())
        config.seed = seed
        config.run_name = f"test_nsga2_grid_seed{seed}"
        run_results.append(run_experiment(config, output_root=tmp_path))

    aggregate = build_grid_summary(base_config, run_results, seeds=2)

    assert aggregate["mean_pareto_ratio"] > 0.0
    assert "multiobjective_progress" in aggregate
    assert aggregate["multiobjective_progress"]
    assert aggregate["hypervolume_reference_point"] == [1.1, 11.0]
    assert aggregate["hypervolume_reference_point_preset"] == [1.1, 11.0]
    assert aggregate["hypervolume_reference_point_override"] is None
    first_row = aggregate["multiobjective_progress"][0]
    assert "mean_pareto_ratio" in first_row
    assert "mean_convergence_speed" in first_row


def test_nsga2_uses_problem_default_objective_directions(tmp_path) -> None:
    config = GAConfig(
        run_name="test_nsga2_default_directions",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=18,
        genome_length=6,
        generations=6,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        tournament_size=2,
        seed=11,
        maximize=True,
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.05},
        log_every=3,
    )

    result = run_experiment(config, output_root=tmp_path)

    assert result.summary["objective_directions"] == [False, False]
