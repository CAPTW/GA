from __future__ import annotations

from ga_lab.config import GAConfig
from ga_lab.runner import run_experiment


def test_ga_reaches_target_fitness_on_onemax(tmp_path) -> None:
    config = GAConfig(
        run_name="test_onemax",
        problem="onemax",
        population_size=60,
        genome_length=32,
        generations=60,
        crossover_rate=0.9,
        mutation_rate=0.02,
        elitism=2,
        tournament_size=3,
        seed=13,
        maximize=True,
        target_fitness=32,
        log_every=1,
    )
    result = run_experiment(config, output_root=tmp_path)
    assert result.summary["best_fitness"] == 32.0
    assert "mean_fitness" in result.summary
    assert "worst_fitness" in result.summary
    assert result.summary["stop_reason"] == "target_fitness_reached"
    assert result.output_dir.exists()


def test_history_contains_generation_zero(tmp_path) -> None:
    config = GAConfig(
        run_name="test_history",
        problem="onemax",
        population_size=20,
        genome_length=16,
        generations=5,
        crossover_rate=0.8,
        mutation_rate=0.05,
        elitism=1,
        tournament_size=3,
        seed=2,
        maximize=True,
        target_fitness=None,
        log_every=1,
    )
    result = run_experiment(config, output_root=tmp_path)
    assert result.history[0]["generation"] == 0
    assert result.history[-1]["generation"] <= 5


def _normalize_summary(summary: dict[str, object]) -> dict[str, object]:
    normalized = dict(summary)
    normalized.pop("runtime_seconds", None)
    return normalized


def _run_onemax_with_seed(seed: int, *, output_root) -> object:
    return run_experiment(
        GAConfig(
            run_name="seed_repro",
            problem="onemax",
            population_size=40,
            genome_length=16,
            generations=6,
            crossover_rate=0.9,
            mutation_rate=0.02,
            elitism=1,
            tournament_size=3,
            seed=seed,
            maximize=True,
            log_every=1,
        ),
        output_root=output_root,
    )


def test_seed_reproducibility(tmp_path) -> None:
    result_a = _run_onemax_with_seed(19, output_root=tmp_path / "seed_a")
    result_b = _run_onemax_with_seed(19, output_root=tmp_path / "seed_b")

    assert _normalize_summary(result_a.summary) == _normalize_summary(result_b.summary)
    assert result_a.history == result_b.history

    result_c = _run_onemax_with_seed(3, output_root=tmp_path / "seed_c")
    assert _normalize_summary(result_a.summary) != _normalize_summary(result_c.summary) or result_a.history != result_c.history
