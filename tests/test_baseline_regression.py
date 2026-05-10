from __future__ import annotations

from pathlib import Path

import pytest

from ga_lab.config import load_config
from ga_lab.runner import run_experiment


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_onemax_baseline_regression(tmp_path) -> None:
    config = load_config(_project_root() / "configs" / "onemax_baseline.json")
    result = run_experiment(config, output_root=tmp_path)

    assert result.summary["best_fitness"] == 64.0
    assert result.summary["stop_reason"] == "target_fitness_reached"
    assert result.summary["final_generation"] == 29
    assert result.summary["mean_fitness"] == pytest.approx(60.3375)
    assert result.summary["worst_fitness"] == 57.0


def test_tsp_baseline_regression(tmp_path) -> None:
    config = load_config(_project_root() / "configs" / "tsp_baseline.json")
    result = run_experiment(config, output_root=tmp_path)

    assert result.summary["best_fitness"] == pytest.approx(-446.8247864893467)
    assert result.summary["stop_reason"] == "max_generations"
    assert result.summary["final_generation"] == 120
    assert result.summary["mean_fitness"] == pytest.approx(-627.4727751254502)
    assert result.summary["best_route_distance"] == pytest.approx(446.8247864893467)


def test_knapsack_baseline_regression(tmp_path) -> None:
    config = load_config(_project_root() / "configs" / "knapsack_baseline.json")
    result = run_experiment(config, output_root=tmp_path)

    assert result.summary["best_fitness"] == pytest.approx(345.71668519093726)
    assert result.summary["stop_reason"] == "max_generations"
    assert result.summary["final_generation"] == 120
    assert result.summary["mean_fitness"] == pytest.approx(194.41588908681132)
    assert result.summary["best_is_feasible"] is True
    assert result.summary["best_constraint_violation"] == 0.0


def test_zdt1_baseline_regression(tmp_path) -> None:
    config = load_config(_project_root() / "configs" / "zdt1_nsga2.json")
    result = run_experiment(config, output_root=tmp_path)

    assert result.summary["best_fitness"] == 0.0
    assert result.summary["stop_reason"] == "max_generations"
    assert result.summary["final_generation"] == 120
    assert result.summary["mean_fitness"] == pytest.approx(0.08766697965601224)
    assert result.summary["pareto_front_size"] == 27
    assert result.summary["hypervolume"] == pytest.approx(10.880970316338033)
    assert result.summary["spread"] == pytest.approx(0.5478408425503725)
