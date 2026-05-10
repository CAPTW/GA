from __future__ import annotations

from scripts.recommend_solver import recommend


def test_onemax_default_prefers_hill_climb() -> None:
    recommendation = recommend("onemax", 128, "default")
    assert recommendation["solver_family"] == "baseline"
    assert recommendation["solver_name"] == "hill_climb"


def test_tsp_medium_quality_prefers_promoted_hybrid() -> None:
    recommendation = recommend("tsp", 20, "quality")
    assert recommendation["solver_family"] == "hybrid-ga"
    assert recommendation["config_path"] == "configs/presets/tsp_medium_hybrid.json"


def test_tsp_large_quality_stays_with_baseline() -> None:
    recommendation = recommend("tsp", 50, "quality")
    assert recommendation["solver_family"] == "baseline"
    assert recommendation["solver_name"] == "nearest_neighbor_2opt"


def test_zdt1_hv_uses_nsga2_preset() -> None:
    recommendation = recommend("zdt1", 50, "hv")
    assert recommendation["solver_family"] == "pure-ga"
    assert recommendation["config_path"] == "configs/presets/zdt1_large.json"
