from __future__ import annotations

from ga_lab.api import (
    DemoInfo,
    PresetInfo,
    RecommendationResult,
    RunResultSummary,
    get_version,
    list_demos,
    list_presets,
    load_builtin_demo,
    load_builtin_preset,
    recommend_preset,
    recommend_solver,
    run_demo,
    run_preset,
)


def test_public_api_lists_packaged_resources() -> None:
    presets = list_presets()
    demos = list_demos()

    assert presets
    assert demos
    assert isinstance(presets[0], PresetInfo)
    assert isinstance(demos[0], DemoInfo)
    assert any(item.name == "onemax_small" for item in presets)
    assert any(item.name == "baseline" for item in demos)


def test_public_api_loaders_and_recommendations() -> None:
    preset = load_builtin_preset("onemax_small")
    demo = load_builtin_demo("baseline")
    preset_recommendation = recommend_preset("onemax", 32, "default")
    solver_recommendation = recommend_solver("onemax", 128, "default")

    assert preset["run_name"] == "onemax_small"
    assert "entries" in demo
    assert isinstance(preset_recommendation, RecommendationResult)
    assert isinstance(solver_recommendation, RecommendationResult)
    assert preset_recommendation.preset_name == "onemax_small"
    assert solver_recommendation.solver_name == "hill_climb"
    assert get_version()


def test_public_api_run_helpers(tmp_path) -> None:
    preset_result = run_preset("onemax_small", output_root=tmp_path / "preset")
    demo_result = run_demo("nsga2", output_root=tmp_path / "demo")

    assert isinstance(preset_result, RunResultSummary)
    assert isinstance(demo_result, RunResultSummary)
    assert preset_result.metrics["best_fitness"] == 32.0
    assert demo_result.metrics["hypervolume"] is not None
    assert preset_result.summary_path is not None
    assert demo_result.summary_path is not None
