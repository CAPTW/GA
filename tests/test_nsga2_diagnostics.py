from __future__ import annotations

import json
from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.experiment.external_mo_comparators import run_internal_nsga2
from ga_lab.experiment.nsga2_diagnostics import (
    Nsga2DiagnosticsConfig,
    compute_zdt1_components,
    summarize_boundary_intervention_count,
    summarize_boundary_retention,
    summarize_duplicate_to_diversity_funnel,
    summarize_front_change,
    summarize_internal_external_distribution_comparison,
    summarize_mutation_retry_objective_effect,
    summarize_parent_contributions,
    summarize_segment0_spacing_detail,
    summarize_survivor_set_diff,
)
from ga_lab.problems.zdt1 import ZDT1Problem


def _base_config() -> GAConfig:
    return GAConfig(
        run_name="nsga2_diagnostics_test",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=20,
        genome_length=6,
        generations=12,
        crossover_rate=0.9,
        mutation_rate=0.15,
        elitism=1,
        tournament_size=2,
        seed=51,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"hypervolume_reference_point": [1.05, 10.5]},
    )


def test_diagnostics_config_defaults_to_disabled() -> None:
    config = Nsga2DiagnosticsConfig.from_algorithm_options({}, default_run_id="diag_default")

    assert config.trace_enabled is False
    assert config.run_id == "diag_default"
    assert config.deep.deep_trace_enabled is False
    assert config.lineage.lineage_trace_enabled is False
    assert config.operator_supply.operator_supply_trace_enabled is False
    assert config.zdt1_component.zdt1_component_trace_enabled is False
    assert config.operator_supply.segment_count == 6
    assert config.deep.generation_sample_stride == 1


def test_trace_disabled_does_not_change_default_internal_result(tmp_path: Path) -> None:
    base = _base_config()
    default_result = run_internal_nsga2(base, seed=51, output_root=str(tmp_path / "default"))

    disabled_config = GAConfig.from_dict(base.to_dict())
    disabled_config.algorithm_options = dict(disabled_config.algorithm_options)
    disabled_config.algorithm_options["nsga2_trace_enabled"] = False
    disabled_config.algorithm_options["nsga2_deep_trace_enabled"] = False
    disabled_result = run_internal_nsga2(
        disabled_config,
        seed=51,
        output_root=str(tmp_path / "disabled"),
    )

    assert default_result.success is True
    assert disabled_result.success is True
    assert default_result.objective_vectors == disabled_result.objective_vectors
    assert "nsga2_diagnostics" not in disabled_result.metadata
    assert "diagnostics_enabled" not in disabled_result.metadata


def test_basic_trace_enabled_keeps_deep_traces_disabled(tmp_path: Path) -> None:
    traced_config = _base_config()
    traced_config.algorithm_options = dict(traced_config.algorithm_options)
    traced_config.algorithm_options["nsga2_trace_enabled"] = True
    traced_config.algorithm_options["nsga2_trace_run_id"] = "diag_basic_test"
    traced_config.algorithm_options["nsga2_deep_trace_enabled"] = False

    result = run_internal_nsga2(traced_config, seed=52, output_root=str(tmp_path / "basic"))

    assert result.success is True
    payload = result.metadata["nsga2_diagnostics"]
    trace_types = set(payload["aggregate"]["trace_types"])
    assert payload["trace_config"]["deep_trace_enabled"] is False
    assert payload["trace_config"]["lineage_trace_enabled"] is False
    assert "parent_contribution_trace" in trace_types
    assert "parent_to_offspring_trace" not in trace_types
    assert "offspring_to_survivor_trace" not in trace_types
    assert "lineage_retention_funnel" not in trace_types


def test_deep_trace_enabled_adds_lineage_and_segment_traces(tmp_path: Path) -> None:
    traced_config = _base_config()
    traced_config.algorithm_options = dict(traced_config.algorithm_options)
    traced_config.algorithm_options["nsga2_trace_enabled"] = True
    traced_config.algorithm_options["nsga2_trace_run_id"] = "diag_deep_test"
    traced_config.algorithm_options["nsga2_deep_trace_enabled"] = True
    traced_config.algorithm_options["nsga2_trace_generation_sample_stride"] = 1

    result = run_internal_nsga2(traced_config, seed=53, output_root=str(tmp_path / "deep"))

    assert result.success is True
    assert result.metadata["diagnostics_enabled"] is True
    payload = result.metadata["nsga2_diagnostics"]
    trace_types = set(payload["aggregate"]["trace_types"])
    assert payload["trace_enabled"] is True
    assert payload["trace_config"]["deep_trace_enabled"] is True
    assert payload["trace_config"]["lineage_trace_enabled"] is False
    assert "parent_to_offspring_trace" in trace_types
    assert "offspring_to_survivor_trace" in trace_types
    assert "objective_boundary_retention_detail" in trace_types
    assert "segment_spacing_attribution" in trace_types
    assert "crowding_decision_attribution" in trace_types
    assert len(json.dumps(payload)) < 1_500_000


def test_lineage_trace_enabled_adds_lineage_traces(tmp_path: Path) -> None:
    traced_config = _base_config()
    traced_config.algorithm_options = dict(traced_config.algorithm_options)
    traced_config.algorithm_options["nsga2_trace_enabled"] = True
    traced_config.algorithm_options["nsga2_trace_run_id"] = "diag_lineage_test"
    traced_config.algorithm_options["nsga2_deep_trace_enabled"] = False
    traced_config.algorithm_options["nsga2_lineage_trace_enabled"] = True
    traced_config.algorithm_options["nsga2_trace_generation_sample_stride"] = 1

    result = run_internal_nsga2(traced_config, seed=54, output_root=str(tmp_path / "lineage"))

    assert result.success is True
    payload = result.metadata["nsga2_diagnostics"]
    trace_types = set(payload["aggregate"]["trace_types"])
    assert payload["trace_config"]["deep_trace_enabled"] is False
    assert payload["trace_config"]["lineage_trace_enabled"] is True
    assert "lineage_retention_funnel" in trace_types
    assert "sparse_lineage_quality" in trace_types
    assert "segment0_spacing_detail" in trace_types
    assert "duplicate_to_diversity_funnel" in trace_types
    assert "boundary_intervention_count" in trace_types
    assert len(json.dumps(payload)) < 1_500_000


def test_operator_supply_trace_enabled_adds_supply_traces(tmp_path: Path) -> None:
    traced_config = _base_config()
    traced_config.algorithm_options = dict(traced_config.algorithm_options)
    traced_config.algorithm_options["nsga2_trace_enabled"] = True
    traced_config.algorithm_options["nsga2_trace_run_id"] = "diag_operator_supply_test"
    traced_config.algorithm_options["nsga2_operator_supply_trace_enabled"] = True
    traced_config.algorithm_options["nsga2_trace_segment_count"] = 6
    traced_config.algorithm_options["nsga2_trace_generation_sample_stride"] = 1

    result = run_internal_nsga2(
        traced_config,
        seed=55,
        output_root=str(tmp_path / "operator_supply"),
    )

    assert result.success is True
    payload = result.metadata["nsga2_diagnostics"]
    trace_types = set(payload["aggregate"]["trace_types"])
    assert payload["trace_config"]["operator_supply_trace_enabled"] is True
    assert payload["trace_config"]["segment_count"] == 6
    assert "initialization_segment_coverage" in trace_types
    assert "variation_segment_transition" in trace_types
    assert "operator_offspring_quality" in trace_types
    assert "mutation_retry_objective_effect" in trace_types
    assert "segment0_supply_funnel" in trace_types
    assert "lineage_retention_funnel" not in trace_types
    assert "parent_to_offspring_trace" not in trace_types
    assert len(json.dumps(payload)) < 1_500_000


def test_zdt1_component_trace_enabled_adds_component_traces(tmp_path: Path) -> None:
    traced_config = _base_config()
    traced_config.algorithm_options = dict(traced_config.algorithm_options)
    traced_config.algorithm_options["nsga2_trace_enabled"] = True
    traced_config.algorithm_options["nsga2_trace_run_id"] = "diag_zdt1_component_test"
    traced_config.algorithm_options["nsga2_zdt1_component_trace_enabled"] = True
    traced_config.algorithm_options["nsga2_trace_segment_count"] = 6
    traced_config.algorithm_options["nsga2_trace_generation_sample_stride"] = 1

    result = run_internal_nsga2(
        traced_config,
        seed=56,
        output_root=str(tmp_path / "zdt1_components"),
    )

    assert result.success is True
    payload = result.metadata["nsga2_diagnostics"]
    trace_types = set(payload["aggregate"]["trace_types"])
    assert payload["trace_config"]["zdt1_component_trace_enabled"] is True
    assert "zdt1_initial_component_coverage" in trace_types
    assert "zdt1_offspring_component_quality" in trace_types
    assert "zdt1_parent_child_component_delta" in trace_types
    assert "zdt1_mutation_retry_component_effect" in trace_types
    assert "zdt1_segment0_quality_funnel" in trace_types
    assert "lineage_retention_funnel" not in trace_types
    assert len(json.dumps(payload)) < 1_500_000


def test_parent_contribution_summary_is_json_serializable() -> None:
    summary = summarize_parent_contributions(
        [
            {
                "winner_index": 1,
                "winner_rank": 0,
                "winner_crowding": 1.25,
                "is_boundary": True,
                "is_sparse": True,
                "sample_same_rank": False,
                "sample_crowding_tie": False,
                "bias_applied": False,
                "selection_kind": "crowded_tournament",
            },
            {
                "winner_index": 2,
                "winner_rank": 0,
                "winner_crowding": float("inf"),
                "is_boundary": True,
                "is_sparse": False,
                "sample_same_rank": True,
                "sample_crowding_tie": True,
                "bias_applied": True,
                "selection_kind": "sparse_parent_bias_light",
            },
        ],
        population_size=10,
        top_parent_limit=3,
    )

    assert summary["selection_event_count"] == 2
    assert summary["unique_parent_count"] == 2
    json.dumps(summary)


def test_boundary_retention_and_singleton_spacing_are_safe() -> None:
    empty = summarize_boundary_retention([], [], [False, False])
    singleton_change = summarize_front_change(
        [[0.1, 0.9]],
        [[0.1, 0.9]],
        [False, False],
    )

    assert empty["previous_boundary_count"] == 0
    assert empty["boundary_retention_rate"] is None
    assert singleton_change["previous_spacing"] is None
    assert singleton_change["current_spacing"] is None
    assert "previous_front_spacing_undefined" in singleton_change["warnings"]
    assert "current_front_spacing_undefined" in singleton_change["warnings"]


def test_survivor_set_diff_is_json_serializable() -> None:
    diff = summarize_survivor_set_diff(
        [[0.1, 0.9], [0.2, 0.8]],
        [[0.1, 0.9], [0.3, 0.7]],
        directions=[False, False],
        bins=4,
    )

    assert diff["candidate_front_size"] == 2
    assert diff["reference_front_size"] == 2
    assert diff["unique_to_candidate_count"] >= 0
    json.dumps(diff)


def test_segment0_duplicate_and_boundary_lineage_helpers_are_safe() -> None:
    segment0 = summarize_segment0_spacing_detail([[0.1, 0.9]], [False, False], bins=4)
    duplicate_funnel = summarize_duplicate_to_diversity_funnel(
        current_population=[[0.1, 0.2], [0.1, 0.2]],
        current_objective_vectors=[[0.1, 0.9], [0.1, 0.9]],
        next_population=[[0.1, 0.2], [0.2, 0.3]],
        next_objective_vectors=[[0.1, 0.9], [0.2, 0.8]],
        next_front_vectors=[[0.1, 0.9], [0.2, 0.8]],
        lineage_records=[],
        directions=[False, False],
        bins=4,
    )
    boundary = summarize_boundary_intervention_count(
        population_size=2,
        combined_fronts=[[0, 1, 2]],
        combined_crowding=[float("inf"), 1.0, 0.5],
        combined_population=[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
        combined_objective_vectors=[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        partial_front_strategy="boundary_preservation_light",
        partial_front_dedup_mode="none",
    )

    assert segment0["boundary_adjacent"] is True
    assert "segment_spacing_singleton_or_empty_front" in segment0["warnings"]
    assert duplicate_funnel["replacement_candidate_count"] == 0
    assert duplicate_funnel["unique_objective_count"] == 2
    assert boundary["boundary_preference_trigger_count"] == 1
    json.dumps(segment0)
    json.dumps(duplicate_funnel)
    json.dumps(boundary)


def test_mutation_retry_and_internal_external_supply_helpers_are_safe() -> None:
    retry_summary = summarize_mutation_retry_objective_effect(
        [
            {
                "duplicate_detected": False,
                "retry_attempt_count": 0,
                "retry_success": False,
                "retry_reinitialized": False,
                "decision_changed_after_retry": False,
                "offspring_decision_hash": "a",
                "offspring_objective": [0.1, 0.9],
                "offspring_survived_next_generation": True,
            }
        ],
        [False, False],
        bins=4,
    )
    diff_summary = summarize_internal_external_distribution_comparison(
        [[0.1, 0.9], [0.2, 0.8]],
        [[0.1, 0.9], [0.25, 0.75]],
        [False, False],
        bins=4,
    )

    assert retry_summary["retry_count"] == 0
    assert "objective_changed_after_retry_requires_extra_evaluation" in retry_summary["warnings"]
    assert diff_summary["candidate_segment0_count"] >= 0
    assert diff_summary["reference_segment0_count"] >= 0
    json.dumps(retry_summary)
    json.dumps(diff_summary)


def test_compute_zdt1_components_matches_problem_evaluator() -> None:
    genome = [0.2, 0.1, 0.2, 0.0, 0.3, 0.4]
    problem = ZDT1Problem("zdt1")
    objective = problem.fitness(genome)

    components = compute_zdt1_components(genome, objective, bins=6)

    assert components["f1"] == objective[0]
    assert components["f2"] == objective[1]
    assert components["g"] == 1.0 + (9.0 * sum(genome[1:]) / len(genome[1:]))
    assert components["segment_id"] == 1
    assert components["segment0_flag"] is False
    assert components["distance_to_zdt1_front"] is not None


def test_compute_zdt1_components_is_safe_for_other_dimensions() -> None:
    short = compute_zdt1_components([0.05, 0.0], [0.05, 0.7763932022500211], bins=10)
    long = compute_zdt1_components(
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0.05, 2.7297038216560505],
        bins=10,
    )

    assert short["tail_mean"] == 0.0
    assert short["segment0_flag"] is True
    assert long["tail_mean"] == 0.3
    assert long["g"] == 3.6999999999999997


def test_zdt1_component_trace_skips_gracefully_for_non_zdt1_problem(tmp_path: Path) -> None:
    traced_config = _base_config()
    traced_config.problem = "zdt2"
    traced_config.algorithm_options = dict(traced_config.algorithm_options)
    traced_config.algorithm_options["nsga2_trace_enabled"] = True
    traced_config.algorithm_options["nsga2_trace_run_id"] = "diag_zdt2_component_skip"
    traced_config.algorithm_options["nsga2_zdt1_component_trace_enabled"] = True

    result = run_internal_nsga2(
        traced_config,
        seed=57,
        output_root=str(tmp_path / "zdt2_components"),
    )

    assert result.success is True
    payload = result.metadata["nsga2_diagnostics"]
    trace_types = set(payload["aggregate"]["trace_types"])
    assert "zdt1_initial_component_coverage" not in trace_types
    assert "zdt1_component_trace_skipped_non_zdt1_problem" in payload["warnings"]
