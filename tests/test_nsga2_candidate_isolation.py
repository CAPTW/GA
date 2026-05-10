from __future__ import annotations

from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.experiment.external_mo_comparators import run_internal_nsga2
from ga_lab.experiment.nsga2_candidate_variants import (
    apply_candidate_variant,
    candidate_d_uniform_crossover,
    candidate_h_uniform_dedup_mutation_boost,
    candidate_l_sparse_parent_bias_light,
    candidate_m_boundary_preservation_light,
    candidate_n_low_g_tail_mutation_light,
    candidate_o_spread_preserving_variation_light,
    candidate_variant_metadata,
)


def _base_config() -> GAConfig:
    return GAConfig(
        run_name="candidate_isolation_test",
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
        seed=41,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"hypervolume_reference_point": [1.05, 10.5]},
    )


def test_default_internal_nsga2_metadata_has_no_candidate_fields(tmp_path: Path) -> None:
    result = run_internal_nsga2(_base_config(), seed=41, output_root=str(tmp_path))

    assert result.success is True
    metadata = result.metadata
    assert "candidate_id" not in metadata
    assert "default_changed" not in metadata
    assert "promotion_status" not in metadata
    assert "base_candidate_id" not in metadata


def test_candidate_d_metadata_only_appears_when_explicitly_merged() -> None:
    base = _base_config()
    variant = candidate_d_uniform_crossover()
    default_result = run_internal_nsga2(base, seed=42, output_root="outputs/test_candidate_isolation")
    candidate_config = apply_candidate_variant(base, variant)
    metadata = dict(default_result.metadata)
    metadata.update(candidate_variant_metadata(variant))

    assert default_result.metadata.get("candidate_id") is None
    assert candidate_config.crossover == "uniform"
    assert metadata["candidate_id"] == "candidate_d_uniform_crossover"
    assert metadata["default_changed"] is False


def test_candidate_h_metadata_only_appears_when_explicitly_merged() -> None:
    base = _base_config()
    variant = candidate_h_uniform_dedup_mutation_boost()
    default_result = run_internal_nsga2(base, seed=43, output_root="outputs/test_candidate_isolation")
    candidate_config = apply_candidate_variant(base, variant)
    metadata = dict(default_result.metadata)
    metadata.update(candidate_variant_metadata(variant))

    assert default_result.metadata.get("candidate_id") is None
    assert candidate_config.algorithm_options["nsga2_duplicate_retry_count"] == 2
    assert candidate_config.algorithm_options["nsga2_duplicate_retry_mutation_scale"] == 2.0
    assert metadata["candidate_id"] == "candidate_h_uniform_dedup_mutation_boost"
    assert metadata["default_changed"] is False


def test_candidate_l_metadata_only_appears_when_explicitly_merged() -> None:
    base = _base_config()
    variant = candidate_l_sparse_parent_bias_light()
    default_result = run_internal_nsga2(base, seed=44, output_root="outputs/test_candidate_isolation")
    candidate_config = apply_candidate_variant(base, variant)
    metadata = dict(default_result.metadata)
    metadata.update(candidate_variant_metadata(variant))

    assert default_result.metadata.get("candidate_id") is None
    assert candidate_config.selection_options["nsga2_sparse_parent_bias_light"] is True
    assert metadata["candidate_id"] == "candidate_l_sparse_parent_bias_light"
    assert metadata["base_candidate_id"] == "candidate_j_h_lite_retry2"
    assert metadata["default_changed"] is False


def test_candidate_m_metadata_only_appears_when_explicitly_merged() -> None:
    base = _base_config()
    variant = candidate_m_boundary_preservation_light()
    default_result = run_internal_nsga2(base, seed=45, output_root="outputs/test_candidate_isolation")
    candidate_config = apply_candidate_variant(base, variant)
    metadata = dict(default_result.metadata)
    metadata.update(candidate_variant_metadata(variant))

    assert default_result.metadata.get("candidate_id") is None
    assert candidate_config.algorithm_options["nsga2_partial_front_strategy"] == "boundary_preservation_light"
    assert metadata["candidate_id"] == "candidate_m_boundary_preservation_light"
    assert metadata["base_candidate_id"] == "candidate_j_h_lite_retry2"
    assert metadata["mechanism"] == "boundary_preservation_light"
    assert metadata["default_changed"] is False


def test_candidate_n_metadata_only_appears_when_explicitly_merged() -> None:
    base = _base_config()
    variant = candidate_n_low_g_tail_mutation_light()
    default_result = run_internal_nsga2(base, seed=46, output_root="outputs/test_candidate_isolation")
    candidate_config = apply_candidate_variant(base, variant)
    metadata = dict(default_result.metadata)
    metadata.update(candidate_variant_metadata(variant))

    assert default_result.metadata.get("candidate_id") is None
    assert candidate_config.algorithm_options["nsga2_low_g_tail_mutation_light"] is True
    assert candidate_config.algorithm_options["nsga2_low_g_tail_zdt1_only"] is True
    assert metadata["candidate_id"] == "candidate_n_low_g_tail_mutation_light"
    assert metadata["base_candidate_id"] == "candidate_j_h_lite_retry2"
    assert metadata["mechanism"] == "low_g_tail_mutation_light"
    assert metadata["promotion_status"] == "phase0_sanity"
    assert metadata["default_changed"] is False


def test_candidate_o_metadata_only_appears_when_explicitly_merged() -> None:
    base = _base_config()
    variant = candidate_o_spread_preserving_variation_light()
    default_result = run_internal_nsga2(base, seed=47, output_root="outputs/test_candidate_isolation")
    candidate_config = apply_candidate_variant(base, variant)
    metadata = dict(default_result.metadata)
    metadata.update(candidate_variant_metadata(variant))

    assert default_result.metadata.get("candidate_id") is None
    assert candidate_config.algorithm_options["nsga2_low_g_tail_mutation_light"] is True
    assert candidate_config.algorithm_options["nsga2_spread_preserving_variation_light"] is True
    assert candidate_config.algorithm_options["nsga2_spread_preserving_zdt1_only"] is True
    assert metadata["candidate_id"] == "candidate_o_spread_preserving_variation_light"
    assert metadata["base_candidate_id"] == "candidate_n_low_g_tail_mutation_light"
    assert metadata["mechanism"] == "spread_preserving_variation_light"
    assert metadata["promotion_status"] == "phase0_sanity"
    assert metadata["default_changed"] is False


def test_applying_candidate_variant_does_not_mutate_default_config() -> None:
    base = _base_config()
    _ = apply_candidate_variant(base, candidate_d_uniform_crossover())
    _ = apply_candidate_variant(base, candidate_h_uniform_dedup_mutation_boost())
    _ = apply_candidate_variant(base, candidate_l_sparse_parent_bias_light())
    _ = apply_candidate_variant(base, candidate_m_boundary_preservation_light())
    _ = apply_candidate_variant(base, candidate_n_low_g_tail_mutation_light())
    _ = apply_candidate_variant(base, candidate_o_spread_preserving_variation_light())

    assert base.crossover == "arithmetic"
    assert base.algorithm_options == {"hypervolume_reference_point": [1.05, 10.5]}
    assert base.selection_options == {}
