from __future__ import annotations

from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.experiment.nsga2_candidate_suite import safe_artifact_path
from ga_lab.experiment.nsga2_candidate_variants import (
    apply_candidate_variant,
    candidate_h_uniform_dedup_mutation_boost,
    candidate_i_h_lite_retry1,
    candidate_j_h_lite_retry2,
    candidate_k_dedup_only_lite,
    candidate_variant_metadata,
)


def _base_config() -> GAConfig:
    return GAConfig(
        run_name="h_lite_candidate_test",
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


def test_h_lite_metadata_stays_opt_in_and_unique() -> None:
    variants = [
        candidate_h_uniform_dedup_mutation_boost(),
        candidate_i_h_lite_retry1(),
        candidate_j_h_lite_retry2(),
        candidate_k_dedup_only_lite(),
    ]
    ids = {variant.candidate_id for variant in variants}
    assert len(ids) == len(variants)
    for variant in variants[1:]:
        metadata = candidate_variant_metadata(variant)
        assert metadata["default_changed"] is False
        assert metadata["promotion_status"] == "under_validation"


def test_h_lite_candidates_do_not_change_default_nsga2_path() -> None:
    base = _base_config()
    i_cfg = apply_candidate_variant(base, candidate_i_h_lite_retry1())
    j_cfg = apply_candidate_variant(base, candidate_j_h_lite_retry2())
    k_cfg = apply_candidate_variant(base, candidate_k_dedup_only_lite())

    assert base.crossover == "arithmetic"
    assert base.algorithm_options == {"hypervolume_reference_point": [1.05, 10.5]}
    assert i_cfg.algorithm_options["nsga2_duplicate_retry_count"] == 1
    assert i_cfg.algorithm_options["nsga2_duplicate_retry_mutation_scale"] == 1.25
    assert j_cfg.algorithm_options["nsga2_duplicate_retry_count"] == 2
    assert j_cfg.algorithm_options["nsga2_duplicate_retry_mutation_scale"] == 1.5
    assert k_cfg.algorithm_options["nsga2_duplicate_reinitialize_fallback"] is True
    assert "nsga2_duplicate_retry_count" not in k_cfg.algorithm_options


def test_h_lite_runner_artifact_suffix_is_respected(tmp_path: Path) -> None:
    result_path = safe_artifact_path(
        tmp_path,
        "nsga2_h_lite_candidate_results",
        "h_lite_run1",
        ".json",
    )
    assert result_path.name == "nsga2_h_lite_candidate_results_h_lite_run1.json"
