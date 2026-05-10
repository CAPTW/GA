from __future__ import annotations

from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.experiment.nsga2_candidate_suite import mo_candidate_suite_specs, safe_artifact_path
from ga_lab.experiment.nsga2_candidate_variants import (
    apply_candidate_variant,
    candidate_d_uniform_crossover,
    candidate_e_uniform_decision_dedup,
    candidate_f_uniform_objective_dedup,
    candidate_g_uniform_crowding_survival,
    candidate_h_uniform_dedup_mutation_boost,
    candidate_variant_metadata,
)


def _base_config() -> GAConfig:
    return GAConfig(
        run_name="diversity_candidate_test",
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
        seed=31,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"hypervolume_reference_point": [1.05, 10.5]},
    )


def test_diversity_candidate_metadata_stays_opt_in_only() -> None:
    variants = [
        candidate_d_uniform_crossover(),
        candidate_e_uniform_decision_dedup(),
        candidate_f_uniform_objective_dedup(),
        candidate_g_uniform_crowding_survival(),
        candidate_h_uniform_dedup_mutation_boost(),
    ]
    variant_ids = {variant.candidate_id for variant in variants}

    assert len(variant_ids) == len(variants)
    for variant in variants:
        metadata = candidate_variant_metadata(variant)
        assert metadata["default_changed"] is False
        assert metadata["promotion_status"] == "under_validation"


def test_follow_up_candidates_do_not_mutate_the_default_config() -> None:
    base = _base_config()

    decision_dedup = apply_candidate_variant(base, candidate_e_uniform_decision_dedup())
    objective_dedup = apply_candidate_variant(base, candidate_f_uniform_objective_dedup())
    crowding = apply_candidate_variant(base, candidate_g_uniform_crowding_survival())
    boosted = apply_candidate_variant(base, candidate_h_uniform_dedup_mutation_boost())

    assert base.crossover == "arithmetic"
    assert decision_dedup.crossover == "uniform"
    assert objective_dedup.algorithm_options["nsga2_partial_front_dedup_mode"] == "objective"
    assert crowding.algorithm_options["nsga2_partial_front_strategy"] == "novelty_crowding"
    assert boosted.algorithm_options["nsga2_duplicate_retry_mutation_scale"] == 2.0


def test_candidate_suite_specs_include_dtlz3_smoke() -> None:
    spec = mo_candidate_suite_specs()["dtlz3"]

    assert spec.problem == "dtlz3"
    assert spec.objectives == 2
    assert spec.problem_options == {"objective_count": 2}


def test_diversity_runner_artifact_suffix_is_respected(tmp_path: Path) -> None:
    result_path = safe_artifact_path(
        tmp_path,
        "nsga2_diversity_candidate_results",
        "diversity_run1",
        ".json",
    )

    assert result_path.name == "nsga2_diversity_candidate_results_diversity_run1.json"
