from __future__ import annotations

from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.experiment.nsga2_candidate_suite import (
    build_problem_config,
    mo_candidate_suite_specs,
    safe_artifact_path,
)
from ga_lab.experiment.nsga2_candidate_variants import (
    apply_candidate_variant,
    candidate_d_uniform_crossover,
    candidate_variant_metadata,
)


def _base_config() -> GAConfig:
    return GAConfig(
        run_name="candidate_suite_test",
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
        seed=21,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"hypervolume_reference_point": [1.05, 10.5]},
    )


def test_candidate_d_metadata_and_config_are_separated_from_default() -> None:
    base = _base_config()
    variant = candidate_d_uniform_crossover()

    candidate_config = apply_candidate_variant(base, variant)
    metadata = candidate_variant_metadata(variant)

    assert base.crossover == "arithmetic"
    assert candidate_config.crossover == "uniform"
    assert candidate_config.run_name.endswith(variant.candidate_id)
    assert metadata["default_changed"] is False
    assert metadata["promotion_status"] == "under_validation"


def test_build_problem_config_retargets_cross_benchmark_problem() -> None:
    base = _base_config()
    spec = mo_candidate_suite_specs()["dtlz2"]

    config = build_problem_config(base, spec)

    assert config.problem == "dtlz2"
    assert config.problem_options == {"objective_count": 2}
    assert config.genome_length == spec.variables
    assert config.algorithm_options["hypervolume_reference_point"] == [1.1, 1.1]
    assert base.problem == "zdt1"


def test_safe_artifact_path_respects_suffix_and_existing_files(tmp_path: Path) -> None:
    first = safe_artifact_path(tmp_path, "nsga2_candidate_suite_validation_results", "zdt_suite", ".json")
    assert first.name == "nsga2_candidate_suite_validation_results_zdt_suite.json"

    first.write_text("{}", encoding="utf-8")
    second = safe_artifact_path(tmp_path, "nsga2_candidate_suite_validation_results", "zdt_suite", ".json")
    assert second.name.startswith("nsga2_candidate_suite_validation_results_zdt_suite_")
    assert second.suffix == ".json"
