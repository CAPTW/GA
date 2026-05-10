from __future__ import annotations

from pathlib import Path

from ga_lab.experiment.nsga2_candidate_suite import mo_candidate_suite_specs, safe_artifact_path
from ga_lab.experiment.nsga2_candidate_variants import candidate_j_h_lite_retry2, candidate_variant_metadata


def test_candidate_j_metadata_stays_opt_in() -> None:
    metadata = candidate_variant_metadata(candidate_j_h_lite_retry2())

    assert metadata["candidate_id"] == "candidate_j_h_lite_retry2"
    assert metadata["default_changed"] is False
    assert metadata["promotion_status"] == "under_validation"
    assert metadata["base_candidate_id"] == "candidate_h_uniform_dedup_mutation_boost"


def test_candidate_j_extended_runner_can_target_dtlz4_suite_member() -> None:
    specs = mo_candidate_suite_specs()

    assert "dtlz4" in specs
    assert specs["dtlz4"].reference_front_name == "analytic_dtlz4_m2"
    assert specs["dtlz4"].hv_reference_point == (1.1, 1.1)


def test_candidate_j_extended_runner_artifact_suffix_is_respected(tmp_path: Path) -> None:
    result_path = safe_artifact_path(
        tmp_path,
        "nsga2_candidate_j_extended_results",
        "candidate_j_ext1",
        ".json",
    )

    assert result_path.name == "nsga2_candidate_j_extended_results_candidate_j_ext1.json"
