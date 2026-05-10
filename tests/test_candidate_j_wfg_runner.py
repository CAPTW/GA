from __future__ import annotations

from ga_lab.experiment.nsga2_candidate_suite import mo_candidate_suite_specs, safe_artifact_path


def test_candidate_j_wfg_specs_are_registered() -> None:
    specs = mo_candidate_suite_specs()

    assert "wfg1" in specs
    assert "wfg2" in specs
    assert specs["wfg1"].bounds == (0.0, 1.0)
    assert specs["wfg2"].bounds == (0.0, 1.0)
    assert specs["wfg1"].reference_front_name.startswith("pymoo_wfg1")
    assert specs["wfg2"].reference_front_name.startswith("pymoo_wfg2")


def test_candidate_j_wfg_artifact_suffix_is_respected(tmp_path) -> None:
    result_path = safe_artifact_path(
        tmp_path,
        "nsga2_candidate_j_wfg_results",
        "wfg_smoke1",
        ".json",
    )

    assert result_path.name == "nsga2_candidate_j_wfg_results_wfg_smoke1.json"
