from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from ga_lab.local_failure_trace import build_failure_hypothesis_registry
from ga_lab.local_experiments import load_local_study
from ga_lab.local_stress_refresh import build_stress_refresh_registry
from ga_lab.local_stress_target_reduction import build_stress_target_reduction_registry


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_json_command(*args: str, cwd: Path | None = None) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=cwd or _project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _write_study_manifest(tmp_path: Path, file_name: str, payload: dict[str, object]) -> Path:
    manifest_path = tmp_path / file_name
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _write_json_file(tmp_path: Path, file_name: str, payload: dict[str, object]) -> Path:
    output_path = tmp_path / file_name
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str] | tuple[str, ...]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def test_builtin_local_studies_load() -> None:
    studies_dir = _project_root() / "configs" / "local_studies"
    manifest_names = sorted(path.stem for path in studies_dir.glob("*.json"))

    assert manifest_names
    for study_name in manifest_names:
        study = load_local_study(study_name)
        assert study.study_name
        assert study.problem in {"onemax", "knapsack", "tsp", "zdt1"}
        assert study.primary_metric
        assert 1 <= len(study.sweep) <= 3
        assert study.seeds


def test_variant_override_study_loads() -> None:
    study = load_local_study("tsp_fixed_stack_coarse")

    assert "fixed_swap_seed50" in study.variant_overrides
    assert study.variant_overrides["fixed_swap_seed50"]["algorithm"] == "hybrid_ga"
    assert "study_variant" in study.sweep
    assert "fixed_inversion_seed50" in study.sweep["study_variant"]


def test_knapsack_family_and_anatomy_studies_load() -> None:
    family_study = load_local_study("knapsack_family_suite")
    anatomy_study = load_local_study("knapsack_seeded_repair_anatomy_study")
    control_study = load_local_study("knapsack_feasibility_control_study")
    confirm_study = load_local_study("knapsack_canonical_experimental_confirm")

    assert family_study.problem == "knapsack"
    assert family_study.cases
    assert "greedy_local_search" in family_study.variant_overrides
    assert family_study.variant_overrides["greedy_local_search"]["__local_baseline__"] == (
        "knapsack_greedy_local_search"
    )
    assert "seeded_repair" in anatomy_study.variant_overrides
    assert anatomy_study.variant_overrides["repair_only"]["algorithm"] == "hybrid_ga"
    assert control_study.variant_overrides["feasibility_aware_mutation_v1"]["algorithm_options"][
        "adaptive_policy"
    ] == "feasibility_aware_mutation_v1"
    assert confirm_study.variant_overrides["repair_only"]["algorithm_options"][
        "repair_strategy"
    ] == "knapsack_greedy_fill"


def test_two_stage_gate_studies_load() -> None:
    tsp_study = load_local_study("tsp_two_stage_gate_study")
    tsp_confirm = load_local_study("tsp_two_stage_gate_confirm")
    zdt1_study = load_local_study("zdt1_two_stage_gate_study")
    zdt1_confirm = load_local_study("zdt1_two_stage_gate_confirm")
    knapsack_sanity = load_local_study("knapsack_rerun_gate_sanity_study")
    onemax_gate = load_local_study("onemax_control_gate_check")

    assert tsp_study.problem == "tsp"
    assert tsp_study.variant_overrides["pilot_then_canonical_v1"]["__local_baseline__"] == (
        "tsp_two_stage_gate"
    )
    assert "always_fast" in tsp_confirm.sweep["study_variant"]
    assert zdt1_study.problem == "zdt1"
    assert zdt1_study.variant_overrides["pilot_then_canonical_v2"]["__local_baseline__"] == (
        "zdt1_two_stage_gate"
    )
    assert zdt1_confirm.seeds
    assert knapsack_sanity.problem == "knapsack"
    assert knapsack_sanity.variant_overrides["repair_rerun_gate"]["__local_baseline__"] == (
        "knapsack_repair_rerun_gate"
    )
    assert onemax_gate.problem == "onemax"
    assert "pilot_stop_variant" in onemax_gate.sweep["study_variant"]


def test_restart_portfolio_studies_load() -> None:
    tsp_study = load_local_study("tsp_restart_portfolio_study")
    tsp_confirm = load_local_study("tsp_restart_portfolio_confirm")
    zdt1_study = load_local_study("zdt1_restart_portfolio_study")
    zdt1_confirm = load_local_study("zdt1_restart_portfolio_confirm")
    knapsack_study = load_local_study("knapsack_multistart_sanity_study")
    onemax_study = load_local_study("onemax_control_portfolio_check")

    assert tsp_study.problem == "tsp"
    assert tsp_study.variant_overrides["fast_x3_equal_split"]["__local_baseline__"] == (
        "tsp_restart_portfolio"
    )
    assert "fast_x3_budget075" in tsp_confirm.sweep["study_variant"]
    assert zdt1_study.problem == "zdt1"
    assert zdt1_confirm.variant_overrides["fast_x2_equal_split"]["__local_baseline__"] == (
        "zdt1_restart_portfolio"
    )
    assert "fast_x2_budget075" in zdt1_confirm.sweep["study_variant"]
    assert knapsack_study.problem == "knapsack"
    assert knapsack_study.variant_overrides["none_x2_equal_split"]["__local_baseline__"] == (
        "knapsack_restart_portfolio"
    )
    assert onemax_study.problem == "onemax"
    assert "early_stop_plateau" in onemax_study.sweep["study_variant"]


def test_ranking_fidelity_studies_load() -> None:
    tsp_study = load_local_study("tsp_ranking_fidelity_study")
    tsp_confirm = load_local_study("tsp_triage_confirm")
    zdt1_study = load_local_study("zdt1_ranking_fidelity_study")
    zdt1_confirm = load_local_study("zdt1_triage_confirm")
    knapsack_study = load_local_study("knapsack_triage_sanity_study")
    onemax_study = load_local_study("onemax_control_ranking_check")

    assert tsp_study.problem == "tsp"
    assert "seeded_swap_seed50__canonical" in tsp_study.variant_overrides
    assert "seeded_inversion_seed50__fast" in tsp_confirm.sweep["study_variant"]
    assert zdt1_study.problem == "zdt1"
    assert zdt1_study.variant_overrides["threshold_055_refresh010__fast"]["population_size"] == 45
    assert "threshold_055_refresh020__canonical" in zdt1_confirm.sweep["study_variant"]
    assert knapsack_study.problem == "knapsack"
    assert knapsack_study.variant_overrides["repair_rerun_gate"]["__local_baseline__"] == (
        "knapsack_repair_rerun_gate"
    )
    assert onemax_study.problem == "onemax"
    assert "fast_budget_variant" in onemax_study.sweep["study_variant"]


def test_qf_tolerance_studies_load() -> None:
    tsp_study = load_local_study("tsp_qf_tolerance_study")
    tsp_confirm = load_local_study("tsp_qf_tolerance_confirm")
    zdt1_study = load_local_study("zdt1_qf_tolerance_study")
    zdt1_confirm = load_local_study("zdt1_qf_tolerance_confirm")
    knapsack_freeze = load_local_study("knapsack_note_freeze_check")
    onemax_freeze = load_local_study("onemax_control_freeze_check")

    assert tsp_study.problem == "tsp"
    assert tsp_study.analysis["qf_tolerance"]["quality_variant"] == "quality_first"
    assert tsp_confirm.analysis["qf_tolerance"]["fast_variant"] == "budget_first"
    assert zdt1_study.problem == "zdt1"
    assert zdt1_study.analysis["qf_tolerance"]["pareto_ratio_drop_threshold"] == 0.01
    assert zdt1_confirm.analysis["qf_tolerance"]["spread_degradation_threshold"] == 0.05
    assert knapsack_freeze.problem == "knapsack"
    assert "repair_only" in knapsack_freeze.variant_overrides
    assert onemax_freeze.problem == "onemax"
    assert "early_stop_reference" in onemax_freeze.sweep["study_variant"]


def test_tsp_fast_tail_hardening_studies_load() -> None:
    coarse = load_local_study("tsp_fast_tail_hardening_study")
    confirm = load_local_study("tsp_fast_tail_confirm")
    zdt1_freeze = load_local_study("zdt1_qf_tiny_freeze_check")

    assert coarse.problem == "tsp"
    assert coarse.analysis["tsp_fast_tail"]["quality_variant"] == "quality_first"
    assert coarse.analysis["tsp_fast_tail"]["baseline_fast_variant"] == "current_fast"
    assert "fast_inversion" in coarse.variant_overrides
    assert "current_fast" in confirm.sweep["study_variant"]
    assert zdt1_freeze.problem == "zdt1"
    assert zdt1_freeze.analysis["qf_tolerance"]["fast_variant"] == "budget_first"


def test_qf_recalibration_studies_load() -> None:
    tsp_study = load_local_study("tsp_qf_recalibration_study")
    tsp_confirm = load_local_study("tsp_qf_recalibration_confirm")
    legacy_check = load_local_study("tsp_fast_legacy_reference_check")
    tsp_recalibrated = load_local_study("tsp_qf_recalibration_after_hardening")
    zdt1_recalibrated = load_local_study("zdt1_qf_recalibration_after_hardening")
    zdt1_recheck = load_local_study("zdt1_qf_tiny_freeze_recheck")
    knapsack_recheck = load_local_study("knapsack_note_freeze_recheck")
    onemax_recheck = load_local_study("onemax_control_freeze_recheck")

    assert tsp_study.problem == "tsp"
    assert tsp_study.analysis["qf_tolerance"]["plot_name_suffix"] == "_recalibrated"
    assert tsp_study.analysis["tsp_fast_tail"]["legacy_reference_plot_name"] == (
        "plot_old_fast_vs_new_fast_tail.png"
    )
    assert "budget_first_legacy" in tsp_confirm.sweep["study_variant"]
    assert legacy_check.analysis["tsp_fast_tail"]["baseline_fast_variant"] == "budget_first_legacy"
    assert tsp_recalibrated.variant_overrides["budget_first"]["population_size"] == 40
    assert tsp_recalibrated.variant_overrides["budget_first"]["generations"] == 33
    assert tsp_recalibrated.analysis["tsp_fast_tail"]["baseline_fast_variant"] == (
        "budget_first_legacy"
    )
    assert zdt1_recalibrated.variant_overrides["budget_first"]["algorithm_options"][
        "adaptation_cooldown"
    ] == 3
    assert zdt1_recheck.problem == "zdt1"
    assert zdt1_recheck.analysis["qf_tolerance"]["plot_name_suffix"] == "_recheck"
    assert knapsack_recheck.problem == "knapsack"
    assert "repair_only" in knapsack_recheck.variant_overrides
    assert onemax_recheck.problem == "onemax"
    assert "early_stop_reference" in onemax_recheck.sweep["study_variant"]


def test_seed_budget_studies_load() -> None:
    tsp_study = load_local_study("tsp_seed_budget_calibration")
    tsp_recheck = load_local_study("tsp_seed_budget_recheck_after_hardening")
    zdt1_study = load_local_study("zdt1_seed_budget_calibration")
    zdt1_recheck = load_local_study("zdt1_seed_budget_recheck_after_hardening")
    knapsack_study = load_local_study("knapsack_seed_budget_sanity")
    onemax_study = load_local_study("onemax_seed_budget_control")

    assert tsp_study.problem == "tsp"
    assert tsp_study.analysis["seed_budget"]["seed_counts"] == [1, 3, 5, 8, 10]
    assert tsp_recheck.analysis["seed_budget"]["seed_counts"] == [1, 3, 5, 8, 10]
    assert tsp_recheck.variant_overrides["budget_first"]["generations"] == 33
    assert zdt1_study.problem == "zdt1"
    assert zdt1_study.analysis["seed_budget"]["decision_tolerance_pct"] == 0.25
    assert zdt1_recheck.analysis["seed_budget"]["decision_tolerance_pct"] == 0.25
    assert zdt1_recheck.variant_overrides["budget_first"]["algorithm_options"][
        "adaptation_cooldown"
    ] == 3
    assert knapsack_study.problem == "knapsack"
    assert knapsack_study.analysis["seed_budget"]["repair_variant"] == "repair_only"
    assert onemax_study.problem == "onemax"
    assert onemax_study.analysis["seed_budget"]["control_variant"] == "none"


def test_sequential_compare_studies_load() -> None:
    tsp_study = load_local_study("tsp_sequential_compare_study")
    zdt1_study = load_local_study("zdt1_sequential_compare_study")
    knapsack_study = load_local_study("knapsack_sequential_sanity")
    onemax_study = load_local_study("onemax_control_sequential_check")

    assert tsp_study.problem == "tsp"
    assert tsp_study.analysis["sequential_compare"]["seed_stages"] == [3, 5, 8, 10]
    assert zdt1_study.problem == "zdt1"
    assert zdt1_study.analysis["sequential_compare"]["final_hv_tolerance_pct"] == 0.25
    assert knapsack_study.problem == "knapsack"
    assert knapsack_study.analysis["sequential_compare"]["repair_variant"] == "repair_only"
    assert onemax_study.problem == "onemax"
    assert onemax_study.analysis["sequential_compare"]["reference_variant"] == "early_stop_reference"


def test_stress_suite_studies_load() -> None:
    tsp_study = load_local_study("tsp_stress_suite")
    zdt1_study = load_local_study("zdt1_stress_suite")
    knapsack_study = load_local_study("knapsack_stress_suite")
    onemax_study = load_local_study("onemax_control_stress_check")
    tsp_refresh = load_local_study("tsp_stress_refresh_suite")
    zdt1_refresh = load_local_study("zdt1_stress_refresh_suite")
    knapsack_refresh = load_local_study("knapsack_stress_refresh_suite")
    onemax_refresh = load_local_study("onemax_control_refresh_check")

    assert tsp_study.problem == "tsp"
    assert tsp_study.analysis["stress_suite"]["fast_variant"] == "budget_first"
    assert "bridge_spoke_holdout_18" in {case.case_id for case in tsp_study.cases}
    assert zdt1_study.problem == "zdt1"
    assert zdt1_study.analysis["stress_suite"]["quality_variant"] == "quality_first"
    assert knapsack_study.problem == "knapsack"
    assert knapsack_study.analysis["stress_suite"]["repair_variant"] == "repair_only"
    assert onemax_study.problem == "onemax"
    assert onemax_study.analysis["stress_suite"]["control_variant"] == "none"
    assert tsp_refresh.variant_overrides["budget_first"]["population_size"] == 40
    assert tsp_refresh.variant_overrides["budget_first"]["generations"] == 33
    assert zdt1_refresh.variant_overrides["budget_first"]["algorithm_options"]["adaptation_cooldown"] == 3
    assert knapsack_refresh.analysis["stress_suite"]["budget_band_label"] == "repair_note_refresh"
    assert onemax_refresh.analysis["stress_suite"]["budget_band_label"] == "control_refresh"


def test_stress_hardening_studies_load() -> None:
    tsp_study = load_local_study("tsp_fast_stress_hardening_study")
    tsp_confirm = load_local_study("tsp_fast_stress_hardening_confirm")
    zdt1_study = load_local_study("zdt1_fast_stress_hardening_study")
    zdt1_confirm = load_local_study("zdt1_fast_stress_hardening_confirm")
    knapsack_study = load_local_study("knapsack_stress_note_freeze_check")
    onemax_study = load_local_study("onemax_control_stress_freeze_check")

    assert tsp_study.analysis["tsp_fast_tail"]["baseline_fast_variant"] == "current_fast"
    assert "micro_split_pop40_gen33" in tsp_study.sweep["study_variant"]
    assert "micro_inversion_pop32_gen42" in tsp_confirm.sweep["study_variant"]
    assert zdt1_study.analysis["zdt1_fast_hardening"]["baseline_fast_variant"] == "current_fast"
    assert "micro_refresh015" in zdt1_confirm.sweep["study_variant"]
    assert knapsack_study.seeds == (8301, 8302, 8303)
    assert onemax_study.seeds == (8401,)


def test_stress_target_reduction_studies_load() -> None:
    tsp_study = load_local_study("tsp_stress_target_reduction_study")
    tsp_confirm = load_local_study("tsp_stress_target_reduction_confirm")
    zdt1_study = load_local_study("zdt1_stress_target_reduction_study")
    zdt1_confirm = load_local_study("zdt1_stress_target_reduction_confirm")
    knapsack_study = load_local_study("knapsack_stress_freeze_check")
    onemax_study = load_local_study("onemax_control_stress_freeze_check")

    assert tsp_study.problem == "tsp"
    assert tsp_study.analysis["stress_target_reduction"]["primary_target"] == (
        "tsp_fast_anti_case_tail"
    )
    assert "micro_inversion_pop40_gen33" in tsp_confirm.sweep["study_variant"]
    assert zdt1_study.problem == "zdt1"
    assert zdt1_study.analysis["stress_target_reduction"]["primary_target"] == (
        "zdt1_fast_spread_safety_fail"
    )
    assert "micro_refresh012_cooldown3" in zdt1_confirm.sweep["study_variant"]
    assert knapsack_study.problem == "knapsack"
    assert knapsack_study.seeds == (8301, 8302, 8303)
    assert onemax_study.problem == "onemax"


def test_failure_trace_studies_load() -> None:
    tsp_study = load_local_study("tsp_failure_trace_suite")
    zdt1_study = load_local_study("zdt1_failure_trace_suite")
    knapsack_study = load_local_study("knapsack_stress_freeze_check")
    onemax_study = load_local_study("onemax_control_stress_freeze_check")

    assert tsp_study.problem == "tsp"
    assert tsp_study.analysis["failure_trace"]["target_id"] == "tsp_fast_anti_case_tail"
    assert "current_fast" in tsp_study.sweep["study_variant"]
    assert "budget_first_legacy" in tsp_study.analysis["failure_trace"]["reference_variants"]
    assert zdt1_study.problem == "zdt1"
    assert zdt1_study.analysis["failure_trace"]["secondary_target_id"] == (
        "zdt1_fast_joint_safety_fail"
    )
    assert "micro_refresh012_cooldown3" in zdt1_study.sweep["study_variant"]
    assert knapsack_study.problem == "knapsack"
    assert onemax_study.problem == "onemax"


def test_target_hypothesis_studies_load() -> None:
    tsp_study = load_local_study("tsp_target_hypothesis_probe_study")
    tsp_confirm = load_local_study("tsp_target_hypothesis_probe_confirm")
    zdt1_study = load_local_study("zdt1_target_hypothesis_probe_study")
    zdt1_confirm = load_local_study("zdt1_target_hypothesis_probe_confirm")

    assert tsp_study.problem == "tsp"
    assert tsp_study.analysis["failure_trace"]["hypothesis_probe"]["late_refinement_probe_variant"] == (
        "probe_late_refine_pop35_gen38"
    )
    assert tsp_study.analysis["failure_trace"]["hypothesis_probe"]["seed_fraction_probe_variant"] == (
        "probe_seed25_pop40_gen33"
    )
    assert "probe_pop45_gen29" in tsp_confirm.sweep["study_variant"]
    assert zdt1_study.problem == "zdt1"
    assert zdt1_study.analysis["failure_trace"]["hypothesis_probe"]["refresh_probe_variant"] == (
        "probe_refresh012_cooldown3"
    )
    assert zdt1_confirm.analysis["failure_trace"]["hypothesis_probe"]["cooldown_probe_variant"] == (
        "probe_cooldown2"
    )


def test_population_generation_and_timing_vs_pg_studies_load() -> None:
    tsp_study = load_local_study("tsp_population_generation_probe_study")
    tsp_confirm = load_local_study("tsp_population_generation_probe_confirm")
    zdt1_study = load_local_study("zdt1_timing_vs_pg_probe_study")
    zdt1_confirm = load_local_study("zdt1_timing_vs_pg_probe_confirm")

    assert tsp_study.problem == "tsp"
    assert tsp_study.analysis["failure_trace"]["hypothesis_probe"]["late_refinement_probe_variant"] == (
        "probe_pg_gen38_pop35"
    )
    assert tsp_study.analysis["failure_trace"]["hypothesis_probe"]["population_probe_variant"] == (
        "probe_pg_gen30_pop44"
    )
    assert tsp_study.analysis["failure_trace"]["hypothesis_probe"]["seed_fraction_probe_variant"] == (
        "probe_seed_fraction025"
    )
    assert "probe_pg_gen30_pop44" in tsp_confirm.sweep["study_variant"]

    assert zdt1_study.problem == "zdt1"
    assert zdt1_study.analysis["failure_trace"]["hypothesis_probe"]["cooldown_probe_variant"] == (
        "probe_timing_cooldown2"
    )
    assert zdt1_study.analysis["failure_trace"]["hypothesis_probe"][
        "population_generation_probe_variants"
    ] == ["probe_pg_pop41_gen88", "probe_pg_pop50_gen72"]
    assert "probe_pg_pop41_gen88" in zdt1_confirm.sweep["study_variant"]


def test_tail_first_and_split_target_studies_load() -> None:
    tsp_study = load_local_study("tsp_extreme_tail_pg_contour_study")
    tsp_confirm = load_local_study("tsp_extreme_tail_pg_contour_confirm")
    zdt1_spread_study = load_local_study("zdt1_spread_pg_probe_study")
    zdt1_spread_confirm = load_local_study("zdt1_spread_pg_probe_confirm")
    zdt1_joint_study = load_local_study("zdt1_joint_timing_probe_study")
    zdt1_joint_confirm = load_local_study("zdt1_joint_timing_probe_confirm")

    assert tsp_study.problem == "tsp"
    assert tsp_study.analysis["failure_trace"]["hypothesis_probe"]["tail_focus"] == "p95_max"
    assert tsp_study.analysis["failure_trace"]["hypothesis_probe"]["contour_probe_variants"] == [
        "contour_pg_gen34_pop39",
        "contour_pg_gen32_pop41",
        "contour_pg_gen31_pop42",
    ]
    assert tsp_confirm.analysis["failure_trace"]["hypothesis_probe"]["tail_focus"] == "p95_max"
    assert tsp_confirm.analysis["failure_trace"]["hypothesis_probe"]["contour_probe_variants"] == [
        "contour_pg_gen32_pop41",
        "contour_pg_gen31_pop42",
    ]
    assert "contour_pg_gen31_pop42" in tsp_confirm.sweep["study_variant"]

    assert zdt1_spread_study.problem == "zdt1"
    assert zdt1_spread_study.analysis["failure_trace"]["hypothesis_probe"]["probe_mode"] == (
        "spread_only"
    )
    assert zdt1_spread_study.analysis["failure_trace"]["hypothesis_probe"][
        "population_generation_probe_variants"
    ] == ["spread_pg_pop41_gen88", "spread_pg_pop44_gen82", "spread_pg_pop46_gen78"]
    assert "spread_pg_pop41_gen88" in zdt1_spread_confirm.sweep["study_variant"]

    assert zdt1_joint_study.problem == "zdt1"
    assert zdt1_joint_study.analysis["failure_trace"]["hypothesis_probe"]["probe_mode"] == (
        "joint_only"
    )
    assert zdt1_joint_study.analysis["failure_trace"]["hypothesis_probe"][
        "joint_unisolated_hypothesis_id"
    ] == "zdt1_fast_joint_mechanism_still_unisolated"
    assert zdt1_joint_confirm.analysis["failure_trace"]["hypothesis_probe"][
        "cooldown_probe_variant"
    ] == "joint_timing_cooldown2"


def test_tail_freeze_and_spread_candidate_validation_studies_load() -> None:
    tsp_freeze = load_local_study("tsp_tail_freeze_recheck")
    zdt1_spread_study = load_local_study("zdt1_spread_candidate_boundary_study")
    zdt1_spread_confirm = load_local_study("zdt1_spread_candidate_boundary_confirm")
    zdt1_joint_note = load_local_study("zdt1_joint_note_freeze_check")

    assert tsp_freeze.problem == "tsp"
    assert tsp_freeze.analysis["tsp_tail_freeze"]["fast_variant"] == "current_fast"
    assert tsp_freeze.analysis["failure_trace"]["hypothesis_probe"]["tail_focus"] == "p95_max"
    assert tsp_freeze.seeds == (9051, 9052, 9053, 9054, 9055, 9056, 9057, 9058, 9059, 9060)

    assert zdt1_spread_study.problem == "zdt1"
    assert zdt1_spread_study.analysis["zdt1_spread_candidate_boundary"]["candidate_variant"] == (
        "spread_pg_pop41_gen88"
    )
    assert "spread_pg_pop41_gen88" in zdt1_spread_confirm.sweep["study_variant"]
    assert zdt1_spread_confirm.analysis["zdt1_spread_candidate_boundary"]["slices"][
        "normal_holdout"
    ] == [8231, 8232, 8233]

    assert zdt1_joint_note.problem == "zdt1"
    assert zdt1_joint_note.analysis["failure_trace"]["hypothesis_probe"]["probe_mode"] == (
        "joint_only"
    )
    assert zdt1_joint_note.analysis["failure_trace"]["hypothesis_probe"][
        "cooldown_probe_variant"
    ] == "joint_timing_cooldown2"


def test_run_local_experiment_smoke(tmp_path: Path) -> None:
    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_experiment.py"),
        "--preset",
        "onemax_small",
        "--output-root",
        str(tmp_path),
    )

    assert Path(str(payload["local_dir"])).exists()
    assert Path(str(payload["raw_result_json"])).exists()
    assert Path(str(payload["summary_csv"])).exists()
    assert Path(str(payload["summary_md"])).exists()
    assert Path(str(payload["plots"]["plot_convergence"])).exists()


def test_run_local_sweep_and_rerender_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_tiny_study.json",
        {
            "study_name": "tsp_tiny_study",
            "description": "Tiny TSP sweep for automated smoke testing.",
            "problem": "tsp",
            "base_preset": "tsp_small",
            "shared_overrides": {"population_size": 20, "generations": 20},
            "sweep": {"mutation_rate": [0.02, 0.05]},
            "seeds": [1],
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny smoke study.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "studies"),
    )

    study_dir = Path(str(payload["study_dir"]))
    assert study_dir.exists()
    assert Path(str(payload["summary_csv"])).exists()
    assert Path(str(payload["summary_md"])).exists()
    assert Path(str(payload["raw_results_csv"])).exists()
    assert Path(str(payload["history_summary_csv"])).exists()
    assert Path(str(payload["plots"]["plot_convergence"])).exists()
    assert Path(str(payload["plots"]["plot_primary_metric"])).exists()

    rerendered = _run_json_command(
        str(_project_root() / "scripts" / "plot_local_results.py"),
        "--study-dir",
        str(study_dir),
    )
    assert Path(str(rerendered["plots"]["plot_convergence"])).exists()


def test_run_local_zdt1_sweep_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_tiny_study.json",
        {
            "study_name": "zdt1_tiny_study",
            "description": "Tiny ZDT1 sweep for automated smoke testing.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {"population_size": 20, "generations": 30},
            "sweep": {"mutation_rate": [0.1, 0.2]},
            "seeds": [1],
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny smoke study.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "studies"),
    )

    assert Path(str(payload["plots"]["plot_convergence"])).exists()
    assert Path(str(payload["plots"]["plot_primary_metric"])).exists()
    assert Path(str(payload["plots"]["plot_final_pareto_front"])).exists()


def test_tsp_fast_tail_hardening_smoke(tmp_path: Path) -> None:
    canonical_profile = _write_json_file(
        tmp_path,
        "tsp_fast_tail_q.json",
        {
            "run_name": "tsp_fast_tail_q",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 16,
            "genome_length": 8,
            "generations": 10,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "seed": 11,
                "bounds": [0.0, 100.0],
                "return_to_start": True,
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none",
            },
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_fast_tail_smoke.json",
        {
            "study_name": "tsp_fast_tail_smoke",
            "description": "Tiny smoke for TSP fast tail hardening.",
            "problem": "tsp",
            "base_config": str(canonical_profile),
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny rescue-target bridge case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0],
                                [2.0, 3.0],
                                [-2.0, 2.0],
                                [0.0, -3.0],
                                [30.0, 0.0],
                                [32.0, 3.0],
                                [28.0, -2.0],
                                [15.0, 10.0],
                            ],
                            "return_to_start": True,
                        },
                    },
                },
                {
                    "case_id": "corridor_8",
                    "group": "anti_case",
                    "note": "tiny anti-case corridor",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0],
                                [8.0, 3.0],
                                [16.0, -2.0],
                                [24.0, 4.0],
                                [32.0, -3.0],
                                [40.0, 5.0],
                                [48.0, -4.0],
                                [24.0, 14.0],
                            ],
                            "return_to_start": True,
                        },
                    },
                },
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {},
                "current_fast": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "fast_gen_cut_pop14_gen8": {
                    "population_size": 14,
                    "generations": 8,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "fast_seed25": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.25,
                        "local_search_strategy": "none",
                    },
                },
                "fast_inversion": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "inversion",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
            },
            "analysis": {
                "tsp_fast_tail": {
                    "quality_variant": "quality_first",
                    "baseline_fast_variant": "current_fast",
                }
            },
            "sweep": {
                "study_variant": [
                    "quality_first",
                    "current_fast",
                    "fast_gen_cut_pop14_gen8",
                    "fast_seed25",
                    "fast_inversion",
                ]
            },
            "seeds": [1],
            "budget_ceiling": 192,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny tail-hardening smoke study.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    tail_summary_rows = _read_csv_rows(Path(str(payload["tsp_fast_tail_summary_csv"])))
    overall_current_fast = next(
        row
        for row in tail_summary_rows
        if row["scope"] == "overall" and row["study_variant"] == "current_fast"
    )

    assert Path(str(payload["tsp_fast_tail_rows_csv"])).exists()
    assert Path(str(payload["plots"]["plot_q_vs_f_tail_distribution"])).exists()
    assert Path(str(payload["plots"]["plot_candidate_vs_anti_case_p90"])).exists()
    assert Path(str(payload["plots"]["plot_candidate_vs_rescue_mean"])).exists()
    assert Path(str(payload["plots"]["plot_budget_vs_tail_loss"])).exists()
    assert Path(str(payload["plots"]["plot_seed_fraction_vs_tail"])).exists()
    assert Path(str(payload["plots"]["plot_operator_vs_tail"])).exists()
    assert overall_current_fast["anti_case_p90_loss_pct"] != ""
    assert overall_current_fast["rescue_target_mean_loss_pct"] != ""


def test_tsp_fast_stress_hardening_smoke(tmp_path: Path) -> None:
    canonical_profile = _write_json_file(
        tmp_path,
        "tsp_fast_stress_q.json",
        {
            "run_name": "tsp_fast_stress_q",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 16,
            "genome_length": 8,
            "generations": 10,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "seed": 11,
                "bounds": [0.0, 100.0],
                "return_to_start": True
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none"
            }
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_fast_stress_hardening_smoke.json",
        {
            "study_name": "tsp_fast_stress_hardening_smoke",
            "description": "Tiny smoke for TSP fast stress hardening.",
            "problem": "tsp",
            "base_config": str(canonical_profile),
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny rescue-target bridge case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [2.0, 3.0], [-2.0, 2.0], [0.0, -3.0],
                                [30.0, 0.0], [32.0, 3.0], [28.0, -2.0], [15.0, 10.0]
                            ],
                            "return_to_start": True
                        }
                    }
                },
                {
                    "case_id": "corridor_8",
                    "group": "anti_case",
                    "note": "tiny anti-case corridor",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [8.0, 3.0], [16.0, -2.0], [24.0, 4.0],
                                [32.0, -3.0], [40.0, 5.0], [48.0, -4.0], [24.0, 14.0]
                            ],
                            "return_to_start": True
                        }
                    }
                }
            ],
            "shared_overrides": {
                "log_every": 1
            },
            "variant_overrides": {
                "quality_first": {},
                "current_fast": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none"
                    }
                },
                "micro_split": {
                    "population_size": 14,
                    "generations": 8,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none"
                    }
                },
                "micro_inversion": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "inversion",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none"
                    }
                }
            },
            "analysis": {
                "tsp_fast_tail": {
                    "quality_variant": "quality_first",
                    "baseline_fast_variant": "current_fast"
                }
            },
            "sweep": {
                "study_variant": [
                    "quality_first",
                    "current_fast",
                    "micro_split",
                    "micro_inversion"
                ]
            },
            "seeds": [1],
            "budget_ceiling": 192,
            "primary_metric": "best_route_distance",
            "plotting": {
                "history_metric": "best_route_distance"
            },
            "runtime_budget_note": "Tiny TSP fast stress-hardening smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["tail_risk_summary_csv"])).exists()
    assert Path(str(payload["plots"]["plot_currentF_vs_candidate_tail"])).exists()
    assert Path(str(payload["plots"]["plot_rescue_vs_anticase_tail"])).exists()
    assert Path(str(payload["plots"]["plot_budget_vs_tail_loss"])).exists()


def test_zdt1_qf_tiny_freeze_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_qf_tiny_freeze_smoke.json",
        {
            "study_name": "zdt1_qf_tiny_freeze_smoke",
            "description": "Tiny smoke for ZDT1 Q/F freeze checks.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {
                "population_size": 18,
                "generations": 18,
                "log_every": 1,
            },
            "variant_overrides": {
                "quality_first": {
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4,
                    }
                },
                "budget_first": {
                    "population_size": 14,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4,
                    }
                },
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                }
            },
            "sweep": {"study_variant": ["quality_first", "budget_first"]},
            "seeds": [1],
            "budget_ceiling": 360,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny ZDT1 Q/F freeze smoke.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["tolerance_table_csv"])).exists()
    assert Path(str(payload["plots"]["plot_q_vs_f_hv_loss_distribution"])).exists()
    assert Path(str(payload["plots"]["plot_tolerance_accept_rate"])).exists()
    assert Path(str(payload["plots"]["plot_spread_safety_failures"])).exists()
    assert Path(str(payload["plots"]["plot_budget_savings_vs_hv_loss"])).exists()


def test_zdt1_fast_stress_hardening_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_fast_stress_hardening_smoke.json",
        {
            "study_name": "zdt1_fast_stress_hardening_smoke",
            "description": "Tiny smoke for ZDT1 fast stress hardening.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {
                "population_size": 18,
                "generations": 18,
                "log_every": 1
            },
            "variant_overrides": {
                "quality_first": {
                    "population_size": 18,
                    "generations": 18,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4
                    }
                },
                "current_fast": {
                    "population_size": 14,
                    "generations": 18,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4
                    }
                },
                "micro_refresh015": {
                    "population_size": 14,
                    "generations": 18,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.15,
                        "adaptation_cooldown": 4
                    }
                },
                "micro_cooldown3": {
                    "population_size": 14,
                    "generations": 18,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3
                    }
                }
            },
            "analysis": {
                "zdt1_fast_hardening": {
                    "quality_variant": "quality_first",
                    "baseline_fast_variant": "current_fast",
                    "candidate_variants": [
                        "current_fast",
                        "micro_refresh015",
                        "micro_cooldown3"
                    ],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05
                }
            },
            "sweep": {
                "study_variant": [
                    "quality_first",
                    "current_fast",
                    "micro_refresh015",
                    "micro_cooldown3"
                ]
            },
            "seeds": [1, 2],
            "budget_ceiling": 1000,
            "primary_metric": "hypervolume",
            "plotting": {
                "history_metric": "hypervolume"
            },
            "runtime_budget_note": "Tiny ZDT1 fast stress-hardening smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["tail_risk_summary_csv"])).exists()
    assert Path(str(payload["zdt1_fast_hardening_rows_csv"])).exists()
    assert Path(str(payload["zdt1_fast_hardening_summary_csv"])).exists()
    assert Path(str(payload["plots"]["plot_currentF_vs_candidate_hv_loss"])).exists()
    assert Path(str(payload["plots"]["plot_spread_safety_failures"])).exists()
    assert Path(str(payload["plots"]["plot_budget_vs_hv_loss"])).exists()


def test_tsp_tail_freeze_smoke(tmp_path: Path) -> None:
    canonical_profile = _write_json_file(
        tmp_path,
        "tsp_tail_freeze_q.json",
        {
            "run_name": "tsp_tail_freeze_q",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 16,
            "genome_length": 8,
            "generations": 10,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "seed": 11,
                "bounds": [0.0, 100.0],
                "return_to_start": True,
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none",
            },
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_tail_freeze_smoke.json",
        {
            "study_name": "tsp_tail_freeze_smoke",
            "description": "Tiny smoke for the TSP irreducible-tail freeze summary.",
            "problem": "tsp",
            "base_config": str(canonical_profile),
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny rescue-target bridge case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0],
                                [2.0, 3.0],
                                [-2.0, 2.0],
                                [0.0, -3.0],
                                [30.0, 0.0],
                                [32.0, 3.0],
                                [28.0, -2.0],
                                [15.0, 10.0],
                            ],
                            "return_to_start": True,
                        },
                    },
                },
                {
                    "case_id": "corridor_8",
                    "group": "anti_case",
                    "note": "tiny anti-case corridor",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0],
                                [8.0, 3.0],
                                [16.0, -2.0],
                                [24.0, 4.0],
                                [32.0, -3.0],
                                [40.0, 5.0],
                                [48.0, -4.0],
                                [24.0, 14.0],
                            ],
                            "return_to_start": True,
                        },
                    },
                },
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {},
                "current_fast": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
            },
            "analysis": {
                "tsp_fast_tail": {
                    "quality_variant": "quality_first",
                    "baseline_fast_variant": "current_fast",
                },
                "tsp_tail_freeze": {
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                },
                "failure_trace": {
                    "target_id": "tsp_fast_anti_case_tail",
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                    "collapse_diversity_threshold": 0.12,
                    "major_improvement_min_delta": 1e-09,
                    "hypothesis_ids": [
                        "tsp_anticase_late_refinement_deficit",
                        "tsp_anticase_seed_lockin_secondary",
                    ],
                    "hypothesis_probe": {
                        "baseline_variant": "current_fast",
                        "focus_case_group": "anti_case",
                        "preserve_case_group": "rescue_target",
                        "tail_focus": "p95_max",
                    },
                },
            },
            "sweep": {"study_variant": ["quality_first", "current_fast"]},
            "seeds": [1],
            "budget_ceiling": 192,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny TSP tail-freeze smoke.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["tsp_fast_tail_summary_csv"])).exists()
    assert Path(str(payload["tsp_irreducible_tail_freeze_summary_md"])).exists()
    assert Path(str(payload["tsp_protocol_limitation_freeze_summary_md"])).exists()
    assert Path(str(payload["failure_trace_table_csv"])).exists()
    assert Path(str(payload["plots"]["plot_tsp_tail_freeze_summary"])).exists()
    assert Path(str(payload["plots"]["plot_anticase_q_vs_f_tail"])).exists()


def test_zdt1_spread_candidate_boundary_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_spread_candidate_boundary_smoke.json",
        {
            "study_name": "zdt1_spread_candidate_boundary_smoke",
            "description": "Tiny smoke for ZDT1 spread-candidate boundary validation.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {
                "population_size": 18,
                "generations": 18,
                "log_every": 1,
            },
            "variant_overrides": {
                "quality_first": {
                    "population_size": 18,
                    "generations": 18,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4,
                    },
                },
                "current_fast": {
                    "population_size": 14,
                    "generations": 18,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3,
                    },
                },
                "spread_pg_pop41_gen88": {
                    "population_size": 13,
                    "generations": 19,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3,
                    },
                },
            },
            "analysis": {
                "zdt1_fast_hardening": {
                    "quality_variant": "quality_first",
                    "baseline_fast_variant": "current_fast",
                    "candidate_variants": ["current_fast", "spread_pg_pop41_gen88"],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                },
                "zdt1_spread_candidate_boundary": {
                    "baseline_fast_variant": "current_fast",
                    "candidate_variant": "spread_pg_pop41_gen88",
                    "slices": {
                        "spread_stress": [1, 2],
                        "joint_non_regression": [2],
                        "stable_contrast": [1],
                        "normal_holdout": [3],
                    },
                },
                "failure_trace": {
                    "target_id": "zdt1_fast_spread_safety_fail",
                    "secondary_target_id": "zdt1_fast_joint_safety_fail",
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                    "reference_variants": ["spread_pg_pop41_gen88"],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                    "hv_plateau_slope_threshold": 0.005,
                    "hypothesis_ids": [
                        "zdt1_fast_spread_mechanism_population_generation_candidate",
                        "zdt1_fast_spread_mechanism_still_unisolated",
                    ],
                    "hypothesis_probe": {
                        "baseline_variant": "current_fast",
                        "population_generation_probe_variants": ["spread_pg_pop41_gen88"],
                        "probe_mode": "spread_only",
                    },
                },
            },
            "sweep": {"study_variant": ["quality_first", "current_fast", "spread_pg_pop41_gen88"]},
            "seeds": [1, 2, 3],
            "budget_ceiling": 1000,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny ZDT1 spread validation smoke.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["zdt1_spread_candidate_validation_summary_csv"])).exists()
    assert Path(str(payload["zdt1_spread_candidate_boundary_table_csv"])).exists()
    assert Path(str(payload["zdt1_spread_candidate_boundary_notes_md"])).exists()
    assert Path(str(payload["failure_trace_table_csv"])).exists()
    assert Path(str(payload["plots"]["plot_spread_candidate_vs_currentF"])).exists()
    assert Path(str(payload["plots"]["plot_spread_tail_validation"])).exists()
    assert Path(str(payload["plots"]["plot_hv_preservation_vs_spread_gain"])).exists()
    assert Path(str(payload["plots"]["plot_joint_non_regression"])).exists()
    assert Path(str(payload["plots"]["plot_stable_normal_non_regression"])).exists()


def test_zdt1_joint_note_freeze_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_joint_note_freeze_smoke.json",
        {
            "study_name": "zdt1_joint_note_freeze_smoke",
            "description": "Tiny smoke for ZDT1 joint-note freeze checks.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {
                "population_size": 18,
                "generations": 18,
                "log_every": 1,
            },
            "variant_overrides": {
                "quality_first": {
                    "population_size": 18,
                    "generations": 18,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4,
                    },
                },
                "current_fast": {
                    "population_size": 14,
                    "generations": 18,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3,
                    },
                },
                "joint_timing_cooldown2": {
                    "population_size": 14,
                    "generations": 18,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 2,
                    },
                },
            },
            "analysis": {
                "zdt1_fast_hardening": {
                    "quality_variant": "quality_first",
                    "baseline_fast_variant": "current_fast",
                    "candidate_variants": ["current_fast", "joint_timing_cooldown2"],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                },
                "failure_trace": {
                    "target_id": "zdt1_fast_spread_safety_fail",
                    "secondary_target_id": "zdt1_fast_joint_safety_fail",
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                    "reference_variants": ["joint_timing_cooldown2"],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                    "hv_plateau_slope_threshold": 0.005,
                    "hypothesis_ids": [
                        "zdt1_fast_joint_timing_mismatch_still_plausible",
                        "zdt1_fast_joint_mechanism_still_unisolated",
                    ],
                    "hypothesis_probe": {
                        "baseline_variant": "current_fast",
                        "cooldown_probe_variant": "joint_timing_cooldown2",
                        "probe_mode": "joint_only",
                        "joint_unisolated_hypothesis_id": "zdt1_fast_joint_mechanism_still_unisolated",
                    },
                },
            },
            "sweep": {"study_variant": ["quality_first", "current_fast", "joint_timing_cooldown2"]},
            "seeds": [1, 2],
            "budget_ceiling": 1000,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny ZDT1 joint-note freeze smoke.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["zdt1_fast_hardening_summary_csv"])).exists()
    assert Path(str(payload["failure_trace_table_csv"])).exists()
    assert Path(str(payload["failure_hypotheses_csv"])).exists()
    assert Path(str(payload["plots"]["plot_currentF_vs_candidate_hv_loss"])).exists()
    assert Path(str(payload["plots"]["plot_spread_safety_failures"])).exists()
    assert Path(str(payload["plots"]["plot_budget_vs_hv_loss"])).exists()


def test_tsp_two_stage_gate_smoke(tmp_path: Path) -> None:
    canonical_profile = _write_json_file(
        tmp_path,
        "tsp_gate_canonical.json",
        {
            "run_name": "tsp_gate_canonical",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 20,
            "genome_length": 8,
            "generations": 12,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "seed": 11,
                "bounds": [0.0, 100.0],
                "return_to_start": True,
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none",
            },
        },
    )
    fast_profile = _write_json_file(
        tmp_path,
        "tsp_gate_fast.json",
        {
            "run_name": "tsp_gate_fast",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 15,
            "genome_length": 8,
            "generations": 12,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "seed": 11,
                "bounds": [0.0, 100.0],
                "return_to_start": True,
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none",
            },
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_two_stage_gate_smoke.json",
        {
            "study_name": "tsp_two_stage_gate_smoke",
            "description": "Tiny smoke for the TSP two-stage gate.",
            "problem": "tsp",
            "base_config": str(canonical_profile),
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny bridge-like gate case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0],
                                [2.0, 3.0],
                                [-2.0, 2.0],
                                [0.0, -3.0],
                                [30.0, 0.0],
                                [32.0, 3.0],
                                [28.0, -2.0],
                                [15.0, 10.0],
                            ],
                        },
                    },
                }
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "always_canonical": {},
                "always_fast": {
                    "population_size": 15,
                    "generations": 12,
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "pilot_then_canonical_v1": {
                    "__local_baseline__": "tsp_two_stage_gate",
                    "algorithm_options": {
                        "gate_fast_profile": str(fast_profile),
                        "gate_canonical_profile": str(canonical_profile),
                        "gate_pilot_budget_fraction": 0.25,
                        "gate_signal_policy": "tsp_gain_or_stagnation",
                        "gate_min_gain_ratio": 999.0,
                        "gate_late_improvement_floor": 999,
                    },
                },
            },
            "sweep": {"study_variant": ["always_canonical", "always_fast", "pilot_then_canonical_v1"]},
            "seeds": [1],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny gate smoke study.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    summary_rows = _read_csv_rows(Path(str(payload["summary_csv"])))
    gate_row = next(
        row for row in summary_rows if row["study_variant"] == "pilot_then_canonical_v1"
    )

    assert Path(str(payload["plots"]["plot_pilot_fraction_vs_regret"])).exists()
    assert Path(str(payload["plots"]["plot_escalation_rate_vs_budget"])).exists()
    assert Path(str(payload["plots"]["plot_false_keep_vs_false_escalate"])).exists()
    assert Path(str(payload["plots"]["plot_actual_eval_vs_quality"])).exists()
    assert float(gate_row["escalation_rate"]) >= 1.0
    assert float(gate_row["pilot_actual_evaluations_used_mean"]) > 0.0
    assert float(gate_row["actual_evaluations_used_mean"]) >= float(
        gate_row["pilot_actual_evaluations_used_mean"]
    )
    assert gate_row["regret_vs_fast_profile_mean"] != ""


def test_zdt1_two_stage_gate_smoke(tmp_path: Path) -> None:
    canonical_profile = _write_json_file(
        tmp_path,
        "zdt1_gate_canonical.json",
        {
            "run_name": "zdt1_gate_canonical",
            "problem": "zdt1",
            "algorithm": "nsga2",
            "representation": "real",
            "selection": "tournament",
            "crossover": "arithmetic",
            "mutation": "gaussian",
            "mutation_options": {"sigma": 0.1},
            "population_size": 18,
            "genome_length": 10,
            "generations": 18,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "elitism": 1,
            "tournament_size": 2,
            "seed": 1,
            "maximize": False,
            "objective_directions": [False, False],
            "log_every": 1,
            "algorithm_options": {
                "adaptive_policy": "low_diversity_injection",
                "diversity_threshold": 0.55,
                "refresh_fraction": 0.1,
                "adaptation_cooldown": 4,
            },
        },
    )
    fast_profile = _write_json_file(
        tmp_path,
        "zdt1_gate_fast.json",
        {
            "run_name": "zdt1_gate_fast",
            "problem": "zdt1",
            "algorithm": "nsga2",
            "representation": "real",
            "selection": "tournament",
            "crossover": "arithmetic",
            "mutation": "gaussian",
            "mutation_options": {"sigma": 0.1},
            "population_size": 14,
            "genome_length": 10,
            "generations": 18,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "elitism": 1,
            "tournament_size": 2,
            "seed": 1,
            "maximize": False,
            "objective_directions": [False, False],
            "log_every": 1,
            "algorithm_options": {
                "adaptive_policy": "low_diversity_injection",
                "diversity_threshold": 0.55,
                "refresh_fraction": 0.1,
                "adaptation_cooldown": 4,
            },
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_two_stage_gate_smoke.json",
        {
            "study_name": "zdt1_two_stage_gate_smoke",
            "description": "Tiny smoke for the ZDT1 two-stage gate.",
            "problem": "zdt1",
            "base_config": str(canonical_profile),
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "always_canonical": {},
                "always_fast": {
                    "population_size": 14,
                    "generations": 18,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4,
                    },
                },
                "pilot_then_canonical_v1": {
                    "__local_baseline__": "zdt1_two_stage_gate",
                    "algorithm_options": {
                        "gate_fast_profile": str(fast_profile),
                        "gate_canonical_profile": str(canonical_profile),
                        "gate_pilot_budget_fraction": 0.25,
                        "gate_signal_policy": "zdt1_hv_or_plateau",
                        "gate_min_hypervolume": 999.0,
                        "gate_late_improvement_floor": 999,
                    },
                },
            },
            "sweep": {"study_variant": ["always_canonical", "always_fast", "pilot_then_canonical_v1"]},
            "seeds": [1],
            "budget_ceiling": 1000,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny gate smoke study.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    summary_rows = _read_csv_rows(Path(str(payload["summary_csv"])))
    gate_row = next(
        row for row in summary_rows if row["study_variant"] == "pilot_then_canonical_v1"
    )

    assert Path(str(payload["plots"]["plot_pilot_fraction_vs_hv"])).exists()
    assert Path(str(payload["plots"]["plot_escalation_rate_vs_hv"])).exists()
    assert Path(str(payload["plots"]["plot_actual_eval_vs_hv"])).exists()
    assert float(gate_row["escalation_rate"]) >= 1.0
    assert float(gate_row["pilot_actual_evaluations_used_mean"]) > 0.0
    assert float(gate_row["actual_evaluations_used_mean"]) >= float(
        gate_row["pilot_actual_evaluations_used_mean"]
    )
    assert gate_row["regret_vs_fast_profile_mean"] != ""


def test_tsp_restart_portfolio_smoke(tmp_path: Path) -> None:
    canonical_profile = _write_json_file(
        tmp_path,
        "tsp_portfolio_canonical.json",
        {
            "run_name": "tsp_portfolio_canonical",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 20,
            "genome_length": 8,
            "generations": 12,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "coordinates": [
                    [0.0, 0.0],
                    [2.0, 3.0],
                    [-2.0, 2.0],
                    [0.0, -3.0],
                    [30.0, 0.0],
                    [32.0, 3.0],
                    [28.0, -2.0],
                    [15.0, 10.0],
                ],
                "return_to_start": True,
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none",
            },
        },
    )
    fast_profile = _write_json_file(
        tmp_path,
        "tsp_portfolio_fast.json",
        {
            "run_name": "tsp_portfolio_fast",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 15,
            "genome_length": 8,
            "generations": 12,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "coordinates": [
                    [0.0, 0.0],
                    [2.0, 3.0],
                    [-2.0, 2.0],
                    [0.0, -3.0],
                    [30.0, 0.0],
                    [32.0, 3.0],
                    [28.0, -2.0],
                    [15.0, 10.0],
                ],
                "return_to_start": True,
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none",
            },
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_restart_portfolio_smoke.json",
        {
            "study_name": "tsp_restart_portfolio_smoke",
            "description": "Tiny smoke for the TSP restart portfolio.",
            "problem": "tsp",
            "base_config": str(canonical_profile),
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny bridge-like restart case",
                    "overrides": {"genome_length": 8},
                }
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "canonical_once": {},
                "fast_once": {
                    "population_size": 15,
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "fast_x2_equal_split": {
                    "__local_baseline__": "tsp_restart_portfolio",
                    "algorithm_options": {
                        "portfolio_profile": str(fast_profile),
                        "portfolio_restart_count": 2,
                        "portfolio_total_budget_factor": 1.0,
                    },
                },
            },
            "sweep": {"study_variant": ["canonical_once", "fast_once", "fast_x2_equal_split"]},
            "seeds": [1],
            "budget_ceiling": 300,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny restart portfolio smoke.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    summary_rows = _read_csv_rows(Path(str(payload["summary_csv"])))
    portfolio_row = next(row for row in summary_rows if row["study_variant"] == "fast_x2_equal_split")

    assert Path(str(payload["plots"]["plot_total_budget_vs_best_of_k"])).exists()
    assert Path(str(payload["plots"]["plot_restart_count_vs_regret"])).exists()
    assert Path(str(payload["plots"]["plot_multistart_vs_single_gap"])).exists()
    assert float(portfolio_row["portfolio_restart_count_mean"]) == 2.0
    assert portfolio_row["regret_vs_fast_once_mean"] != ""


def test_zdt1_restart_portfolio_smoke(tmp_path: Path) -> None:
    canonical_profile = _write_json_file(
        tmp_path,
        "zdt1_portfolio_canonical.json",
        {
            "run_name": "zdt1_portfolio_canonical",
            "problem": "zdt1",
            "algorithm": "nsga2",
            "representation": "real",
            "selection": "tournament",
            "crossover": "arithmetic",
            "mutation": "gaussian",
            "mutation_options": {"sigma": 0.1},
            "population_size": 18,
            "genome_length": 10,
            "generations": 18,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "elitism": 1,
            "tournament_size": 2,
            "seed": 1,
            "maximize": False,
            "objective_directions": [False, False],
            "log_every": 1,
            "algorithm_options": {
                "adaptive_policy": "low_diversity_injection",
                "diversity_threshold": 0.55,
                "refresh_fraction": 0.1,
                "adaptation_cooldown": 4,
            },
        },
    )
    fast_profile = _write_json_file(
        tmp_path,
        "zdt1_portfolio_fast.json",
        {
            "run_name": "zdt1_portfolio_fast",
            "problem": "zdt1",
            "algorithm": "nsga2",
            "representation": "real",
            "selection": "tournament",
            "crossover": "arithmetic",
            "mutation": "gaussian",
            "mutation_options": {"sigma": 0.1},
            "population_size": 14,
            "genome_length": 10,
            "generations": 18,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "elitism": 1,
            "tournament_size": 2,
            "seed": 1,
            "maximize": False,
            "objective_directions": [False, False],
            "log_every": 1,
            "algorithm_options": {
                "adaptive_policy": "low_diversity_injection",
                "diversity_threshold": 0.55,
                "refresh_fraction": 0.1,
                "adaptation_cooldown": 4,
            },
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_restart_portfolio_smoke.json",
        {
            "study_name": "zdt1_restart_portfolio_smoke",
            "description": "Tiny smoke for the ZDT1 restart portfolio.",
            "problem": "zdt1",
            "base_config": str(canonical_profile),
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "canonical_once": {},
                "fast_once": {
                    "population_size": 14,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4,
                    },
                },
                "fast_x2_equal_split": {
                    "__local_baseline__": "zdt1_restart_portfolio",
                    "algorithm_options": {
                        "portfolio_profile": str(fast_profile),
                        "portfolio_restart_count": 2,
                        "portfolio_total_budget_factor": 1.0,
                    },
                },
            },
            "sweep": {"study_variant": ["canonical_once", "fast_once", "fast_x2_equal_split"]},
            "seeds": [1],
            "budget_ceiling": 1000,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny restart portfolio smoke.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    summary_rows = _read_csv_rows(Path(str(payload["summary_csv"])))
    portfolio_row = next(row for row in summary_rows if row["study_variant"] == "fast_x2_equal_split")

    assert Path(str(payload["plots"]["plot_total_budget_vs_merged_hv"])).exists()
    assert Path(str(payload["plots"]["plot_restart_count_vs_hv"])).exists()
    assert Path(str(payload["plots"]["plot_merged_archive_vs_single_run"])).exists()
    assert float(portfolio_row["portfolio_restart_count_mean"]) == 2.0
    assert portfolio_row["merged_archive_hv_mean"] != ""


def test_knapsack_multistart_sanity_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "knapsack_multistart_sanity_smoke.json",
        {
            "study_name": "knapsack_multistart_sanity_smoke",
            "description": "Tiny smoke for the knapsack multistart sanity pass.",
            "problem": "knapsack",
            "base_preset": "knapsack_small",
            "cases": [
                {
                    "case_id": "tight_knapsack",
                    "family_label": "tight_capacity_small",
                    "note": "tiny tight-capacity sanity case",
                    "overrides": {
                        "genome_length": 10,
                        "problem_options": {
                            "num_items": 10,
                            "seed": 17,
                            "capacity": 24.0,
                            "weight_scale": [1.0, 10.0],
                            "value_scale": [1.0, 12.0],
                            "penalty_factor": 20.0,
                        },
                    },
                }
            ],
            "shared_overrides": {
                "population_size": 16,
                "generations": 16,
                "log_every": 1,
            },
            "variant_overrides": {
                "none": {"algorithm": "ga"},
                "repair_only": {
                    "algorithm": "hybrid_ga",
                    "algorithm_options": {
                        "repair_strategy": "knapsack_greedy_fill",
                        "local_search_strategy": "none",
                    },
                },
                "none_x2_equal_split": {
                    "__local_baseline__": "knapsack_restart_portfolio",
                    "algorithm_options": {
                        "portfolio_restart_count": 2,
                        "portfolio_total_budget_factor": 1.0,
                    },
                },
            },
            "sweep": {"study_variant": ["none", "repair_only", "none_x2_equal_split"]},
            "seeds": [1],
            "budget_ceiling": 400,
            "primary_metric": "best_feasible_fitness",
            "plotting": {"history_metric": "best_feasible_fitness"},
            "runtime_budget_note": "Tiny knapsack multistart smoke.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    summary_rows = _read_csv_rows(Path(str(payload["summary_csv"])))
    portfolio_row = next(row for row in summary_rows if row["study_variant"] == "none_x2_equal_split")

    assert Path(str(payload["plots"]["plot_multistart_vs_repair_gap"])).exists()
    assert float(portfolio_row["portfolio_restart_count_mean"]) == 2.0
    assert portfolio_row["regret_vs_repair_only_reference_mean"] != ""


def test_tsp_ranking_fidelity_smoke(tmp_path: Path) -> None:
    base_config = _write_json_file(
        tmp_path,
        "tsp_triage_base.json",
        {
            "run_name": "tsp_triage_base",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 20,
            "genome_length": 8,
            "generations": 12,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "seed": 11,
                "bounds": [0.0, 100.0],
                "return_to_start": True
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none"
            }
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_ranking_fidelity_smoke.json",
        {
            "study_name": "tsp_ranking_fidelity_smoke",
            "description": "Tiny TSP ranking-fidelity smoke.",
            "problem": "tsp",
            "base_config": str(base_config),
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny bridge-like case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [2.0, 3.0], [-2.0, 2.0], [0.0, -3.0],
                                [30.0, 0.0], [32.0, 3.0], [28.0, -2.0], [15.0, 10.0]
                            ]
                        }
                    }
                }
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "seeded_swap_seed50__canonical": {},
                "seeded_swap_seed50__fast": {"population_size": 14},
                "none__canonical": {
                    "algorithm": "ga",
                    "mutation": "swap",
                    "algorithm_options": {"adaptive_policy": "none"}
                },
                "none__fast": {
                    "algorithm": "ga",
                    "mutation": "swap",
                    "population_size": 14,
                    "algorithm_options": {"adaptive_policy": "none"}
                }
            },
            "sweep": {
                "study_variant": [
                    "seeded_swap_seed50__canonical",
                    "seeded_swap_seed50__fast",
                    "none__canonical",
                    "none__fast"
                ]
            },
            "seeds": [1, 2],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny TSP triage smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    ranking_rows = _read_csv_rows(Path(str(payload["ranking_fidelity_csv"])))
    workflow_rows = _read_csv_rows(Path(str(payload["triage_workflow_summary_csv"])))

    assert ranking_rows
    assert workflow_rows
    assert Path(str(payload["plots"]["plot_fast_vs_canonical_rank"])).exists()
    assert Path(str(payload["plots"]["plot_topk_recall_vs_budget"])).exists()
    assert Path(str(payload["plots"]["plot_triage_cost_vs_regret"])).exists()
    assert any(row["workflow"] == "fast_screen_then_confirm_top2" for row in workflow_rows)


def test_zdt1_ranking_fidelity_smoke(tmp_path: Path) -> None:
    base_config = _write_json_file(
        tmp_path,
        "zdt1_triage_base.json",
        {
            "run_name": "zdt1_triage_base",
            "problem": "zdt1",
            "algorithm": "nsga2",
            "representation": "real",
            "selection": "tournament",
            "crossover": "arithmetic",
            "mutation": "gaussian",
            "mutation_options": {"sigma": 0.1},
            "population_size": 18,
            "genome_length": 10,
            "generations": 18,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "elitism": 1,
            "tournament_size": 2,
            "seed": 1,
            "maximize": False,
            "objective_directions": [False, False],
            "log_every": 1,
            "algorithm_options": {
                "adaptive_policy": "low_diversity_injection",
                "diversity_threshold": 0.55,
                "refresh_fraction": 0.1,
                "adaptation_cooldown": 4
            }
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_ranking_fidelity_smoke.json",
        {
            "study_name": "zdt1_ranking_fidelity_smoke",
            "description": "Tiny ZDT1 ranking-fidelity smoke.",
            "problem": "zdt1",
            "base_config": str(base_config),
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "threshold_055_refresh010__canonical": {},
                "threshold_055_refresh010__fast": {"population_size": 14},
                "none__canonical": {
                    "algorithm_options": {"adaptive_policy": "none"}
                },
                "none__fast": {
                    "population_size": 14,
                    "algorithm_options": {"adaptive_policy": "none"}
                }
            },
            "sweep": {
                "study_variant": [
                    "threshold_055_refresh010__canonical",
                    "threshold_055_refresh010__fast",
                    "none__canonical",
                    "none__fast"
                ]
            },
            "seeds": [1, 2],
            "budget_ceiling": 1000,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny ZDT1 triage smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    ranking_rows = _read_csv_rows(Path(str(payload["ranking_fidelity_csv"])))
    workflow_rows = _read_csv_rows(Path(str(payload["triage_workflow_summary_csv"])))

    assert ranking_rows
    assert workflow_rows
    assert Path(str(payload["plots"]["plot_fast_vs_canonical_hv_rank"])).exists()
    assert Path(str(payload["plots"]["plot_topk_recall_vs_budget"])).exists()
    assert Path(str(payload["plots"]["plot_triage_cost_vs_hv_regret"])).exists()
    assert Path(str(payload["plots"]["plot_spread_safety_failures"])).exists()
    assert ranking_rows[0]["spearman_rank_correlation"] != ""


def test_tsp_qf_tolerance_smoke(tmp_path: Path) -> None:
    base_config = _write_json_file(
        tmp_path,
        "tsp_qf_base.json",
        {
            "run_name": "tsp_qf_base",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 20,
            "genome_length": 8,
            "generations": 16,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "seed": 11,
                "bounds": [0.0, 100.0],
                "return_to_start": True
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none"
            }
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_qf_tolerance_smoke.json",
        {
            "study_name": "tsp_qf_tolerance_smoke",
            "description": "Tiny TSP Q/F tolerance smoke.",
            "problem": "tsp",
            "base_config": str(base_config),
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny bridge-like case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [2.0, 3.0], [-2.0, 2.0], [0.0, -3.0],
                                [30.0, 0.0], [32.0, 3.0], [28.0, -2.0], [15.0, 10.0]
                            ]
                        }
                    }
                },
                {
                    "case_id": "corridor_8",
                    "group": "anti_case",
                    "note": "tiny corridor-like case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [6.0, 2.0], [12.0, -1.0], [18.0, 3.0],
                                [24.0, -2.0], [30.0, 4.0], [36.0, -3.0], [20.0, 12.0]
                            ]
                        }
                    }
                }
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {},
                "budget_first": {"population_size": 14}
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0]
                }
            },
            "sweep": {"study_variant": ["quality_first", "budget_first"]},
            "seeds": [1, 2],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny TSP Q/F tolerance smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    tolerance_rows = _read_csv_rows(Path(str(payload["tolerance_table_csv"])))
    assert tolerance_rows
    assert Path(str(payload["plots"]["plot_q_vs_f_loss_distribution"])).exists()
    assert Path(str(payload["plots"]["plot_tolerance_accept_rate"])).exists()
    assert Path(str(payload["plots"]["plot_rescue_vs_anticase_loss"])).exists()
    assert Path(str(payload["plots"]["plot_budget_savings_vs_quality_loss"])).exists()
    assert any(row["case_group"] == "rescue_target" for row in tolerance_rows)
    assert any(row["case_group"] == "anti_case" for row in tolerance_rows)


def test_tsp_qf_recalibration_smoke(tmp_path: Path) -> None:
    base_config = _write_json_file(
        tmp_path,
        "tsp_qf_recalibration_base.json",
        {
            "run_name": "tsp_qf_recalibration_base",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 20,
            "genome_length": 8,
            "generations": 16,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "seed": 11,
                "bounds": [0.0, 100.0],
                "return_to_start": True
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none"
            }
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_qf_recalibration_smoke.json",
        {
            "study_name": "tsp_qf_recalibration_smoke",
            "description": "Tiny TSP Q/F recalibration smoke.",
            "problem": "tsp",
            "base_config": str(base_config),
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny bridge-like case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [2.0, 3.0], [-2.0, 2.0], [0.0, -3.0],
                                [30.0, 0.0], [32.0, 3.0], [28.0, -2.0], [15.0, 10.0]
                            ]
                        }
                    }
                },
                {
                    "case_id": "corridor_8",
                    "group": "anti_case",
                    "note": "tiny corridor-like case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [6.0, 2.0], [12.0, -1.0], [18.0, 3.0],
                                [24.0, -2.0], [30.0, 4.0], [36.0, -3.0], [20.0, 12.0]
                            ]
                        }
                    }
                }
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {},
                "budget_first_legacy": {"population_size": 12, "generations": 16},
                "budget_first": {"population_size": 14, "generations": 14}
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0],
                    "plot_name_suffix": "_recalibrated"
                },
                "tsp_fast_tail": {
                    "quality_variant": "quality_first",
                    "baseline_fast_variant": "budget_first_legacy",
                    "legacy_reference_case_group": "anti_case",
                    "legacy_reference_plot_name": "plot_old_fast_vs_new_fast_tail.png"
                }
            },
            "sweep": {
                "study_variant": ["quality_first", "budget_first_legacy", "budget_first"]
            },
            "seeds": [1, 2],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny TSP Q/F recalibration smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    tolerance_rows = _read_csv_rows(Path(str(payload["tolerance_table_csv"])))
    tail_rows = _read_csv_rows(Path(str(payload["tsp_fast_tail_summary_csv"])))
    assert tolerance_rows
    assert tail_rows
    assert Path(str(payload["plots"]["plot_q_vs_f_loss_distribution"])).name.endswith(
        "_recalibrated.png"
    )
    assert Path(str(payload["plots"]["plot_tolerance_accept_rate"])).name.endswith(
        "_recalibrated.png"
    )
    assert Path(str(payload["plots"]["plot_rescue_vs_anticase_loss"])).name.endswith(
        "_recalibrated.png"
    )
    assert Path(str(payload["plots"]["plot_budget_savings_vs_quality_loss"])).name.endswith(
        "_recalibrated.png"
    )
    assert Path(str(payload["plots"]["plot_old_fast_vs_new_fast_tail"])).exists()
    assert any(row["study_variant"] == "budget_first" for row in tail_rows)
    assert any(row["case_group"] == "anti_case" for row in tolerance_rows)


def test_zdt1_qf_tolerance_smoke(tmp_path: Path) -> None:
    base_config = _write_json_file(
        tmp_path,
        "zdt1_qf_base.json",
        {
            "run_name": "zdt1_qf_base",
            "problem": "zdt1",
            "algorithm": "nsga2",
            "representation": "real",
            "selection": "tournament",
            "crossover": "arithmetic",
            "mutation": "gaussian",
            "mutation_options": {"sigma": 0.1},
            "population_size": 18,
            "genome_length": 10,
            "generations": 20,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "elitism": 1,
            "tournament_size": 2,
            "seed": 1,
            "maximize": False,
            "objective_directions": [False, False],
            "log_every": 1,
            "algorithm_options": {
                "adaptive_policy": "low_diversity_injection",
                "diversity_threshold": 0.55,
                "refresh_fraction": 0.1,
                "adaptation_cooldown": 4
            }
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_qf_tolerance_smoke.json",
        {
            "study_name": "zdt1_qf_tolerance_smoke",
            "description": "Tiny ZDT1 Q/F tolerance smoke.",
            "problem": "zdt1",
            "base_config": str(base_config),
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {},
                "budget_first": {"population_size": 14}
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05
                }
            },
            "sweep": {"study_variant": ["quality_first", "budget_first"]},
            "seeds": [1, 2],
            "budget_ceiling": 378,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny ZDT1 Q/F tolerance smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    tolerance_rows = _read_csv_rows(Path(str(payload["tolerance_table_csv"])))
    assert tolerance_rows
    assert Path(str(payload["plots"]["plot_q_vs_f_hv_loss_distribution"])).exists()
    assert Path(str(payload["plots"]["plot_tolerance_accept_rate"])).exists()
    assert Path(str(payload["plots"]["plot_spread_safety_failures"])).exists()
    assert Path(str(payload["plots"]["plot_budget_savings_vs_hv_loss"])).exists()
    assert any(row["hv_only_accept_rate"] != "" for row in tolerance_rows)


def test_zdt1_qf_tiny_freeze_recheck_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_qf_tiny_freeze_recheck_smoke.json",
        {
            "study_name": "zdt1_qf_tiny_freeze_recheck_smoke",
            "description": "Tiny ZDT1 Q/F freeze recheck smoke.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {"population_size": 18, "generations": 18, "log_every": 1},
            "variant_overrides": {
                "quality_first": {
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4
                    }
                },
                "budget_first": {
                    "population_size": 14,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4
                    }
                }
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                    "plot_name_suffix": "_recheck"
                }
            },
            "sweep": {"study_variant": ["quality_first", "budget_first"]},
            "seeds": [1],
            "budget_ceiling": 360,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny ZDT1 Q/F freeze recheck smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["tolerance_table_csv"])).exists()
    assert Path(str(payload["plots"]["plot_q_vs_f_hv_loss_distribution"])).name.endswith(
        "_recheck.png"
    )
    assert Path(str(payload["plots"]["plot_tolerance_accept_rate"])).exists()
    assert Path(str(payload["plots"]["plot_spread_safety_failures"])).exists()
    assert Path(str(payload["plots"]["plot_budget_savings_vs_hv_loss"])).exists()


def test_tsp_seed_budget_smoke(tmp_path: Path) -> None:
    base_config = _write_json_file(
        tmp_path,
        "tsp_seed_budget_base.json",
        {
            "run_name": "tsp_seed_budget_base",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 20,
            "genome_length": 8,
            "generations": 16,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "seed": 11,
                "bounds": [0.0, 100.0],
                "return_to_start": True
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none"
            }
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_seed_budget_smoke.json",
        {
            "study_name": "tsp_seed_budget_smoke",
            "description": "Tiny TSP seed-budget calibration smoke.",
            "problem": "tsp",
            "base_config": str(base_config),
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny bridge-like case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [2.0, 3.0], [-2.0, 2.0], [0.0, -3.0],
                                [30.0, 0.0], [32.0, 3.0], [28.0, -2.0], [15.0, 10.0]
                            ]
                        }
                    }
                },
                {
                    "case_id": "corridor_8",
                    "group": "anti_case",
                    "note": "tiny corridor-like case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [6.0, 2.0], [12.0, -1.0], [18.0, 3.0],
                                [24.0, -2.0], [30.0, 4.0], [36.0, -3.0], [20.0, 12.0]
                            ]
                        }
                    }
                }
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {},
                "budget_first": {"population_size": 14, "generations": 14}
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0]
                },
                "seed_budget": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "seed_counts": [1, 3],
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0],
                    "decision_tolerance_pct": 0.5
                }
            },
            "sweep": {"study_variant": ["quality_first", "budget_first"]},
            "seeds": [1, 2, 3],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny TSP seed-budget smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    seed_rows = _read_csv_rows(Path(str(payload["seed_budget_table_csv"])))
    assert seed_rows
    assert Path(str(payload["plots"]["plot_seed_count_vs_loss_ci"])).exists()
    assert Path(str(payload["plots"]["plot_seed_count_vs_decision_flip"])).exists()
    assert Path(str(payload["plots"]["plot_seed_count_vs_accept_rate"])).exists()
    assert Path(str(payload["plots"]["plot_rescue_vs_anticase_seed_stability"])).exists()
    assert any(row["case_group"] == "rescue_target" for row in seed_rows)
    assert any(row["case_group"] == "anti_case" for row in seed_rows)


def test_zdt1_seed_budget_smoke(tmp_path: Path) -> None:
    base_config = _write_json_file(
        tmp_path,
        "zdt1_seed_budget_base.json",
        {
            "run_name": "zdt1_seed_budget_base",
            "problem": "zdt1",
            "algorithm": "nsga2",
            "representation": "real",
            "selection": "tournament",
            "crossover": "arithmetic",
            "mutation": "gaussian",
            "mutation_options": {"sigma": 0.1},
            "population_size": 18,
            "genome_length": 10,
            "generations": 20,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "elitism": 1,
            "tournament_size": 2,
            "seed": 1,
            "maximize": False,
            "objective_directions": [False, False],
            "log_every": 1,
            "algorithm_options": {
                "adaptive_policy": "low_diversity_injection",
                "diversity_threshold": 0.55,
                "refresh_fraction": 0.1,
                "adaptation_cooldown": 4
            }
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_seed_budget_smoke.json",
        {
            "study_name": "zdt1_seed_budget_smoke",
            "description": "Tiny ZDT1 seed-budget calibration smoke.",
            "problem": "zdt1",
            "base_config": str(base_config),
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {},
                "budget_first": {"population_size": 14}
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05
                },
                "seed_budget": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "seed_counts": [1, 3],
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0],
                    "decision_tolerance_pct": 0.25
                }
            },
            "sweep": {"study_variant": ["quality_first", "budget_first"]},
            "seeds": [1, 2, 3],
            "budget_ceiling": 378,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny ZDT1 seed-budget smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    seed_rows = _read_csv_rows(Path(str(payload["seed_budget_table_csv"])))
    assert seed_rows
    assert Path(str(payload["plots"]["plot_seed_count_vs_hv_ci"])).exists()
    assert Path(str(payload["plots"]["plot_seed_count_vs_safety_fail_rate"])).exists()
    assert Path(str(payload["plots"]["plot_seed_count_vs_decision_flip"])).exists()
    assert any(row["decision"] != "" for row in seed_rows)


def test_tsp_sequential_compare_smoke(tmp_path: Path) -> None:
    base_config = _write_json_file(
        tmp_path,
        "tsp_sequential_base.json",
        {
            "run_name": "tsp_sequential_base",
            "problem": "tsp",
            "algorithm": "hybrid_ga",
            "representation": "permutation",
            "selection": "tournament",
            "crossover": "order",
            "mutation": "swap",
            "population_size": 20,
            "genome_length": 8,
            "generations": 16,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "elitism": 1,
            "tournament_size": 3,
            "seed": 1,
            "maximize": True,
            "log_every": 1,
            "problem_options": {
                "num_cities": 8,
                "seed": 11,
                "bounds": [0.0, 100.0],
                "return_to_start": True
            },
            "algorithm_options": {
                "adaptive_policy": "none",
                "init_strategy": "tsp_nearest_neighbor_mix",
                "seed_fraction": 0.5,
                "local_search_strategy": "none"
            }
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_sequential_compare_smoke.json",
        {
            "study_name": "tsp_sequential_compare_smoke",
            "description": "Tiny TSP sequential compare smoke.",
            "problem": "tsp",
            "base_config": str(base_config),
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny bridge-like case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [2.0, 3.0], [-2.0, 2.0], [0.0, -3.0],
                                [30.0, 0.0], [32.0, 3.0], [28.0, -2.0], [15.0, 10.0]
                            ]
                        }
                    }
                },
                {
                    "case_id": "corridor_8",
                    "group": "anti_case",
                    "note": "tiny corridor-like case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0], [6.0, 2.0], [12.0, -1.0], [18.0, 3.0],
                                [24.0, -2.0], [30.0, 4.0], [36.0, -3.0], [20.0, 12.0]
                            ]
                        }
                    }
                }
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {},
                "budget_first": {"population_size": 14, "generations": 14}
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0]
                },
                "sequential_compare": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "seed_stages": [3],
                    "modes": ["exploratory", "quality_sensitive"],
                    "bootstrap_seed": 901,
                    "bootstrap_replicates": 100
                }
            },
            "sweep": {"study_variant": ["quality_first", "budget_first"]},
            "seeds": [1, 2, 3],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny TSP sequential smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    rows = _read_csv_rows(Path(str(payload["sequential_decision_table_csv"])))
    assert rows
    assert any(row["mode"] == "exploratory" for row in rows)
    assert Path(str(payload["plots"]["plot_seed_stage_vs_ci_width"])).exists()
    assert Path(str(payload["plots"]["plot_seed_stage_vs_decision_flip"])).exists()
    assert Path(str(payload["plots"]["plot_rescue_vs_anticase_escalation_rate"])).exists()
    assert Path(str(payload["plots"]["plot_cost_savings_vs_false_decision"])).exists()


def test_zdt1_sequential_compare_smoke(tmp_path: Path) -> None:
    base_config = _write_json_file(
        tmp_path,
        "zdt1_sequential_base.json",
        {
            "run_name": "zdt1_sequential_base",
            "problem": "zdt1",
            "algorithm": "nsga2",
            "representation": "real",
            "selection": "tournament",
            "crossover": "arithmetic",
            "mutation": "gaussian",
            "mutation_options": {"sigma": 0.1},
            "population_size": 18,
            "genome_length": 10,
            "generations": 20,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "elitism": 1,
            "tournament_size": 2,
            "seed": 1,
            "maximize": False,
            "objective_directions": [False, False],
            "log_every": 1,
            "algorithm_options": {
                "adaptive_policy": "low_diversity_injection",
                "diversity_threshold": 0.55,
                "refresh_fraction": 0.1,
                "adaptation_cooldown": 4
            }
        },
    )
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_sequential_compare_smoke.json",
        {
            "study_name": "zdt1_sequential_compare_smoke",
            "description": "Tiny ZDT1 sequential compare smoke.",
            "problem": "zdt1",
            "base_config": str(base_config),
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {},
                "budget_first": {"population_size": 14}
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.1, 0.25, 0.5, 1.0],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05
                },
                "sequential_compare": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "seed_stages": [3],
                    "modes": ["exploratory", "final_safety"],
                    "bootstrap_seed": 902,
                    "bootstrap_replicates": 100
                }
            },
            "sweep": {"study_variant": ["quality_first", "budget_first"]},
            "seeds": [1, 2, 3],
            "budget_ceiling": 378,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny ZDT1 sequential smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    rows = _read_csv_rows(Path(str(payload["sequential_decision_table_csv"])))
    assert rows
    assert any(row["mode"] == "final_safety" for row in rows)
    assert Path(str(payload["plots"]["plot_seed_stage_vs_hv_ci"])).exists()
    assert Path(str(payload["plots"]["plot_seed_stage_vs_safety_fail_rate"])).exists()
    assert Path(str(payload["plots"]["plot_seed_stage_vs_decision_flip"])).exists()


def test_knapsack_sequential_sanity_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "knapsack_sequential_sanity_smoke.json",
        {
            "study_name": "knapsack_sequential_sanity_smoke",
            "description": "Tiny knapsack sequential sanity smoke.",
            "problem": "knapsack",
            "base_preset": "knapsack_small",
            "cases": [
                {
                    "case_id": "tight_capacity_small_smoke",
                    "group": "tight_capacity_small",
                    "note": "tiny tight-capacity case",
                    "overrides": {
                        "genome_length": 12,
                        "__study_metadata__": {
                            "family_label": "tight_capacity_small",
                            "capacity_ratio": 0.2,
                            "correlation_note": "smoke"
                        },
                        "problem_options": {
                            "num_items": 12,
                            "weights": [2, 3, 5, 7, 11, 13, 4, 6, 8, 9, 10, 12],
                            "values": [3, 5, 9, 13, 22, 25, 7, 10, 13, 14, 16, 18],
                            "capacity": 28.0,
                            "penalty_factor": 20.0
                        }
                    }
                }
            ],
            "shared_overrides": {
                "population_size": 30,
                "generations": 20,
                "log_every": 1,
                "mutation_rate": 0.02
            },
            "variant_overrides": {
                "greedy_local_search": {
                    "__local_baseline__": "knapsack_greedy_local_search",
                    "algorithm_options": {"adaptive_policy": "none"}
                },
                "none": {
                    "algorithm": "ga",
                    "algorithm_options": {"adaptive_policy": "none"}
                },
                "repair_only": {
                    "algorithm": "hybrid_ga",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "none",
                        "seed_fraction": 0.0,
                        "repair_strategy": "knapsack_greedy_fill",
                        "local_search_strategy": "none"
                    }
                }
            },
            "analysis": {
                "sequential_compare": {
                    "seed_stages": [3],
                    "none_variant": "none",
                    "repair_variant": "repair_only",
                    "greedy_variant": "greedy_local_search",
                    "bootstrap_seed": 903,
                    "bootstrap_replicates": 100
                }
            },
            "sweep": {"study_variant": ["greedy_local_search", "none", "repair_only"]},
            "seeds": [1, 2, 3],
            "budget_ceiling": 660,
            "primary_metric": "best_feasible_fitness",
            "plotting": {"history_metric": "best_fitness"},
            "runtime_budget_note": "Tiny knapsack sequential smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    rows = _read_csv_rows(Path(str(payload["sequential_decision_table_csv"])))
    assert rows
    assert Path(str(payload["plots"]["plot_seed_stage_vs_repair_note_stability"])).exists()


def test_knapsack_triage_sanity_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "knapsack_triage_sanity_smoke.json",
        {
            "study_name": "knapsack_triage_sanity_smoke",
            "description": "Tiny knapsack triage sanity smoke.",
            "problem": "knapsack",
            "base_preset": "knapsack_small",
            "cases": [
                {
                    "case_id": "tight_capacity_small_smoke",
                    "group": "boundary_like",
                    "note": "tiny tight-capacity case",
                    "overrides": {
                        "genome_length": 12,
                        "__study_metadata__": {
                            "family_label": "tight_capacity_small",
                            "capacity_ratio": 0.2,
                            "correlation_note": "smoke"
                        },
                        "problem_options": {
                            "num_items": 12,
                            "weights": [2, 3, 5, 7, 11, 13, 4, 6, 8, 9, 10, 12],
                            "values": [3, 5, 9, 13, 22, 25, 7, 10, 13, 14, 16, 18],
                            "capacity": 28.0,
                            "penalty_factor": 20.0
                        }
                    }
                }
            ],
            "shared_overrides": {
                "population_size": 30,
                "generations": 20,
                "log_every": 1,
                "mutation_rate": 0.02
            },
            "variant_overrides": {
                "none": {
                    "algorithm": "ga",
                    "algorithm_options": {"adaptive_policy": "none"}
                },
                "repair_only": {
                    "algorithm": "hybrid_ga",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "none",
                        "seed_fraction": 0.0,
                        "repair_strategy": "knapsack_greedy_fill",
                        "local_search_strategy": "none"
                    }
                },
                "repair_rerun_gate": {
                    "__local_baseline__": "knapsack_repair_rerun_gate",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "repair_strategy": "knapsack_greedy_fill",
                        "gate_pilot_generation_fraction": 0.25,
                        "gate_initial_feasible_threshold": 0.05,
                        "gate_first_feasible_generation_limit": 5
                    }
                }
            },
            "sweep": {"study_variant": ["none", "repair_only", "repair_rerun_gate"]},
            "seeds": [1, 2],
            "budget_ceiling": 660,
            "primary_metric": "best_feasible_fitness",
            "plotting": {"history_metric": "best_fitness"},
            "runtime_budget_note": "Tiny knapsack triage smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["plots"]["plot_triage_cost_vs_feasible_gain"])).exists()
    assert Path(str(payload["plots"]["plot_rerun_gate_vs_regret"])).exists()


def test_stress_case_catalog_generation(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "knapsack_stress_catalog.json",
        {
            "study_name": "knapsack_stress_catalog",
            "description": "Tiny knapsack stress catalog smoke.",
            "problem": "knapsack",
            "base_preset": "knapsack_small",
            "cases": [
                {
                    "case_id": "subset_sum_like_small_smoke",
                    "group": "subset_sum_like_small",
                    "note": "tiny subset-sum-like smoke case",
                    "overrides": {
                        "genome_length": 12,
                        "__study_metadata__": {
                            "family_label": "subset_sum_like_small",
                            "capacity_ratio": 0.28,
                            "correlation_note": "smoke"
                        },
                        "problem_options": {
                            "num_items": 12,
                            "instance_name": "subset_sum_like_small_smoke",
                            "instance_source": "test_knapsack_stress_catalog",
                            "weights": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                            "values": [5.0, 6.2, 7.1, 8.0, 9.3, 10.4, 11.2, 12.0, 13.4, 14.1, 15.2, 16.3],
                            "capacity": 42.0,
                            "penalty_factor": 20.0
                        }
                    }
                }
            ],
            "shared_overrides": {
                "population_size": 30,
                "generations": 30,
                "mutation_rate": 0.03,
                "log_every": 1
            },
            "variant_overrides": {
                "greedy_local_search": {
                    "__local_baseline__": "knapsack_greedy_local_search",
                    "algorithm_options": {
                        "adaptive_policy": "none"
                    }
                },
                "none": {
                    "algorithm": "ga",
                    "algorithm_options": {
                        "adaptive_policy": "none"
                    }
                },
                "repair_only": {
                    "algorithm": "hybrid_ga",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "none",
                        "seed_fraction": 0.0,
                        "repair_strategy": "knapsack_greedy_fill",
                        "local_search_strategy": "none"
                    }
                }
            },
            "analysis": {
                "stress_suite": {
                    "default_variant": "greedy_local_search",
                    "baseline_variant": "none",
                    "repair_variant": "repair_only",
                    "budget_band_label": "smoke"
                }
            },
            "sweep": {"study_variant": ["greedy_local_search", "none", "repair_only"]},
            "seeds": [1],
            "budget_ceiling": 900,
            "primary_metric": "best_feasible_fitness",
            "plotting": {"history_metric": "best_fitness"},
            "runtime_budget_note": "Tiny stress catalog smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["stress_case_catalog_csv"])).exists()
    assert Path(str(payload["stress_case_catalog_md"])).exists()
    assert Path(str(payload["tail_risk_summary_csv"])).exists()
    assert Path(str(payload["stress_suite_notes_md"])).exists()
    catalog_rows = _read_csv_rows(Path(str(payload["stress_case_catalog_csv"])))
    assert "compared_profiles" in catalog_rows[0]
    assert "why_selected" in catalog_rows[0]
    assert "future_target_label" in catalog_rows[0]


def test_tsp_stress_suite_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_stress_suite.json",
        {
            "study_name": "tsp_stress_suite_smoke",
            "description": "Tiny TSP stress suite smoke.",
            "problem": "tsp",
            "base_preset": "tsp_small",
            "cases": [
                {
                    "case_id": "bridge_smoke",
                    "group": "rescue_target",
                    "note": "tiny rescue-target case",
                    "overrides": {
                        "genome_length": 10,
                        "problem_options": {
                            "num_cities": 10,
                            "instance_name": "bridge_smoke",
                            "instance_source": "test_tsp_stress_suite",
                            "coordinates": [
                                [0.0, 0.0], [2.0, 3.0], [-2.0, 3.0], [3.0, -2.0], [-3.0, -2.0],
                                [30.0, 0.0], [32.0, 3.0], [28.0, 3.0], [33.0, -2.0], [27.0, -2.0]
                            ]
                        }
                    }
                },
                {
                    "case_id": "corridor_smoke",
                    "group": "anti_case",
                    "note": "tiny corridor case",
                    "overrides": {
                        "genome_length": 10,
                        "problem_options": {
                            "num_cities": 10,
                            "instance_name": "corridor_smoke",
                            "instance_source": "test_tsp_stress_suite",
                            "coordinates": [
                                [0.0, 0.0], [6.0, 2.0], [12.0, -2.0], [18.0, 3.0], [24.0, -3.0],
                                [30.0, 4.0], [36.0, -4.0], [42.0, 5.0], [48.0, -5.0], [24.0, 12.0]
                            ]
                        }
                    }
                }
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {
                    "population_size": 24,
                    "generations": 24,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none"
                    }
                },
                "budget_first": {
                    "population_size": 18,
                    "generations": 20,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none"
                    }
                }
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.25, 0.5, 1.0]
                },
                "stress_suite": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "budget_band_label": "smoke",
                    "catalog_top_overall": 2,
                    "catalog_top_group": 1,
                    "ambiguity_band_pct": 0.5
                }
            },
            "sweep": {"study_variant": ["quality_first", "budget_first"]},
            "seeds": [1, 2],
            "budget_ceiling": 600,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny stress smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["stress_case_catalog_csv"])).exists()
    assert Path(str(payload["tail_risk_summary_csv"])).exists()
    assert Path(str(payload["plots"]["plot_tail_loss_distribution"])).exists()
    assert Path(str(payload["plots"]["plot_case_group_vs_loss"])).exists()
    assert Path(str(payload["plots"]["plot_decision_flip_vs_case_group"])).exists()


def test_zdt1_stress_suite_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_stress_suite.json",
        {
            "study_name": "zdt1_stress_suite_smoke",
            "description": "Tiny ZDT1 stress suite smoke.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {
                "population_size": 24,
                "generations": 28,
                "log_every": 1
            },
            "variant_overrides": {
                "quality_first": {
                    "population_size": 24,
                    "generations": 28,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4
                    }
                },
                "budget_first": {
                    "population_size": 18,
                    "generations": 28,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4
                    }
                }
            },
            "analysis": {
                "qf_tolerance": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "tolerance_bins_pct": [0.25, 0.5, 1.0],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05
                },
                "stress_suite": {
                    "quality_variant": "quality_first",
                    "fast_variant": "budget_first",
                    "budget_band_label": "smoke",
                    "catalog_top_overall": 2,
                    "catalog_top_group": 1,
                    "hv_boundary_pct": 0.5
                }
            },
            "sweep": {"study_variant": ["quality_first", "budget_first"]},
            "seeds": [1, 2, 3],
            "budget_ceiling": 1000,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
            "runtime_budget_note": "Tiny stress smoke."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["stress_case_catalog_csv"])).exists()
    assert Path(str(payload["tail_risk_summary_csv"])).exists()
    assert Path(str(payload["plots"]["plot_hv_tail_distribution"])).exists()
    assert Path(str(payload["plots"]["plot_spread_safety_tail"])).exists()
    assert Path(str(payload["plots"]["plot_case_group_vs_hv_loss"])).exists()


def test_tsp_failure_trace_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_failure_trace_smoke.json",
        {
            "study_name": "tsp_failure_trace_smoke",
            "description": "Tiny TSP failure trace smoke.",
            "problem": "tsp",
            "base_preset": "tsp_small",
            "cases": [
                {
                    "case_id": "bridge_smoke",
                    "group": "rescue_target",
                    "note": "tiny rescue",
                    "overrides": {
                        "genome_length": 10,
                        "problem_options": {
                            "num_cities": 10,
                            "instance_name": "bridge_smoke",
                            "coordinates": [
                                [0.0, 0.0], [4.0, 4.0], [-4.0, 4.0], [5.0, -3.0], [-5.0, -4.0],
                                [40.0, 0.0], [44.0, 4.0], [40.0, -5.0], [20.0, 12.0], [20.0, -12.0]
                            ]
                        }
                    }
                },
                {
                    "case_id": "corridor_smoke",
                    "group": "anti_case",
                    "note": "tiny corridor",
                    "overrides": {
                        "genome_length": 10,
                        "problem_options": {
                            "num_cities": 10,
                            "instance_name": "corridor_smoke",
                            "coordinates": [
                                [0.0, 0.0], [8.0, 3.0], [16.0, -2.0], [24.0, 4.0], [32.0, -3.0],
                                [40.0, 5.0], [48.0, -4.0], [56.0, 6.0], [20.0, 16.0], [44.0, 16.0]
                            ]
                        }
                    }
                }
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {
                    "population_size": 12,
                    "generations": 14,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none"
                    }
                },
                "current_fast": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none"
                    }
                },
                "probe_late_refine": {
                    "population_size": 10,
                    "generations": 12,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none"
                    }
                },
                "probe_pop14_gen8": {
                    "population_size": 14,
                    "generations": 8,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none"
                    }
                },
                "probe_seed25": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.25,
                        "local_search_strategy": "none"
                    }
                }
            },
            "analysis": {
                "failure_trace": {
                    "target_id": "tsp_fast_anti_case_tail",
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                    "reference_variants": ["probe_late_refine", "probe_pop14_gen8", "probe_seed25"],
                    "hypothesis_ids": [
                        "tsp_anticase_late_refinement_deficit",
                        "tsp_anticase_seed_lockin_secondary"
                    ],
                    "hypothesis_probe": {
                        "baseline_variant": "current_fast",
                        "late_refinement_probe_variant": "probe_late_refine",
                        "population_probe_variant": "probe_pop14_gen8",
                        "seed_fraction_probe_variant": "probe_seed25",
                        "focus_case_group": "anti_case",
                        "preserve_case_group": "rescue_target"
                    }
                }
            },
            "sweep": {"study_variant": ["quality_first", "current_fast", "probe_late_refine", "probe_pop14_gen8", "probe_seed25"]},
            "seeds": [1],
            "budget_ceiling": 168,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"}
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["failure_trace_table_csv"])).exists()
    assert Path(str(payload["failure_hypotheses_csv"])).exists()
    assert Path(str(payload["plots"]["plot_trace_distance_vs_diversity"])).exists()
    assert Path(str(payload["plots"]["plot_late_refinement_gap"])).exists()
    collapse_plot = payload["plots"].get("plot_collapse_onset_vs_last_improvement")
    if collapse_plot is not None:
        assert Path(str(collapse_plot)).exists()
    assert Path(str(payload["plots"]["plot_anticase_vs_rescue_overlay"])).exists()


def test_zdt1_failure_trace_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_failure_trace_smoke.json",
        {
            "study_name": "zdt1_failure_trace_smoke",
            "description": "Tiny ZDT1 failure trace smoke.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {"log_every": 1, "population_size": 16, "generations": 12},
            "variant_overrides": {
                "quality_first": {
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4
                    }
                },
                "current_fast": {
                    "population_size": 14,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3
                    }
                },
                "probe_refresh012": {
                    "population_size": 14,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.12,
                        "adaptation_cooldown": 3
                    }
                },
                "probe_cooldown2": {
                    "population_size": 14,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 2
                    }
                },
                "probe_pop16_gen10": {
                    "population_size": 16,
                    "generations": 10,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3
                    }
                }
            },
            "analysis": {
                "failure_trace": {
                    "target_id": "zdt1_fast_spread_safety_fail",
                    "secondary_target_id": "zdt1_fast_joint_safety_fail",
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                    "reference_variants": ["probe_refresh012", "probe_cooldown2", "probe_pop16_gen10"],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                    "hv_plateau_slope_threshold": 0.01,
                    "hypothesis_ids": [
                        "zdt1_fast_refresh_timing_mismatch",
                        "zdt1_fast_spread_mechanism_still_unisolated"
                    ],
                    "hypothesis_probe": {
                        "baseline_variant": "current_fast",
                        "refresh_probe_variant": "probe_refresh012",
                        "cooldown_probe_variant": "probe_cooldown2",
                        "population_probe_variant": "probe_pop16_gen10"
                    }
                }
            },
            "sweep": {"study_variant": ["quality_first", "current_fast", "probe_refresh012", "probe_cooldown2", "probe_pop16_gen10"]},
            "seeds": [1, 2],
            "budget_ceiling": 400,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"}
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["failure_trace_table_csv"])).exists()
    assert Path(str(payload["failure_hypotheses_csv"])).exists()
    assert Path(str(payload["plots"]["plot_trace_hv_vs_spread"])).exists()
    assert Path(str(payload["plots"]["plot_safety_fail_onset"])).exists()
    assert Path(str(payload["plots"]["plot_safe_vs_fail_overlay"])).exists()


def test_tsp_population_generation_probe_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_population_generation_probe_smoke.json",
        {
            "study_name": "tsp_population_generation_probe_smoke",
            "description": "Tiny TSP population-generation probe smoke.",
            "problem": "tsp",
            "base_preset": "tsp_small",
            "cases": [
                {
                    "case_id": "bridge_smoke",
                    "group": "rescue_target",
                    "note": "tiny rescue",
                    "overrides": {
                        "genome_length": 10,
                        "problem_options": {
                            "num_cities": 10,
                            "instance_name": "bridge_smoke",
                            "coordinates": [
                                [0.0, 0.0],
                                [4.0, 4.0],
                                [-4.0, 4.0],
                                [5.0, -3.0],
                                [-5.0, -4.0],
                                [40.0, 0.0],
                                [44.0, 4.0],
                                [40.0, -5.0],
                                [20.0, 12.0],
                                [20.0, -12.0],
                            ],
                        },
                    },
                },
                {
                    "case_id": "corridor_smoke",
                    "group": "anti_case",
                    "note": "tiny corridor",
                    "overrides": {
                        "genome_length": 10,
                        "problem_options": {
                            "num_cities": 10,
                            "instance_name": "corridor_smoke",
                            "coordinates": [
                                [0.0, 0.0],
                                [8.0, 3.0],
                                [16.0, -2.0],
                                [24.0, 4.0],
                                [32.0, -3.0],
                                [40.0, 5.0],
                                [48.0, -4.0],
                                [56.0, 6.0],
                                [20.0, 16.0],
                                [44.0, 16.0],
                            ],
                        },
                    },
                },
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {
                    "population_size": 12,
                    "generations": 14,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "current_fast": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "probe_pg_gen38_pop35": {
                    "population_size": 10,
                    "generations": 12,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "probe_pg_gen30_pop44": {
                    "population_size": 14,
                    "generations": 8,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "probe_seed_fraction025": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.25,
                        "local_search_strategy": "none",
                    },
                },
            },
            "analysis": {
                "failure_trace": {
                    "target_id": "tsp_fast_anti_case_tail",
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                    "reference_variants": [
                        "probe_pg_gen38_pop35",
                        "probe_pg_gen30_pop44",
                        "probe_seed_fraction025",
                    ],
                    "hypothesis_ids": [
                        "tsp_anticase_late_refinement_deficit",
                        "tsp_anticase_seed_lockin_secondary",
                    ],
                    "hypothesis_probe": {
                        "baseline_variant": "current_fast",
                        "late_refinement_probe_variant": "probe_pg_gen38_pop35",
                        "population_probe_variant": "probe_pg_gen30_pop44",
                        "seed_fraction_probe_variant": "probe_seed_fraction025",
                        "focus_case_group": "anti_case",
                        "preserve_case_group": "rescue_target",
                    },
                }
            },
            "sweep": {
                "study_variant": [
                    "quality_first",
                    "current_fast",
                    "probe_pg_gen38_pop35",
                    "probe_pg_gen30_pop44",
                    "probe_seed_fraction025",
                ]
            },
            "seeds": [1, 2],
            "budget_ceiling": 168,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["failure_trace_table_csv"])).exists()
    assert Path(str(payload["failure_hypotheses_csv"])).exists()
    assert Path(str(payload["plots"]["plot_late_refinement_gap"])).exists()
    assert Path(str(payload["plots"]["plot_population_generation_tradeoff_vs_tail"])).exists()
    assert Path(str(payload["plots"]["plot_anticase_vs_rescue_overlay"])).exists()


def test_zdt1_timing_vs_pg_probe_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_timing_vs_pg_probe_smoke.json",
        {
            "study_name": "zdt1_timing_vs_pg_probe_smoke",
            "description": "Tiny ZDT1 timing-vs-population-generation probe smoke.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {"log_every": 1, "population_size": 16, "generations": 12},
            "variant_overrides": {
                "quality_first": {
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4,
                    }
                },
                "current_fast": {
                    "population_size": 14,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3,
                    },
                },
                "probe_timing_cooldown2": {
                    "population_size": 14,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 2,
                    },
                },
                "probe_pg_pop12_gen14": {
                    "population_size": 12,
                    "generations": 14,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3,
                    },
                },
            },
            "analysis": {
                "failure_trace": {
                    "target_id": "zdt1_fast_spread_safety_fail",
                    "secondary_target_id": "zdt1_fast_joint_safety_fail",
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                    "reference_variants": ["probe_timing_cooldown2", "probe_pg_pop12_gen14"],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                    "hv_plateau_slope_threshold": 0.01,
                    "hypothesis_ids": [
                        "zdt1_fast_refresh_timing_mismatch_joint",
                        "zdt1_fast_spread_mechanism_population_generation_candidate",
                        "zdt1_fast_spread_mechanism_still_unisolated",
                    ],
                    "hypothesis_probe": {
                        "baseline_variant": "current_fast",
                        "cooldown_probe_variant": "probe_timing_cooldown2",
                        "population_generation_probe_variants": ["probe_pg_pop12_gen14"],
                    },
                }
            },
            "sweep": {
                "study_variant": [
                    "quality_first",
                    "current_fast",
                    "probe_timing_cooldown2",
                    "probe_pg_pop12_gen14",
                ]
            },
            "seeds": [1, 2],
            "budget_ceiling": 420,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["failure_trace_table_csv"])).exists()
    assert Path(str(payload["failure_hypotheses_csv"])).exists()
    assert Path(str(payload["plots"]["plot_trace_hv_vs_spread"])).exists()
    assert Path(str(payload["plots"]["plot_safety_fail_onset"])).exists()
    assert Path(str(payload["plots"]["plot_spread_vs_joint_fail_split"])).exists()


def test_tsp_extreme_tail_pg_contour_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "tsp_extreme_tail_pg_contour_smoke.json",
        {
            "study_name": "tsp_extreme_tail_pg_contour_smoke",
            "description": "Tiny TSP extreme-tail contour closeout smoke.",
            "problem": "tsp",
            "base_preset": "tsp_small",
            "cases": [
                {
                    "case_id": "bridge_smoke",
                    "group": "rescue_target",
                    "note": "tiny rescue",
                    "overrides": {
                        "genome_length": 10,
                        "problem_options": {
                            "num_cities": 10,
                            "instance_name": "bridge_smoke",
                            "coordinates": [
                                [0.0, 0.0],
                                [4.0, 4.0],
                                [-4.0, 4.0],
                                [5.0, -3.0],
                                [-5.0, -4.0],
                                [40.0, 0.0],
                                [44.0, 4.0],
                                [40.0, -5.0],
                                [20.0, 12.0],
                                [20.0, -12.0],
                            ],
                        },
                    },
                },
                {
                    "case_id": "corridor_smoke",
                    "group": "anti_case",
                    "note": "tiny corridor",
                    "overrides": {
                        "genome_length": 10,
                        "problem_options": {
                            "num_cities": 10,
                            "instance_name": "corridor_smoke",
                            "coordinates": [
                                [0.0, 0.0],
                                [8.0, 3.0],
                                [16.0, -2.0],
                                [24.0, 4.0],
                                [32.0, -3.0],
                                [40.0, 5.0],
                                [48.0, -4.0],
                                [56.0, 6.0],
                                [20.0, 16.0],
                                [44.0, 16.0],
                            ],
                        },
                    },
                },
            ],
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "quality_first": {
                    "population_size": 12,
                    "generations": 14,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "current_fast": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "contour_pg_gen11_pop11": {
                    "population_size": 11,
                    "generations": 11,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "contour_pg_gen12_pop10": {
                    "population_size": 10,
                    "generations": 12,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "seed_secondary_fraction025": {
                    "population_size": 12,
                    "generations": 10,
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.25,
                        "local_search_strategy": "none",
                    },
                },
            },
            "analysis": {
                "tsp_fast_tail": {
                    "quality_variant": "quality_first",
                    "baseline_fast_variant": "current_fast",
                },
                "failure_trace": {
                    "target_id": "tsp_fast_anti_case_tail",
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                    "reference_variants": [
                        "contour_pg_gen11_pop11",
                        "contour_pg_gen12_pop10",
                        "seed_secondary_fraction025",
                    ],
                    "hypothesis_ids": [
                        "tsp_anticase_late_refinement_deficit",
                        "tsp_anticase_seed_lockin_secondary",
                    ],
                    "hypothesis_probe": {
                        "baseline_variant": "current_fast",
                        "late_refinement_probe_variant": "contour_pg_gen11_pop11",
                        "population_probe_variant": "contour_pg_gen12_pop10",
                        "contour_probe_variants": [
                            "contour_pg_gen11_pop11",
                            "contour_pg_gen12_pop10",
                        ],
                        "seed_fraction_probe_variant": "seed_secondary_fraction025",
                        "focus_case_group": "anti_case",
                        "preserve_case_group": "rescue_target",
                        "tail_focus": "p95_max",
                    },
                },
            },
            "sweep": {
                "study_variant": [
                    "quality_first",
                    "current_fast",
                    "contour_pg_gen11_pop11",
                    "contour_pg_gen12_pop10",
                    "seed_secondary_fraction025",
                ]
            },
            "seeds": [1, 2],
            "budget_ceiling": 168,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["failure_trace_table_csv"])).exists()
    assert Path(str(payload["failure_hypotheses_csv"])).exists()
    assert Path(str(payload["tsp_fast_tail_summary_csv"])).exists()
    assert Path(str(payload["plots"]["plot_late_refinement_gap"])).exists()
    assert Path(str(payload["plots"]["plot_population_generation_tradeoff_vs_tail"])).exists()
    assert Path(str(payload["plots"]["plot_anticase_vs_rescue_overlay"])).exists()
    assert Path(str(payload["plots"]["plot_anticase_p95_max_reduction"])).exists()


def test_zdt1_spread_pg_probe_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_spread_pg_probe_smoke.json",
        {
            "study_name": "zdt1_spread_pg_probe_smoke",
            "description": "Tiny ZDT1 spread-target PG probe smoke.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {"log_every": 1, "population_size": 16, "generations": 12},
            "variant_overrides": {
                "quality_first": {
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4,
                    }
                },
                "current_fast": {
                    "population_size": 14,
                    "generations": 12,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3,
                    },
                },
                "spread_pg_pop12_gen14": {
                    "population_size": 12,
                    "generations": 14,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3,
                    },
                },
                "spread_pg_pop15_gen11": {
                    "population_size": 15,
                    "generations": 11,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3,
                    },
                },
            },
            "analysis": {
                "zdt1_fast_hardening": {
                    "quality_variant": "quality_first",
                    "baseline_fast_variant": "current_fast",
                    "candidate_variants": [
                        "current_fast",
                        "spread_pg_pop12_gen14",
                        "spread_pg_pop15_gen11",
                    ],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                },
                "failure_trace": {
                    "target_id": "zdt1_fast_spread_safety_fail",
                    "secondary_target_id": "zdt1_fast_joint_safety_fail",
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                    "reference_variants": ["spread_pg_pop12_gen14", "spread_pg_pop15_gen11"],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                    "hv_plateau_slope_threshold": 0.01,
                    "hypothesis_ids": [
                        "zdt1_fast_joint_timing_mismatch_still_plausible",
                        "zdt1_fast_spread_mechanism_population_generation_candidate",
                        "zdt1_fast_spread_mechanism_still_unisolated",
                    ],
                    "hypothesis_probe": {
                        "baseline_variant": "current_fast",
                        "population_generation_probe_variants": [
                            "spread_pg_pop12_gen14",
                            "spread_pg_pop15_gen11",
                        ],
                        "probe_mode": "spread_only",
                    },
                },
            },
            "sweep": {
                "study_variant": [
                    "quality_first",
                    "current_fast",
                    "spread_pg_pop12_gen14",
                    "spread_pg_pop15_gen11",
                ]
            },
            "seeds": [1, 2],
            "budget_ceiling": 420,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["failure_trace_table_csv"])).exists()
    assert Path(str(payload["failure_hypotheses_csv"])).exists()
    assert Path(str(payload["zdt1_fast_hardening_summary_csv"])).exists()
    assert Path(str(payload["plots"]["plot_trace_hv_vs_spread"])).exists()
    assert Path(str(payload["plots"]["plot_spread_vs_joint_fail_split"])).exists()
    assert Path(str(payload["plots"]["plot_population_generation_vs_spread_tail"])).exists()
    summary_rows = _read_csv_rows(Path(str(payload["zdt1_fast_hardening_summary_csv"])))
    assert any(row["spread_delta_p95"] != "" for row in summary_rows)
    assert any(row["spread_tail_reduction_score_p95"] != "" for row in summary_rows)


def test_zdt1_joint_timing_probe_smoke(tmp_path: Path) -> None:
    manifest = _write_study_manifest(
        tmp_path,
        "zdt1_joint_timing_probe_smoke.json",
        {
            "study_name": "zdt1_joint_timing_probe_smoke",
            "description": "Tiny ZDT1 joint-target timing probe smoke.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {"log_every": 1, "population_size": 16, "generations": 12},
            "variant_overrides": {
                "quality_first": {
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 4,
                    }
                },
                "current_fast": {
                    "population_size": 14,
                    "generations": 12,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 3,
                    },
                },
                "joint_timing_refresh012": {
                    "population_size": 14,
                    "generations": 12,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.12,
                        "adaptation_cooldown": 3,
                    },
                },
                "joint_timing_cooldown2": {
                    "population_size": 14,
                    "generations": 12,
                    "algorithm_options": {
                        "adaptive_policy": "low_diversity_injection",
                        "diversity_threshold": 0.55,
                        "refresh_fraction": 0.1,
                        "adaptation_cooldown": 2,
                    },
                },
            },
            "analysis": {
                "zdt1_fast_hardening": {
                    "quality_variant": "quality_first",
                    "baseline_fast_variant": "current_fast",
                    "candidate_variants": [
                        "current_fast",
                        "joint_timing_refresh012",
                        "joint_timing_cooldown2",
                    ],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                },
                "failure_trace": {
                    "target_id": "zdt1_fast_spread_safety_fail",
                    "secondary_target_id": "zdt1_fast_joint_safety_fail",
                    "quality_variant": "quality_first",
                    "fast_variant": "current_fast",
                    "reference_variants": [
                        "joint_timing_refresh012",
                        "joint_timing_cooldown2",
                    ],
                    "pareto_ratio_drop_threshold": 0.01,
                    "spread_degradation_threshold": 0.05,
                    "hv_plateau_slope_threshold": 0.01,
                    "hypothesis_ids": [
                        "zdt1_fast_joint_timing_mismatch_still_plausible",
                        "zdt1_fast_spread_mechanism_population_generation_candidate",
                        "zdt1_fast_joint_mechanism_still_unisolated",
                    ],
                    "hypothesis_probe": {
                        "baseline_variant": "current_fast",
                        "refresh_probe_variant": "joint_timing_refresh012",
                        "cooldown_probe_variant": "joint_timing_cooldown2",
                        "probe_mode": "joint_only",
                        "joint_unisolated_hypothesis_id": (
                            "zdt1_fast_joint_mechanism_still_unisolated"
                        ),
                    },
                },
            },
            "sweep": {
                "study_variant": [
                    "quality_first",
                    "current_fast",
                    "joint_timing_refresh012",
                    "joint_timing_cooldown2",
                ]
            },
            "seeds": [1, 2],
            "budget_ceiling": 420,
            "primary_metric": "hypervolume",
            "plotting": {"history_metric": "hypervolume"},
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["failure_trace_table_csv"])).exists()
    assert Path(str(payload["failure_hypotheses_csv"])).exists()
    assert Path(str(payload["zdt1_fast_hardening_summary_csv"])).exists()
    assert Path(str(payload["plots"]["plot_trace_hv_vs_spread"])).exists()
    assert Path(str(payload["plots"]["plot_safety_fail_onset"])).exists()
    assert Path(str(payload["plots"]["plot_timing_vs_joint_fail_rate"])).exists()


def test_stress_refresh_registry_generation(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    tsp_dir = outputs_root / "20260417T000000Z_tsp_stress_refresh_suite"
    zdt1_dir = outputs_root / "20260417T000100Z_zdt1_stress_refresh_suite"
    tsp_dir.mkdir(parents=True)
    zdt1_dir.mkdir(parents=True)

    with (tsp_dir / "stress_case_catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "problem",
                "case_id",
                "instance_label",
                "seed",
                "budget_band",
                "profile_compared",
                "compared_profiles",
                "regret_or_loss",
                "case_group",
                "why_selected_as_stress_case",
                "why_selected",
                "future_target_label",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "problem": "tsp",
                "case_id": "corridor_case",
                "instance_label": "corridor_case",
                "seed": "9054",
                "budget_band": "current_hardened_qf_split",
                "profile_compared": "budget_first_vs_quality_first",
                "compared_profiles": "budget_first_vs_quality_first",
                "regret_or_loss": "2.8",
                "case_group": "anti_case",
                "why_selected_as_stress_case": "anti_case_tail",
                "why_selected": "anti_case_tail",
                "future_target_label": "tsp_fast_anti_case_tail",
            }
        )
    with (tsp_dir / "tail_risk_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scope", "case_group", "study_variant", "mean_loss_pct", "p90_loss_pct", "max_loss_pct"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scope": "overall",
                "case_group": "overall",
                "study_variant": "budget_first",
                "mean_loss_pct": "0.6",
                "p90_loss_pct": "2.5",
                "max_loss_pct": "3.0",
            }
        )
    with (zdt1_dir / "stress_case_catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "problem",
                "case_id",
                "instance_label",
                "seed",
                "budget_band",
                "profile_compared",
                "compared_profiles",
                "regret_or_loss",
                "case_group",
                "why_selected_as_stress_case",
                "why_selected",
                "joint_safety_fail",
                "future_target_label",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "problem": "zdt1",
                "case_id": "zdt1_seed_8218",
                "instance_label": "zdt1_seed_8218",
                "seed": "8218",
                "budget_band": "current_hardened_qf_split",
                "profile_compared": "budget_first_vs_quality_first",
                "compared_profiles": "budget_first_vs_quality_first",
                "regret_or_loss": "0.47",
                "case_group": "safety_fail",
                "why_selected_as_stress_case": "joint_safety_fail",
                "why_selected": "joint_safety_fail",
                "joint_safety_fail": "1.0",
                "future_target_label": "zdt1_fast_joint_safety_fail",
            }
        )
    with (zdt1_dir / "tail_risk_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "case_group",
                "study_variant",
                "mean_loss_pct",
                "p90_loss_pct",
                "max_loss_pct",
                "joint_safety_fail_rate",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scope": "overall",
                "case_group": "overall",
                "study_variant": "budget_first",
                "mean_loss_pct": "0.09",
                "p90_loss_pct": "0.31",
                "max_loss_pct": "0.47",
                "joint_safety_fail_rate": "0.2",
            }
        )

    payload = build_stress_refresh_registry([tsp_dir, zdt1_dir], outputs_root)

    assert Path(str(payload["current_stress_case_catalog_csv"])).exists()
    assert Path(str(payload["future_optimization_targets_csv"])).exists()
    target_rows = _read_csv_rows(Path(str(payload["future_optimization_targets_csv"])))
    assert {row["target_id"] for row in target_rows} >= {
        "tsp_fast_anti_case_tail",
        "zdt1_fast_joint_safety_fail",
    }
    notes = Path(str(payload["stress_refresh_notes_md"])).read_text(encoding="utf-8")
    assert "TSP" in notes
    assert "ZDT1" in notes


def test_stress_target_reduction_registry_generation(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    tsp_dir = outputs_root / "20260417T000000Z_tsp_stress_target_reduction_confirm"
    zdt1_dir = outputs_root / "20260417T000100Z_zdt1_stress_target_reduction_confirm"
    tsp_dir.mkdir(parents=True)
    zdt1_dir.mkdir(parents=True)

    _write_json_file(
        outputs_root,
        "future_optimization_targets.json",
        [
            {
                "target_id": "tsp_fast_anti_case_tail",
                "problem": "tsp",
                "current_default_affected": "tsp_seeded_swap_local_fast",
                "failure_metric": "anti_case_p95_loss_pct",
                "case_group": "anti_case",
                "severity": "high",
                "estimated_optimization_value": "high",
                "keep_as_regression_case": True,
                "suggested_next_pass": "tail reduction",
                "notes": "previous",
            },
            {
                "target_id": "tsp_rescue_target_ambiguity",
                "problem": "tsp",
                "current_default_affected": "tsp_seeded_swap_local_fast",
                "failure_metric": "rescue_target_mean_loss_pct",
                "case_group": "rescue_target",
                "severity": "medium",
                "estimated_optimization_value": "medium",
                "keep_as_regression_case": True,
                "suggested_next_pass": "ambiguity check",
                "notes": "previous",
            },
            {
                "target_id": "zdt1_fast_spread_safety_fail",
                "problem": "zdt1",
                "current_default_affected": "zdt1_diversity_injection_fast",
                "failure_metric": "spread_fail_rate",
                "case_group": "safety_fail",
                "severity": "high",
                "estimated_optimization_value": "high",
                "keep_as_regression_case": True,
                "suggested_next_pass": "safety reduction",
                "notes": "previous",
            },
            {
                "target_id": "zdt1_fast_joint_safety_fail",
                "problem": "zdt1",
                "current_default_affected": "zdt1_diversity_injection_fast",
                "failure_metric": "joint_safety_fail_rate",
                "case_group": "safety_fail",
                "severity": "high",
                "estimated_optimization_value": "high",
                "keep_as_regression_case": True,
                "suggested_next_pass": "safety reduction",
                "notes": "previous",
            },
            {
                "target_id": "knapsack_repair_boundary_subset_sum_tight_capacity",
                "problem": "knapsack",
                "current_default_affected": "knapsack_repair_local_experimental",
                "failure_metric": "best_feasible_fitness_regret",
                "case_group": "borderline",
                "severity": "medium",
                "estimated_optimization_value": "medium",
                "keep_as_regression_case": True,
                "suggested_next_pass": "narrow sanity",
                "notes": "previous",
            },
            {
                "target_id": "onemax_no_active_target",
                "problem": "onemax",
                "current_default_affected": "none",
                "failure_metric": "control_stability",
                "case_group": "control",
                "severity": "low",
                "estimated_optimization_value": "low",
                "keep_as_regression_case": False,
                "suggested_next_pass": "none",
                "notes": "previous",
            },
        ],
    )

    _write_rows_csv(
        tsp_dir / "tail_risk_reduction_summary.csv",
        [
            {
                "scope": "overall",
                "study_variant": "current_fast",
                "mean_loss_pct": "0.63",
                "anti_case_p90_loss_pct": "2.60",
                "anti_case_p95_loss_pct": "2.95",
                "anti_case_max_loss_pct": "3.10",
                "rescue_target_mean_loss_pct": "0.58",
            },
            {
                "scope": "overall",
                "study_variant": "micro_inversion_pop40_gen33",
                "mean_loss_pct": "0.37",
                "anti_case_p90_loss_pct": "2.52",
                "anti_case_p95_loss_pct": "2.88",
                "anti_case_max_loss_pct": "2.84",
                "rescue_target_mean_loss_pct": "0.53",
            },
        ],
        [
            "scope",
            "study_variant",
            "mean_loss_pct",
            "anti_case_p90_loss_pct",
            "anti_case_p95_loss_pct",
            "anti_case_max_loss_pct",
            "rescue_target_mean_loss_pct",
        ],
    )
    _write_rows_csv(
        zdt1_dir / "tail_risk_reduction_summary.csv",
        [
            {
                "scope": "overall",
                "study_variant": "current_fast",
                "mean_loss_pct": "0.09",
                "p90_loss_pct": "0.31",
                "spread_fail_rate": "0.20",
                "joint_safety_fail_rate": "0.20",
            },
            {
                "scope": "overall",
                "study_variant": "micro_refresh012_cooldown3",
                "mean_loss_pct": "0.10",
                "p90_loss_pct": "0.30",
                "spread_fail_rate": "0.30",
                "joint_safety_fail_rate": "0.30",
            },
        ],
        [
            "scope",
            "study_variant",
            "mean_loss_pct",
            "p90_loss_pct",
            "spread_fail_rate",
            "joint_safety_fail_rate",
        ],
    )

    payload = build_stress_target_reduction_registry(
        previous_registry_path=outputs_root / "future_optimization_targets.json",
        output_dir=outputs_root,
        tsp_study_dir=tsp_dir,
        zdt1_study_dir=zdt1_dir,
        include_knapsack_freeze=True,
        include_onemax_freeze=True,
    )

    assert Path(str(payload["future_optimization_targets_json"])).exists()
    assert Path(str(payload["future_optimization_targets_csv"])).exists()
    assert Path(str(payload["future_optimization_targets_md"])).exists()
    assert Path(str(payload["stress_reduction_notes_md"])).exists()

    rows = _read_csv_rows(Path(str(payload["future_optimization_targets_csv"])))
    tsp_row = next(row for row in rows if row["target_id"] == "tsp_fast_anti_case_tail")
    zdt1_row = next(row for row in rows if row["target_id"] == "zdt1_fast_spread_safety_fail")

    assert tsp_row["changed_or_not"] == "unchanged_keep_default"
    assert zdt1_row["recommended_next_action"] == "keep_current_fast_and_leave_final_safety_on_q"

    notes = Path(str(payload["stress_reduction_notes_md"])).read_text(encoding="utf-8")
    assert "TSP" in notes
    assert "ZDT1" in notes


def test_failure_hypothesis_registry_generation(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    tsp_dir = outputs_root / "20260418T000000Z_tsp_tail_freeze_recheck"
    zdt1_spread_dir = outputs_root / "20260418T000100Z_zdt1_spread_candidate_boundary_confirm"
    zdt1_joint_dir = outputs_root / "20260418T000200Z_zdt1_joint_note_freeze_check"
    tsp_dir.mkdir(parents=True)
    zdt1_spread_dir.mkdir(parents=True)
    zdt1_joint_dir.mkdir(parents=True)
    (tsp_dir / "tsp_irreducible_tail_freeze_summary.md").write_text(
        "# TSP Tail Freeze Summary\n\nTreat the remaining anti-case p95/max tail as a protocol limitation.\n",
        encoding="utf-8",
    )
    (tsp_dir / "tsp_protocol_limitation_freeze_summary.md").write_text(
        "# TSP Protocol Limitation Freeze Summary\n\nTreat the remaining anti-case p95/max tail as a protocol limitation.\n",
        encoding="utf-8",
    )

    _write_json_file(
        outputs_root,
        "future_optimization_targets.json",
        [
            {
                "target_id": "tsp_fast_anti_case_tail",
                "problem": "tsp",
                "current_default_affected": "configs/local_profiles/tsp_seeded_swap_local_fast.json",
                "failure_metric": "route_distance_loss_pct",
                "case_group": "anti_case",
                "severity": "high",
                "estimated_optimization_value": "high",
                "keep_as_regression_case": True,
                "suggested_next_pass": "trace diagnosis",
                "notes": "previous"
            },
            {
                "target_id": "tsp_rescue_target_ambiguity",
                "problem": "tsp",
                "current_default_affected": "configs/local_profiles/tsp_seeded_swap_local_fast.json",
                "failure_metric": "route_distance_loss_pct",
                "case_group": "rescue_target",
                "severity": "medium",
                "estimated_optimization_value": "medium",
                "keep_as_regression_case": True,
                "suggested_next_pass": "trace diagnosis",
                "notes": "previous"
            },
            {
                "target_id": "zdt1_fast_spread_safety_fail",
                "problem": "zdt1",
                "current_default_affected": "configs/local_profiles/zdt1_diversity_injection_fast.json",
                "failure_metric": "spread_delta",
                "case_group": "safety_fail",
                "severity": "high",
                "estimated_optimization_value": "high",
                "keep_as_regression_case": True,
                "suggested_next_pass": "trace diagnosis",
                "notes": "previous"
            },
            {
                "target_id": "zdt1_fast_joint_safety_fail",
                "problem": "zdt1",
                "current_default_affected": "configs/local_profiles/zdt1_diversity_injection_fast.json",
                "failure_metric": "joint_safety_fail",
                "case_group": "safety_fail",
                "severity": "high",
                "estimated_optimization_value": "high",
                "keep_as_regression_case": True,
                "suggested_next_pass": "trace diagnosis",
                "notes": "previous"
            },
            {
                "target_id": "knapsack_repair_boundary_subset_sum_tight_capacity",
                "problem": "knapsack",
                "current_default_affected": "configs/local_profiles/knapsack_repair_local_experimental.json",
                "failure_metric": "best_feasible_fitness_regret",
                "case_group": "borderline",
                "severity": "medium",
                "estimated_optimization_value": "medium",
                "keep_as_regression_case": True,
                "suggested_next_pass": "keep note",
                "notes": "previous"
            },
            {
                "target_id": "onemax_no_active_target",
                "problem": "onemax",
                "current_default_affected": "none",
                "failure_metric": "control_stability",
                "case_group": "control",
                "severity": "low",
                "estimated_optimization_value": "low",
                "keep_as_regression_case": False,
                "suggested_next_pass": "none",
                "notes": "previous"
            }
        ],
    )

    _write_rows_csv(
        tsp_dir / "failure_trace_table.csv",
        [
            {
                "target_id": "tsp_fast_anti_case_tail",
                "case_group": "anti_case",
                "study_variant": "current_fast",
                "route_distance_loss_pct_vs_quality": "2.9",
                "seed_lockin_signal": "1.0",
                "late_refinement_deficit_signal": "1.0",
                "joint_safety_fail": "0.0"
            }
        ],
        [
            "target_id",
            "case_group",
            "study_variant",
            "route_distance_loss_pct_vs_quality",
            "seed_lockin_signal",
            "late_refinement_deficit_signal",
            "joint_safety_fail",
        ],
    )
    _write_rows_csv(
        tsp_dir / "failure_hypotheses.csv",
        [
            {
                "hypothesis_id": "tsp_anticase_late_refinement_deficit",
                "target_id": "tsp_fast_anti_case_tail",
                "problem": "tsp",
                "affected_default": "configs/local_profiles/tsp_seeded_swap_local_fast.json",
                "evidence_rows": "4",
                "suspected_mechanism": "late_refinement_deficit",
                "supporting_trace_signals": "late_refinement_deficit_signal_rate=0.80",
                "confidence": "high",
                "current_evidence_strength": "high",
                "probe_attempted": "probe_late_refine_pop35_gen38",
                "confirmed_or_weakened": "strengthened",
                "recommended_next_probe": "investigate_population_generation_tradeoff",
                "recommended_next_action": "investigate_population_generation_tradeoff",
                "keep_as_regression_case": "True",
                "notes": "dominant"
            }
        ],
        [
            "hypothesis_id",
            "target_id",
            "problem",
            "affected_default",
            "evidence_rows",
            "suspected_mechanism",
            "supporting_trace_signals",
            "confidence",
            "current_evidence_strength",
            "probe_attempted",
            "confirmed_or_weakened",
            "recommended_next_probe",
            "recommended_next_action",
            "keep_as_regression_case",
            "notes",
        ],
    )
    _write_rows_csv(
        zdt1_spread_dir / "zdt1_spread_candidate_validation_summary.csv",
        [
            {
                "scope": "overall",
                "slice": "",
                "study_variant": "current_fast",
                "spread_fail_rate": "0.20",
                "joint_safety_fail_rate": "0.20",
                "p90_loss_pct": "0.31",
                "spread_delta_p95": "0.0916",
                "pareto_ratio_delta_mean": "0.0000",
            },
            {
                "scope": "overall",
                "slice": "",
                "study_variant": "spread_pg_pop41_gen88",
                "spread_fail_rate": "0.20",
                "joint_safety_fail_rate": "0.20",
                "p90_loss_pct": "0.24",
                "spread_delta_p95": "0.0791",
                "pareto_ratio_delta_mean": "0.0000",
            },
            {
                "scope": "slice",
                "slice": "spread_stress",
                "study_variant": "current_fast",
                "spread_fail_rate": "0.40",
                "joint_safety_fail_rate": "0.40",
                "p90_loss_pct": "0.33",
                "spread_delta_p95": "0.1086",
                "pareto_ratio_delta_mean": "0.0000",
            },
            {
                "scope": "slice",
                "slice": "spread_stress",
                "study_variant": "spread_pg_pop41_gen88",
                "spread_fail_rate": "0.00",
                "joint_safety_fail_rate": "0.00",
                "p90_loss_pct": "0.24",
                "spread_delta_p95": "0.0208",
                "pareto_ratio_delta_mean": "0.0000",
            },
            {
                "scope": "slice",
                "slice": "stable_contrast",
                "study_variant": "current_fast",
                "spread_fail_rate": "0.00",
                "joint_safety_fail_rate": "0.00",
                "p90_loss_pct": "0.10",
                "spread_delta_p95": "0.0150",
                "pareto_ratio_delta_mean": "0.0000",
            },
            {
                "scope": "slice",
                "slice": "stable_contrast",
                "study_variant": "spread_pg_pop41_gen88",
                "spread_fail_rate": "0.50",
                "joint_safety_fail_rate": "0.50",
                "p90_loss_pct": "0.20",
                "spread_delta_p95": "0.0600",
                "pareto_ratio_delta_mean": "0.0000",
            },
            {
                "scope": "slice",
                "slice": "normal_holdout",
                "study_variant": "current_fast",
                "spread_fail_rate": "0.00",
                "joint_safety_fail_rate": "0.00",
                "p90_loss_pct": "0.12",
                "spread_delta_p95": "0.0200",
                "pareto_ratio_delta_mean": "0.0000",
            },
            {
                "scope": "slice",
                "slice": "normal_holdout",
                "study_variant": "spread_pg_pop41_gen88",
                "spread_fail_rate": "0.33",
                "joint_safety_fail_rate": "0.33",
                "p90_loss_pct": "0.30",
                "spread_delta_p95": "0.0800",
                "pareto_ratio_delta_mean": "0.0000",
            },
        ],
        [
            "scope",
            "slice",
            "study_variant",
            "spread_fail_rate",
            "joint_safety_fail_rate",
            "p90_loss_pct",
            "spread_delta_p95",
            "pareto_ratio_delta_mean",
        ],
    )
    _write_rows_csv(
        zdt1_spread_dir / "zdt1_spread_candidate_boundary_table.csv",
        [
            {
                "scope": "overall",
                "slice": "overall",
                "study_variant": "current_fast",
                "spread_fail_rate": "0.20",
                "joint_safety_fail_rate": "0.20",
                "p90_loss_pct": "0.31",
                "spread_delta_p95": "0.0916",
                "pareto_ratio_delta_mean": "0.0000",
                "boundary_decision": "note_only_stress_slice",
            },
            {
                "scope": "overall",
                "slice": "overall",
                "study_variant": "spread_pg_pop41_gen88",
                "spread_fail_rate": "0.20",
                "joint_safety_fail_rate": "0.20",
                "p90_loss_pct": "0.24",
                "spread_delta_p95": "0.0791",
                "pareto_ratio_delta_mean": "0.0000",
                "boundary_decision": "note_only_stress_slice",
            },
            {
                "scope": "slice",
                "slice": "spread_stress",
                "study_variant": "current_fast",
                "spread_fail_rate": "0.40",
                "joint_safety_fail_rate": "0.40",
                "p90_loss_pct": "0.33",
                "spread_delta_p95": "0.1086",
                "pareto_ratio_delta_mean": "0.0000",
                "boundary_decision": "note_only_stress_slice",
            },
            {
                "scope": "slice",
                "slice": "spread_stress",
                "study_variant": "spread_pg_pop41_gen88",
                "spread_fail_rate": "0.00",
                "joint_safety_fail_rate": "0.00",
                "p90_loss_pct": "0.24",
                "spread_delta_p95": "0.0208",
                "pareto_ratio_delta_mean": "0.0000",
                "boundary_decision": "note_only_stress_slice",
            },
            {
                "scope": "slice",
                "slice": "stable_contrast",
                "study_variant": "current_fast",
                "spread_fail_rate": "0.00",
                "joint_safety_fail_rate": "0.00",
                "p90_loss_pct": "0.10",
                "spread_delta_p95": "0.0150",
                "pareto_ratio_delta_mean": "0.0000",
                "boundary_decision": "note_only_stress_slice",
            },
            {
                "scope": "slice",
                "slice": "stable_contrast",
                "study_variant": "spread_pg_pop41_gen88",
                "spread_fail_rate": "0.50",
                "joint_safety_fail_rate": "0.50",
                "p90_loss_pct": "0.20",
                "spread_delta_p95": "0.0600",
                "pareto_ratio_delta_mean": "0.0000",
                "boundary_decision": "note_only_stress_slice",
            },
            {
                "scope": "slice",
                "slice": "normal_holdout",
                "study_variant": "current_fast",
                "spread_fail_rate": "0.00",
                "joint_safety_fail_rate": "0.00",
                "p90_loss_pct": "0.12",
                "spread_delta_p95": "0.0200",
                "pareto_ratio_delta_mean": "0.0000",
                "boundary_decision": "note_only_stress_slice",
            },
            {
                "scope": "slice",
                "slice": "normal_holdout",
                "study_variant": "spread_pg_pop41_gen88",
                "spread_fail_rate": "0.33",
                "joint_safety_fail_rate": "0.33",
                "p90_loss_pct": "0.30",
                "spread_delta_p95": "0.0800",
                "pareto_ratio_delta_mean": "0.0000",
                "boundary_decision": "note_only_stress_slice",
            }
        ],
        [
            "scope",
            "slice",
            "study_variant",
            "spread_fail_rate",
            "joint_safety_fail_rate",
            "p90_loss_pct",
            "spread_delta_p95",
            "pareto_ratio_delta_mean",
            "boundary_decision",
        ],
    )
    _write_rows_csv(
        zdt1_spread_dir / "failure_trace_table.csv",
        [
            {
                "target_id": "zdt1_fast_spread_safety_fail",
                "study_variant": "current_fast",
                "hv_loss_pct_vs_quality": "0.21",
                "joint_safety_fail": "1.0"
            }
        ],
        [
            "target_id",
            "study_variant",
            "hv_loss_pct_vs_quality",
            "joint_safety_fail",
        ],
    )
    _write_rows_csv(
        zdt1_spread_dir / "failure_hypotheses.csv",
        [
            {
                "hypothesis_id": "zdt1_fast_spread_mechanism_population_generation_candidate",
                "target_id": "zdt1_fast_spread_safety_fail",
                "problem": "zdt1",
                "affected_default": "configs/local_profiles/zdt1_diversity_injection_fast.json",
                "evidence_rows": "3",
                "suspected_mechanism": "spread_mechanism_population_generation_candidate",
                "supporting_trace_signals": "baseline_spread_fail_rate=0.20; pg_spread_reduction=0.10",
                "confidence": "low",
                "current_evidence_strength": "low",
                "probe_attempted": "probe_timing_cooldown2,probe_pg_pop41_gen88",
                "confirmed_or_weakened": "strengthened",
                "recommended_next_probe": "investigate_population_generation_tradeoff",
                "recommended_next_action": "investigate_population_generation_tradeoff",
                "keep_as_regression_case": "True",
                "notes": "spread"
            }
        ],
        [
            "hypothesis_id",
            "target_id",
            "problem",
            "affected_default",
            "evidence_rows",
            "suspected_mechanism",
            "supporting_trace_signals",
            "confidence",
            "current_evidence_strength",
            "probe_attempted",
            "confirmed_or_weakened",
            "recommended_next_probe",
            "recommended_next_action",
            "keep_as_regression_case",
            "notes",
        ],
    )
    _write_rows_csv(
        zdt1_joint_dir / "failure_trace_table.csv",
        [
            {
                "target_id": "zdt1_fast_joint_safety_fail",
                "study_variant": "current_fast",
                "hv_loss_pct_vs_quality": "0.18",
                "joint_safety_fail": "1.0"
            }
        ],
        [
            "target_id",
            "study_variant",
            "hv_loss_pct_vs_quality",
            "joint_safety_fail",
        ],
    )
    _write_rows_csv(
        zdt1_joint_dir / "failure_hypotheses.csv",
        [
            {
                "hypothesis_id": "zdt1_fast_joint_mechanism_still_unisolated",
                "target_id": "zdt1_fast_joint_safety_fail",
                "problem": "zdt1",
                "affected_default": "configs/local_profiles/zdt1_diversity_injection_fast.json",
                "evidence_rows": "1",
                "suspected_mechanism": "joint_mechanism_still_unisolated",
                "supporting_trace_signals": "baseline_joint_fail_rate=0.20; best_probe_joint_fail_rate=0.10",
                "confidence": "low",
                "current_evidence_strength": "low",
                "probe_attempted": "joint_timing_cooldown2,joint_timing_refresh012",
                "confirmed_or_weakened": "weakened",
                "recommended_next_probe": "split_spread_vs_joint_fail_mechanisms",
                "recommended_next_action": "split_spread_vs_joint_fail_mechanisms",
                "keep_as_regression_case": "True",
                "notes": "joint"
            }
        ],
        [
            "hypothesis_id",
            "target_id",
            "problem",
            "affected_default",
            "evidence_rows",
            "suspected_mechanism",
            "supporting_trace_signals",
            "confidence",
            "current_evidence_strength",
            "probe_attempted",
            "confirmed_or_weakened",
            "recommended_next_probe",
            "recommended_next_action",
            "keep_as_regression_case",
            "notes",
        ],
    )

    payload = build_failure_hypothesis_registry(
        previous_registry_path=outputs_root / "future_optimization_targets.json",
        output_dir=outputs_root,
        tsp_study_dir=tsp_dir,
        zdt1_study_dir=zdt1_spread_dir,
        additional_zdt1_study_dirs=[zdt1_joint_dir],
        include_knapsack_freeze=True,
        include_onemax_freeze=True,
    )

    assert Path(str(payload["failure_hypotheses_json"])).exists()
    assert Path(str(payload["failure_hypotheses_csv"])).exists()
    assert Path(str(payload["future_optimization_targets_json"])).exists()
    assert Path(str(payload["failure_trace_notes_md"])).exists()

    target_rows = _read_csv_rows(Path(str(payload["future_optimization_targets_csv"])))
    tsp_row = next(row for row in target_rows if row["target_id"] == "tsp_fast_anti_case_tail")
    zdt1_row = next(row for row in target_rows if row["target_id"] == "zdt1_fast_joint_safety_fail")
    zdt1_spread_row = next(
        row for row in target_rows if row["target_id"] == "zdt1_fast_spread_safety_fail"
    )

    assert tsp_row["current_mechanism_hypothesis"] == "tsp_anticase_late_refinement_deficit"
    assert tsp_row["latest_decision"] == "freeze_as_protocol_limitation"
    assert zdt1_row["recommended_next_action"] == "monitor_with_Q_final_safety"
    assert zdt1_row["latest_decision"] == "monitor_only"
    assert zdt1_spread_row["latest_decision"] == "note_only_stress_slice"
    assert zdt1_spread_row["recommended_next_action"] == "keep_as_slice_conditioned_note"


def test_local_experiment_docs_reference_real_commands() -> None:
    project_root = _project_root()
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    docs_index = (project_root / "docs" / "README.md").read_text(encoding="utf-8")
    guide = (project_root / "docs" / "local_experiment_guide.md").read_text(encoding="utf-8")
    protocol_guide = (project_root / "docs" / "local_protocol_guide.md").read_text(encoding="utf-8")
    examples = (project_root / "examples" / "README.md").read_text(encoding="utf-8")

    assert "python scripts/run_local_experiment.py --preset onemax_small" in readme
    assert "python scripts/run_local_sweep.py --study tsp_quality_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_canonical_default_confirm" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_canonical_default_confirm" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_family_suite" in readme
    assert (
        "python scripts/run_local_sweep.py --study knapsack_canonical_experimental_confirm"
        in readme
    )
    assert "python scripts/run_local_sweep.py --study knapsack_rerun_boundary_holdout" in readme
    assert "python scripts/run_local_sweep.py --study tsp_budget_frontier_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_budget_frontier_study" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_rerun_gate_efficiency_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_two_stage_gate_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_two_stage_gate_study" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_rerun_gate_sanity_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_restart_portfolio_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_restart_portfolio_study" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_multistart_sanity_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_ranking_fidelity_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_ranking_fidelity_study" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_triage_sanity_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_qf_tolerance_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_fast_tail_hardening_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_fast_tail_confirm" in readme
    assert "python scripts/run_local_sweep.py --study tsp_qf_recalibration_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_qf_recalibration_confirm" in readme
    assert "python scripts/run_local_sweep.py --study tsp_fast_legacy_reference_check" in readme
    assert "python scripts/run_local_sweep.py --study tsp_qf_recalibration_after_hardening" in readme
    assert (
        "python scripts/run_local_sweep.py --study tsp_seed_budget_recheck_after_hardening"
        in readme
    )
    assert "python scripts/run_local_sweep.py --study zdt1_qf_tolerance_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_qf_tiny_freeze_check" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_qf_tiny_freeze_recheck" in readme
    assert (
        "python scripts/run_local_sweep.py --study zdt1_qf_recalibration_after_hardening"
        in readme
    )
    assert (
        "python scripts/run_local_sweep.py --study zdt1_seed_budget_recheck_after_hardening"
        in readme
    )
    assert "python scripts/run_local_sweep.py --study knapsack_note_freeze_recheck" in readme
    assert "python scripts/run_local_sweep.py --study onemax_control_freeze_recheck" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_note_freeze_check" in readme
    assert "python scripts/run_local_sweep.py --study tsp_seed_budget_calibration" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_seed_budget_calibration" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_seed_budget_sanity" in readme
    assert "python scripts/run_local_sweep.py --study onemax_seed_budget_control" in readme
    assert "python scripts/run_local_sweep.py --study tsp_sequential_compare_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_sequential_compare_study" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_sequential_sanity" in readme
    assert "python scripts/run_local_sweep.py --study onemax_control_sequential_check" in readme
    assert "python scripts/run_local_sweep.py --study tsp_stress_suite" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_stress_suite" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_stress_suite" in readme
    assert "python scripts/run_local_sweep.py --study onemax_control_stress_check" in readme
    assert "python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_confirm" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_confirm" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_stress_note_freeze_check" in readme
    assert "python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check" in readme
    assert "python scripts/run_local_sweep.py --study tsp_stress_refresh_suite" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_stress_refresh_suite" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_stress_refresh_suite" in readme
    assert "python scripts/run_local_sweep.py --study onemax_control_refresh_check" in readme
    assert "python scripts/build_stress_refresh_registry.py" in readme
    assert "python scripts/run_local_sweep.py --study tsp_stress_target_reduction_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_stress_target_reduction_confirm" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_confirm" in readme
    assert "python scripts/run_local_sweep.py --study knapsack_stress_freeze_check" in readme
    assert "python scripts/build_stress_target_reduction_registry.py" in readme
    assert "python scripts/run_local_sweep.py --study tsp_failure_trace_suite" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_failure_trace_suite" in readme
    assert "python scripts/build_failure_hypotheses_registry.py" in readme
    assert "python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_confirm" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_confirm" in readme
    assert (
        "python scripts/build_failure_hypotheses_registry.py --tsp-study-name "
        "tsp_target_hypothesis_probe_confirm --zdt1-study-name "
        "zdt1_target_hypothesis_probe_confirm"
    ) in readme
    assert "python scripts/run_local_sweep.py --study tsp_population_generation_probe_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_population_generation_probe_confirm" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_confirm" in readme
    assert (
        "python scripts/build_failure_hypotheses_registry.py --tsp-study-name "
        "tsp_population_generation_probe_confirm --zdt1-study-name "
        "zdt1_timing_vs_pg_probe_confirm"
    ) in readme
    assert "python scripts/run_local_sweep.py --study tsp_extreme_tail_pg_contour_study" in readme
    assert "python scripts/run_local_sweep.py --study tsp_extreme_tail_pg_contour_confirm" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_spread_pg_probe_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_spread_pg_probe_confirm" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_joint_timing_probe_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_joint_timing_probe_confirm" in readme
    assert (
        "python scripts/build_failure_hypotheses_registry.py --tsp-study-name "
        "tsp_extreme_tail_pg_contour_confirm --zdt1-spread-study-name "
        "zdt1_spread_pg_probe_confirm --zdt1-joint-study-name "
        "zdt1_joint_timing_probe_confirm"
    ) in readme
    assert "python scripts/run_local_sweep.py --study tsp_tail_freeze_recheck" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_study" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_confirm" in readme
    assert "python scripts/run_local_sweep.py --study zdt1_joint_note_freeze_check" in readme
    assert (
        "python scripts/build_failure_hypotheses_registry.py --tsp-study-name "
        "tsp_tail_freeze_recheck --zdt1-spread-study-name "
        "zdt1_spread_candidate_boundary_confirm --zdt1-joint-study-name "
        "zdt1_joint_note_freeze_check"
    ) in readme
    assert "[Local experiment guide](local_experiment_guide.md)" in docs_index
    assert "python scripts/run_local_sweep.py --study zdt1_nsga2_mutation_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_seed_fraction_ablation_study" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_threshold_anatomy_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_fast_profile_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_fast_profile_confirm" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_rerun_gate_confirm" in guide
    assert "python scripts/run_local_sweep.py --study tsp_two_stage_gate_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_two_stage_gate_confirm" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_rerun_gate_sanity_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_restart_portfolio_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_restart_portfolio_confirm" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_multistart_sanity_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_triage_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_triage_confirm" in guide
    assert "python scripts/run_local_sweep.py --study onemax_control_ranking_check" in guide
    assert "python scripts/run_local_sweep.py --study tsp_qf_tolerance_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_qf_tolerance_confirm" in guide
    assert "python scripts/run_local_sweep.py --study tsp_fast_tail_hardening_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_fast_tail_confirm" in guide
    assert "python scripts/run_local_sweep.py --study tsp_qf_recalibration_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_qf_recalibration_confirm" in guide
    assert "python scripts/run_local_sweep.py --study tsp_fast_legacy_reference_check" in guide
    assert (
        "python scripts/run_local_sweep.py --study tsp_qf_recalibration_after_hardening"
        in guide
    )
    assert (
        "python scripts/run_local_sweep.py --study tsp_seed_budget_recheck_after_hardening"
        in guide
    )
    assert "python scripts/run_local_sweep.py --study zdt1_qf_tolerance_study" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_qf_tolerance_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_qf_tiny_freeze_check" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_qf_tiny_freeze_recheck" in guide
    assert (
        "python scripts/run_local_sweep.py --study zdt1_qf_recalibration_after_hardening"
        in guide
    )
    assert (
        "python scripts/run_local_sweep.py --study zdt1_seed_budget_recheck_after_hardening"
        in guide
    )
    assert "python scripts/run_local_sweep.py --study knapsack_note_freeze_recheck" in guide
    assert "python scripts/run_local_sweep.py --study onemax_control_freeze_recheck" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_note_freeze_check" in guide
    assert "python scripts/run_local_sweep.py --study onemax_control_freeze_check" in guide
    assert "python scripts/run_local_sweep.py --study tsp_seed_budget_calibration" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_seed_budget_calibration" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_seed_budget_sanity" in guide
    assert "python scripts/run_local_sweep.py --study onemax_seed_budget_control" in guide
    assert "python scripts/run_local_sweep.py --study tsp_sequential_compare_study" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_sequential_compare_study" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_sequential_sanity" in guide
    assert "python scripts/run_local_sweep.py --study onemax_control_sequential_check" in guide
    assert "python scripts/run_local_sweep.py --study tsp_stress_suite" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_stress_suite" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_stress_suite" in guide
    assert "python scripts/run_local_sweep.py --study onemax_control_stress_check" in guide
    assert "python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_study" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_confirm" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_stress_note_freeze_check" in guide
    assert "python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check" in guide
    assert "python scripts/run_local_sweep.py --study tsp_stress_refresh_suite" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_stress_refresh_suite" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_stress_refresh_suite" in guide
    assert "python scripts/run_local_sweep.py --study onemax_control_refresh_check" in guide
    assert "python scripts/build_stress_refresh_registry.py" in guide
    assert "python scripts/run_local_sweep.py --study tsp_stress_target_reduction_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_stress_target_reduction_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_study" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_confirm" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_stress_freeze_check" in guide
    assert "python scripts/build_stress_target_reduction_registry.py" in guide
    assert "python scripts/run_local_sweep.py --study tsp_failure_trace_suite" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_failure_trace_suite" in guide
    assert "python scripts/build_failure_hypotheses_registry.py" in guide
    assert "python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_study" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_confirm" in guide
    assert "python scripts/run_local_sweep.py --study tsp_population_generation_probe_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_population_generation_probe_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_study" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_confirm" in guide
    assert (
        "python scripts/build_failure_hypotheses_registry.py --tsp-study-name "
        "tsp_population_generation_probe_confirm --zdt1-study-name "
        "zdt1_timing_vs_pg_probe_confirm"
    ) in guide
    assert "python scripts/run_local_sweep.py --study tsp_extreme_tail_pg_contour_study" in guide
    assert "python scripts/run_local_sweep.py --study tsp_extreme_tail_pg_contour_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_spread_pg_probe_study" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_spread_pg_probe_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_joint_timing_probe_study" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_joint_timing_probe_confirm" in guide
    assert (
        "python scripts/build_failure_hypotheses_registry.py --tsp-study-name "
        "tsp_extreme_tail_pg_contour_confirm --zdt1-spread-study-name "
        "zdt1_spread_pg_probe_confirm --zdt1-joint-study-name "
        "zdt1_joint_timing_probe_confirm"
    ) in guide
    assert "python scripts/run_local_sweep.py --study tsp_tail_freeze_recheck" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_study" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_confirm" in guide
    assert "python scripts/run_local_sweep.py --study zdt1_joint_note_freeze_check" in guide
    assert (
        "python scripts/build_failure_hypotheses_registry.py --tsp-study-name "
        "tsp_tail_freeze_recheck --zdt1-spread-study-name "
        "zdt1_spread_candidate_boundary_confirm --zdt1-joint-study-name "
        "zdt1_joint_note_freeze_check"
    ) in guide
    assert "python scripts/run_local_sweep.py --study tsp_tail_freeze_recheck" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_confirm" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_joint_note_freeze_check" in examples
    assert (
        "python scripts/build_failure_hypotheses_registry.py --tsp-study-name "
        "tsp_tail_freeze_recheck --zdt1-spread-study-name "
        "zdt1_spread_candidate_boundary_confirm --zdt1-joint-study-name "
        "zdt1_joint_note_freeze_check"
    ) in examples
    assert "python scripts/run_local_sweep.py --study tsp_tail_freeze_recheck" in protocol_guide
    assert "python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_confirm" in protocol_guide
    assert "python scripts/run_local_sweep.py --study zdt1_joint_note_freeze_check" in protocol_guide
    assert "python scripts/run_local_sweep.py --study knapsack_seeded_repair_anatomy_study" in guide
    assert "python scripts/run_local_sweep.py --study knapsack_rerun_boundary_suite" in guide
    assert (
        "python scripts/run_local_experiment.py --config "
        "configs/local_profiles/knapsack_repair_local_experimental.json"
    ) in guide
    assert (
        "python scripts/run_local_experiment.py --config "
        "configs/local_profiles/tsp_seeded_swap_local_fast.json"
    ) in guide
    assert (
        "python scripts/run_local_experiment.py --config "
        "configs/local_profiles/zdt1_diversity_injection_fast.json"
    ) in guide
    assert "plot_local_results.py --study-dir" in guide
    assert "python scripts/run_local_sweep.py --study onemax_mutation_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_seed_fraction_ablation_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_cooldown_anatomy_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_two_stage_gate_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_two_stage_gate_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_restart_portfolio_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_restart_portfolio_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_ranking_fidelity_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_ranking_fidelity_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_qf_tolerance_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_fast_tail_hardening_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_fast_tail_confirm" in examples
    assert "python scripts/run_local_sweep.py --study tsp_qf_recalibration_study" in examples
    assert (
        "python scripts/run_local_sweep.py --study tsp_qf_recalibration_after_hardening"
        in examples
    )
    assert (
        "python scripts/run_local_sweep.py --study tsp_seed_budget_recheck_after_hardening"
        in examples
    )
    assert "python scripts/run_local_sweep.py --study tsp_stress_suite" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_stress_suite" in examples
    assert "python scripts/run_local_sweep.py --study knapsack_stress_suite" in examples
    assert "python scripts/run_local_sweep.py --study onemax_control_stress_check" in examples
    assert "python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_confirm" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_confirm" in examples
    assert "python scripts/run_local_sweep.py --study knapsack_stress_note_freeze_check" in examples
    assert "python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check" in examples
    assert "python scripts/run_local_sweep.py --study tsp_stress_refresh_suite" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_stress_refresh_suite" in examples
    assert "python scripts/run_local_sweep.py --study knapsack_stress_refresh_suite" in examples
    assert "python scripts/run_local_sweep.py --study onemax_control_refresh_check" in examples
    assert "python scripts/build_stress_refresh_registry.py" in examples
    assert "python scripts/run_local_sweep.py --study tsp_stress_target_reduction_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_stress_target_reduction_confirm" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_confirm" in examples
    assert "python scripts/run_local_sweep.py --study knapsack_stress_freeze_check" in examples
    assert "python scripts/build_stress_target_reduction_registry.py" in examples
    assert "python scripts/run_local_sweep.py --study tsp_failure_trace_suite" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_failure_trace_suite" in examples
    assert "python scripts/build_failure_hypotheses_registry.py" in examples
    assert "python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_confirm" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_confirm" in examples
    assert "python scripts/run_local_sweep.py --study tsp_population_generation_probe_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_population_generation_probe_confirm" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_confirm" in examples
    assert (
        "python scripts/build_failure_hypotheses_registry.py --tsp-study-name "
        "tsp_population_generation_probe_confirm --zdt1-study-name "
        "zdt1_timing_vs_pg_probe_confirm"
    ) in examples
    assert "python scripts/run_local_sweep.py --study tsp_extreme_tail_pg_contour_study" in examples
    assert "python scripts/run_local_sweep.py --study tsp_extreme_tail_pg_contour_confirm" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_spread_pg_probe_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_spread_pg_probe_confirm" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_joint_timing_probe_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_joint_timing_probe_confirm" in examples
    assert (
        "python scripts/build_failure_hypotheses_registry.py --tsp-study-name "
        "tsp_extreme_tail_pg_contour_confirm --zdt1-spread-study-name "
        "zdt1_spread_pg_probe_confirm --zdt1-joint-study-name "
        "zdt1_joint_timing_probe_confirm"
    ) in examples
    assert "python scripts/run_local_sweep.py --study tsp_qf_recalibration_confirm" in examples
    assert "python scripts/run_local_sweep.py --study tsp_fast_legacy_reference_check" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_qf_tolerance_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_qf_tiny_freeze_check" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_qf_tiny_freeze_recheck" in examples
    assert (
        "python scripts/run_local_sweep.py --study zdt1_qf_recalibration_after_hardening"
        in examples
    )
    assert (
        "python scripts/run_local_sweep.py --study zdt1_seed_budget_recheck_after_hardening"
        in examples
    )
    assert "python scripts/run_local_sweep.py --study knapsack_note_freeze_recheck" in examples
    assert "python scripts/run_local_sweep.py --study onemax_control_freeze_recheck" in examples
    assert "python scripts/run_local_sweep.py --study knapsack_note_freeze_check" in examples
    assert "python scripts/run_local_sweep.py --study tsp_seed_budget_calibration" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_seed_budget_calibration" in examples
    assert "python scripts/run_local_sweep.py --study knapsack_seed_budget_sanity" in examples
    assert "python scripts/run_local_sweep.py --study onemax_seed_budget_control" in examples
    assert "python scripts/run_local_sweep.py --study tsp_sequential_compare_study" in examples
    assert "python scripts/run_local_sweep.py --study zdt1_sequential_compare_study" in examples
    assert "python scripts/run_local_sweep.py --study knapsack_sequential_sanity" in examples
    assert "python scripts/run_local_sweep.py --study onemax_control_sequential_check" in examples
    assert "python scripts/run_local_sweep.py --study onemax_control_budget_check" in examples
    assert (
        "python scripts/run_local_sweep.py --study knapsack_feasibility_control_study"
        in examples
    )
    assert (
        "python scripts/run_local_sweep.py --study knapsack_repair_vs_restart_confirm"
        in examples
    )
    assert "python scripts/run_local_sweep.py --study knapsack_rerun_gate_sanity_study" in examples
    assert "python scripts/run_local_sweep.py --study knapsack_multistart_sanity_study" in examples
    assert "python scripts/run_local_sweep.py --study knapsack_triage_sanity_study" in examples
