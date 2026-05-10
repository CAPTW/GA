from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from ga_lab.config import GAConfig, load_config
from ga_lab.convergence_diagnostics import (
    ProgressState,
    configured_evaluation_budget,
    diversity_metrics,
    update_progress_state,
)
from ga_lab.local_experiments import load_local_study
from ga_lab.problems.onemax import OneMaxProblem
from ga_lab.problems.tsp import TSPProblem
from ga_lab.problems.zdt1 import ZDT1Problem


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_json_command(*args: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=_project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _write_manifest(tmp_path: Path, file_name: str, payload: dict[str, object]) -> Path:
    manifest_path = tmp_path / file_name
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_preset(name: str) -> GAConfig:
    path = _project_root() / "configs" / "presets" / f"{name}.json"
    return GAConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def test_progress_state_reports_stagnation_and_recent_improvement() -> None:
    state = ProgressState(maximize=True, recent_window=3)
    first = update_progress_state(state, generation=0, value=10.0)
    second = update_progress_state(state, generation=1, value=10.0)
    third = update_progress_state(state, generation=2, value=12.0)

    assert first["generations_since_last_improvement"] == 0
    assert second["generations_since_last_improvement"] == 1
    assert third["improvement_delta"] == 2.0
    assert third["recent_window_improvement"] >= 2.0


def test_diversity_metrics_cover_bit_permutation_and_real() -> None:
    onemax_config = _load_preset("onemax_small")
    tsp_config = _load_preset("tsp_small")
    zdt_config = _load_preset("zdt1_small")

    bit_metrics = diversity_metrics(
        onemax_config,
        OneMaxProblem(),
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    )
    tsp_problem = TSPProblem(num_cities=4, coordinates=[[0, 0], [1, 0], [1, 1], [0, 1]])
    permutation_metrics = diversity_metrics(
        tsp_config,
        tsp_problem,
        [[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 1.0, 3.0]],
    )
    real_metrics = diversity_metrics(
        zdt_config,
        ZDT1Problem(),
        [[0.1, 0.2, 0.3], [0.8, 0.4, 0.6]],
    )

    assert "allele_entropy" in bit_metrics
    assert "edge_diversity_ratio" in permutation_metrics
    assert "population_spread" in real_metrics


def test_configured_budget_matches_expected_formulas() -> None:
    ga_config = _load_preset("onemax_small")
    nsga_config = _load_preset("zdt1_small")

    assert configured_evaluation_budget(ga_config) == (
        ga_config.population_size * (ga_config.generations + 2)
    )
    assert configured_evaluation_budget(nsga_config) == (
        nsga_config.population_size * ((3 * nsga_config.generations) + 2)
    )


def test_onemax_adaptive_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "onemax_adaptive_smoke.json",
        {
            "study_name": "onemax_adaptive_smoke",
            "description": "Tiny adaptive onemax smoke.",
            "problem": "onemax",
            "base_preset": "onemax_small",
            "shared_overrides": {
                "population_size": 20,
                "generations": 20,
                "algorithm_options": {
                    "adaptive_mutation_min_rate": 0.005,
                    "adaptive_mutation_max_rate": 0.05,
                    "stagnation_window": 4
                }
            },
            "sweep": {
                "algorithm_options.adaptive_policy": ["none", "stagnation_boost"]
            },
            "seeds": [1],
            "budget_ceiling": 440,
            "primary_metric": "evaluations_to_target",
            "plotting": {
                "history_metric": "best_fitness"
            },
            "runtime_budget_note": "Tiny smoke study."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(payload["plots"]["plot_convergence"])).exists()
    assert Path(str(payload["plots"]["plot_diversity"])).exists()
    assert Path(str(payload["plots"]["plot_stagnation"])).exists()


def test_tsp_adaptive_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "tsp_diversity_smoke.json",
        {
            "study_name": "tsp_diversity_smoke",
            "description": "Tiny adaptive tsp diversity-injection smoke.",
            "problem": "tsp",
            "base_preset": "tsp_small",
            "shared_overrides": {
                "population_size": 20,
                "generations": 20,
                "algorithm_options": {
                    "adaptive_policy": "low_diversity_injection",
                    "refresh_fraction": 0.25,
                    "diversity_threshold": 1.0,
                    "adaptation_cooldown": 2,
                }
            },
            "sweep": {
                "algorithm_options.adaptive_policy": ["none", "low_diversity_injection"]
            },
            "seeds": [1],
            "budget_ceiling": 440,
            "primary_metric": "best_route_distance",
            "plotting": {
                "history_metric": "best_route_distance"
            },
            "runtime_budget_note": "Tiny smoke study."
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
    adaptive_row = next(
        row
        for row in summary_rows
        if row["algorithm_options.adaptive_policy"] == "low_diversity_injection"
    )

    assert Path(str(payload["plots"]["plot_route_distance"])).exists()
    assert Path(str(payload["plots"]["plot_diversity"])).exists()
    assert Path(str(payload["plots"]["plot_stagnation"])).exists()
    assert Path(str(payload["plots"]["plot_diversity_vs_distance"])).exists()
    assert Path(str(payload["plots"]["plot_trigger_events"])).exists()
    assert Path(str(payload["plots"]["plot_post_trigger_gain"])).exists()
    assert Path(str(payload["plots"]["plot_refresh_schedule_vs_gain"])).exists()
    assert float(adaptive_row["trigger_fire_count_mean"]) > 0.0
    assert float(adaptive_row["post_trigger_improvement_mean"]) > 0.0
    assert float(adaptive_row["average_refresh_fraction_realized_mean"]) > 0.0


def test_tsp_delayed_and_periodic_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "tsp_delayed_periodic_smoke.json",
        {
            "study_name": "tsp_delayed_periodic_smoke",
            "description": "Tiny smoke for delayed trigger and periodic refresh control.",
            "problem": "tsp",
            "base_preset": "tsp_small",
            "shared_overrides": {
                "population_size": 20,
                "generations": 20,
                "algorithm_options": {
                    "refresh_fraction": 0.25,
                    "diversity_threshold": 1.0,
                    "adaptation_cooldown": 1,
                    "warmup_generation": 4,
                    "stagnation_window": 2,
                    "periodic_refresh_every": 3,
                }
            },
            "sweep": {
                "algorithm_options.adaptive_policy": [
                    "none",
                    "delayed_trigger_v1",
                    "periodic_refresh_control",
                ]
            },
            "seeds": [1],
            "budget_ceiling": 440,
            "primary_metric": "best_route_distance",
            "plotting": {
                "history_metric": "best_route_distance"
            },
            "runtime_budget_note": "Tiny delayed/periodic trigger smoke study."
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
    delayed_row = next(
        row
        for row in summary_rows
        if row["algorithm_options.adaptive_policy"] == "delayed_trigger_v1"
    )
    periodic_row = next(
        row
        for row in summary_rows
        if row["algorithm_options.adaptive_policy"] == "periodic_refresh_control"
    )

    assert Path(str(payload["plots"]["plot_trigger_events"])).exists()
    assert Path(str(payload["plots"]["plot_refresh_schedule_vs_gain"])).exists()
    assert float(delayed_row["trigger_fire_count_mean"]) > 0.0
    assert float(delayed_row["first_trigger_generation_mean"]) >= 4.0
    assert float(periodic_row["trigger_fire_count_mean"]) > 0.0
    assert float(periodic_row["first_trigger_generation_mean"]) >= 4.0


def test_tsp_regime_switching_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "tsp_regime_switching_smoke.json",
        {
            "study_name": "tsp_regime_switching_smoke",
            "description": "Tiny smoke for online regime switching.",
            "problem": "tsp",
            "base_preset": "tsp_small",
            "cases": [
                {
                    "case_id": "switch_bridge_8",
                    "note": "tiny bridge-like switch case",
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
            "shared_overrides": {
                "population_size": 20,
                "generations": 12,
                "algorithm_options": {
                    "diversity_threshold": 1.0,
                    "refresh_fraction": 0.25,
                    "adaptation_cooldown": 1,
                    "warmup_generation": 2,
                    "stagnation_window": 2,
                    "improvement_epsilon": 999.0,
                    "rescue_window": 3,
                    "post_trigger_nontrivial_improvement": 1.0,
                    "useless_trigger_window": 3,
                },
            },
            "sweep": {
                "algorithm_options.adaptive_policy": [
                    "none",
                    "switch_controller_v1",
                    "switch_controller_v2",
                ]
            },
            "seeds": [1],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny switching smoke study.",
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
    switch_v1 = next(
        row
        for row in summary_rows
        if row["algorithm_options.adaptive_policy"] == "switch_controller_v1"
    )
    switch_v2 = next(
        row
        for row in summary_rows
        if row["algorithm_options.adaptive_policy"] == "switch_controller_v2"
    )

    assert Path(str(payload["plots"]["plot_mode_switch_timeline"])).exists()
    assert Path(str(payload["plots"]["plot_diversity_vs_mode"])).exists()
    assert Path(str(payload["plots"]["plot_collapse_onset_vs_switch"])).exists()
    assert Path(str(payload["plots"]["plot_regret_vs_policy"])).exists()
    assert Path(str(payload["plots"]["plot_budget_band_vs_regret"])).exists()
    assert float(switch_v1["mode_switch_count_mean"]) > 0.0
    assert float(switch_v1["first_switch_generation_mean"]) >= 2.0
    assert float(switch_v2["mode_switch_count_mean"]) > 0.0
    assert float(switch_v2["time_in_trigger_mode_mean"]) > 0.0


def test_tsp_case_group_summary_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "tsp_case_group_summary_smoke.json",
        {
            "study_name": "tsp_case_group_summary_smoke",
            "description": "Tiny smoke for rescue-target versus anti-case TSP summaries.",
            "problem": "tsp",
            "base_preset": "tsp_small",
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
                                [7.0, 2.0],
                                [14.0, -1.0],
                                [21.0, 3.0],
                                [28.0, -2.0],
                                [35.0, 4.0],
                                [42.0, -3.0],
                                [20.0, 12.0],
                            ],
                        },
                    },
                },
            ],
            "shared_overrides": {
                "population_size": 20,
                "generations": 12,
                "algorithm_options": {
                    "diversity_threshold": 1.0,
                    "refresh_fraction": 0.25,
                    "adaptation_cooldown": 1,
                    "warmup_generation": 2,
                    "stagnation_window": 2,
                    "improvement_epsilon": 999.0,
                    "rescue_window": 3,
                    "post_trigger_nontrivial_improvement": 1.0,
                    "useless_trigger_window": 3,
                },
            },
            "sweep": {
                "algorithm_options.adaptive_policy": [
                    "none",
                    "decay_mutation",
                    "switch_controller_v1",
                ]
            },
            "seeds": [1],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny rescue-target versus anti-case smoke study.",
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert payload["case_group_summary_csv"] is not None
    case_group_rows = _read_csv_rows(Path(str(payload["case_group_summary_csv"])))

    assert Path(str(payload["plots"]["plot_rescue_target_vs_anticase_gap"])).exists()
    assert any(row["case_group"] == "rescue_target" for row in case_group_rows)
    assert any(row["case_group"] == "anti_case" for row in case_group_rows)
    switch_row = next(
        row
        for row in case_group_rows
        if row["case_group"] == "overall"
        and row["combo_label"].startswith("algorithm_options.adaptive_policy=switch_controller_v1")
    )
    assert switch_row["mode_switch_count_mean"] != ""


def test_tsp_fixed_stack_anatomy_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "tsp_fixed_stack_anatomy_smoke.json",
        {
            "study_name": "tsp_fixed_stack_anatomy_smoke",
            "description": "Tiny smoke for TSP fixed-stack anatomy metrics and plots.",
            "problem": "tsp",
            "base_preset": "tsp_small",
            "cases": [
                {
                    "case_id": "bridge_8",
                    "group": "rescue_target",
                    "note": "tiny bridge-like fixed-stack case",
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
            "shared_overrides": {
                "population_size": 20,
                "generations": 12,
                "log_every": 1,
            },
            "variant_overrides": {
                "none": {
                    "algorithm": "ga",
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none"
                    }
                },
                "seeded_swap_seed25": {
                    "algorithm": "hybrid_ga",
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.25,
                        "local_search_strategy": "none"
                    }
                },
                "seeded_swap_seed50": {
                    "algorithm": "hybrid_ga",
                    "mutation": "swap",
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none"
                    }
                }
            },
            "sweep": {
                "study_variant": ["none", "seeded_swap_seed25", "seeded_swap_seed50"]
            },
            "seeds": [1],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny fixed-stack anatomy smoke.",
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
    seeded_row = next(
        row for row in summary_rows if row["study_variant"] == "seeded_swap_seed50"
    )

    assert Path(str(payload["plots"]["plot_seed_fraction_vs_gap"])).exists()
    assert Path(str(payload["plots"]["plot_seed_source_vs_gap"])).exists()
    assert Path(str(payload["plots"]["plot_mutation_operator_vs_gap"])).exists()
    assert Path(str(payload["plots"]["plot_initial_quality_vs_final_gap"])).exists()
    assert seeded_row["initial_best_route_distance_mean"] != ""
    assert seeded_row["init_to_final_gain_mean"] != ""
    assert seeded_row["regret_vs_current_preferred_profile_mean"] != ""


def test_tsp_hardcase_case_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "tsp_hardcase_case_smoke.json",
        {
            "study_name": "tsp_hardcase_case_smoke",
            "description": (
                "Tiny hard-case smoke with explicit case metadata "
                "and trigger selectivity metrics."
            ),
            "problem": "tsp",
            "base_preset": "tsp_small",
            "cases": [
                {
                    "case_id": "bridge_8",
                    "note": "tiny bridge case",
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
                                [15.0, 10.0]
                            ]
                        }
                    }
                },
                {
                    "case_id": "corridor_8",
                    "note": "tiny corridor case",
                    "overrides": {
                        "genome_length": 8,
                        "problem_options": {
                            "num_cities": 8,
                            "coordinates": [
                                [0.0, 0.0],
                                [6.0, 2.0],
                                [12.0, -1.0],
                                [18.0, 3.0],
                                [24.0, -2.0],
                                [30.0, 4.0],
                                [36.0, -3.0],
                                [18.0, 14.0]
                            ]
                        }
                    }
                }
            ],
            "shared_overrides": {
                "population_size": 20,
                "generations": 12,
                "algorithm_options": {
                    "refresh_fraction": 0.25,
                    "diversity_threshold": 1.0,
                    "adaptation_cooldown": 2,
                    "collapse_warmup_generation": 1,
                    "improvement_epsilon": 999.0,
                    "post_trigger_nontrivial_improvement": 1.0,
                    "useless_trigger_window": 3
                }
            },
            "sweep": {
                "algorithm_options.adaptive_policy": ["none", "low_diversity_injection"]
            },
            "seeds": [1],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {
                "history_metric": "best_route_distance"
            },
            "runtime_budget_note": "Tiny hard-case smoke study."
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
    adaptive_row = next(
        row
        for row in summary_rows
        if row["algorithm_options.adaptive_policy"] == "low_diversity_injection"
    )

    assert Path(str(payload["plots"]["plot_collapse_onset_vs_trigger"])).exists()
    assert Path(str(payload["plots"]["plot_refresh_volume_vs_gain"])).exists()
    assert Path(str(payload["plots"]["plot_budget_band_vs_gap"])).exists()
    assert adaptive_row["case_id"] in {"bridge_8", "corridor_8"}
    assert float(adaptive_row["trigger_fire_count_mean"]) > 0.0
    assert adaptive_row["collapse_onset_generation_mean"] != ""
    assert adaptive_row["useless_trigger_rate_mean"] != ""


def test_zdt1_adaptive_smoke_and_plot_rerender(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "zdt1_adaptive_smoke.json",
        {
            "study_name": "zdt1_adaptive_smoke",
            "description": "Tiny adaptive zdt1 smoke.",
            "problem": "zdt1",
            "base_preset": "zdt1_small",
            "shared_overrides": {
                "population_size": 20,
                "generations": 20,
                "algorithm_options": {
                    "diversity_threshold": 0.4,
                    "refresh_fraction": 0.2,
                    "adaptation_cooldown": 4
                }
            },
            "sweep": {
                "algorithm_options.adaptive_policy": ["none", "low_diversity_injection"]
            },
            "seeds": [1],
            "budget_ceiling": 1240,
            "primary_metric": "hypervolume",
            "plotting": {
                "history_metric": "hypervolume"
            },
            "runtime_budget_note": "Tiny smoke study."
        },
    )

    payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    study_dir = Path(str(payload["study_dir"]))
    assert Path(str(payload["plots"]["plot_hypervolume"])).exists()
    assert Path(str(payload["plots"]["plot_diversity"])).exists()
    assert Path(str(payload["plots"]["plot_hv_vs_spread"])).exists()
    assert Path(str(payload["plots"]["plot_budget_vs_hv"])).exists()
    assert Path(str(payload["plots"]["plot_threshold_vs_hv"])).exists()
    assert Path(str(payload["plots"]["plot_refresh_vs_hv"])).exists()
    assert Path(str(payload["plots"]["plot_cooldown_vs_hv"])).exists()
    assert Path(str(payload["plots"]["plot_final_pareto_front"])).exists()
    assert Path(str(payload["plots"]["plot_trigger_events"])).exists()

    summary_rows = _read_csv_rows(Path(str(payload["summary_csv"])))
    adaptive_row = next(
        row
        for row in summary_rows
        if row["algorithm_options.adaptive_policy"] == "low_diversity_injection"
    )
    assert adaptive_row["regret_vs_none_mean"] != ""
    assert adaptive_row["front_size_mean"] != ""

    rerendered = _run_json_command(
        str(_project_root() / "scripts" / "plot_local_results.py"),
        "--study-dir",
        str(study_dir),
    )
    assert Path(str(rerendered["plots"]["plot_hypervolume"])).exists()


def test_knapsack_restart_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "knapsack_restart_smoke.json",
        {
            "study_name": "knapsack_restart_smoke",
            "description": "Tiny knapsack restart smoke.",
            "problem": "knapsack",
            "base_preset": "knapsack_small",
            "shared_overrides": {
                "population_size": 20,
                "generations": 20,
                "algorithm_options": {
                    "adaptive_policy": "stagnation_restart",
                    "stagnation_window": 2,
                    "refresh_fraction": 0.25,
                    "adaptation_cooldown": 2,
                }
            },
            "sweep": {
                "algorithm_options.stagnation_window": [2, 999]
            },
            "seeds": [1],
            "budget_ceiling": 440,
            "primary_metric": "best_feasible_fitness",
            "plotting": {
                "history_metric": "best_fitness"
            },
            "runtime_budget_note": "Tiny knapsack restart smoke study."
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
    adaptive_row = next(
        row
        for row in summary_rows
        if row["algorithm_options.stagnation_window"] == "2"
    )

    assert Path(str(payload["plots"]["plot_feasibility"])).exists()
    assert Path(str(payload["plots"]["plot_violation"])).exists()
    assert Path(str(payload["plots"]["plot_mutation_rate"])).exists()
    assert Path(str(payload["plots"]["plot_trigger_events"])).exists()
    assert float(adaptive_row["trigger_fire_count_mean"]) > 0.0


def test_knapsack_family_and_repair_anatomy_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "knapsack_family_repair_smoke.json",
        {
            "study_name": "knapsack_family_repair_smoke",
            "description": "Tiny knapsack family anatomy smoke.",
            "problem": "knapsack",
            "base_preset": "knapsack_small",
            "__study_metadata__": {
                "family_label": "tight_capacity_tiny",
                "capacity_ratio": 0.2,
                "correlation_note": "tiny_tight_capacity"
            },
            "cases": [
                {
                    "case_id": "tight_capacity_tiny_a",
                    "group": "tight_capacity_tiny",
                    "note": "tiny tight-capacity smoke case",
                    "overrides": {
                        "population_size": 20,
                        "generations": 20,
                        "genome_length": 12,
                        "problem_options": {
                            "num_items": 12,
                            "weights": [9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2],
                            "values": [16, 15, 14, 13, 9, 8, 7, 6, 5, 5, 4, 4],
                            "capacity": 18
                        }
                    }
                }
            ],
            "variant_overrides": {
                "greedy_local_search": {
                    "__local_baseline__": "knapsack_greedy_local_search"
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
            "shared_overrides": {
                "population_size": 20,
                "generations": 20
            },
            "sweep": {
                "study_variant": [
                    "greedy_local_search",
                    "none",
                    "repair_only"
                ]
            },
            "seeds": [1],
            "budget_ceiling": 440,
            "primary_metric": "best_feasible_fitness",
            "plotting": {
                "history_metric": "best_fitness"
            },
            "runtime_budget_note": "Tiny knapsack family anatomy smoke."
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
    case_group_rows = _read_csv_rows(Path(str(payload["case_group_summary_csv"])))
    greedy_row = next(row for row in summary_rows if row["study_variant"] == "greedy_local_search")
    repair_row = next(row for row in summary_rows if row["study_variant"] == "repair_only")

    assert Path(str(payload["plots"]["plot_family_vs_regret"])).exists()
    assert Path(str(payload["plots"]["plot_seeded_vs_repair_gap"])).exists()
    assert Path(str(payload["plots"]["plot_repair_vs_greedy_gap"])).exists()
    assert Path(str(payload["plots"]["plot_init_feasible_vs_final_gain"])).exists()
    assert Path(str(payload["plots"]["plot_initial_feasible_fraction_vs_gain"])).exists()
    if "plot_capacity_tightness_vs_gain" in payload["plots"]:
        assert Path(str(payload["plots"]["plot_capacity_tightness_vs_gain"])).exists()
    assert Path(str(payload["plots"]["plot_budget_vs_feasible_gain"])).exists()
    assert greedy_row["algorithm"] == "knapsack_greedy_local_search"
    assert greedy_row["extra_evaluations_from_adaptation_mean"] == "0.0"
    assert repair_row["regret_vs_none_mean"] != ""
    assert any(row["case_group"] == "overall" for row in case_group_rows)


def test_mechanism_and_holdout_manifests_load() -> None:
    manifest_names = [
        "tsp_mechanism_isolation_study",
        "tsp_mechanism_confirm_seedblock_b",
        "tsp_budget_sensitivity_study",
        "tsp_delayed_trigger_study",
        "tsp_periodic_refresh_control_study",
        "tsp_delayed_trigger_confirm_holdout",
        "tsp_budget_sensitivity_delayed_trigger",
        "zdt1_profile_holdout_study",
        "zdt1_budget_sensitivity_study",
        "zdt1_threshold_budget_rule_study",
        "zdt1_threshold_holdout_confirm_study",
        "knapsack_feasibility_mutation_study",
        "knapsack_feasibility_confirm_study",
        "knapsack_feasibility_tiny_confirm",
        "onemax_control_mechanism_check",
        "onemax_control_delayed_trigger_check",
        "tsp_hardcase_trigger_suite",
        "tsp_hardcase_holdout_suite",
        "zdt1_budget_note_freeze",
        "onemax_control_hardcase_check",
        "tsp_instance_feature_labeling_study",
        "tsp_policy_router_train_study",
        "tsp_policy_router_holdout_study",
        "tsp_policy_router_budget_band_study",
        "tsp_regime_switching_coarse",
        "tsp_regime_switching_holdout",
        "tsp_regime_switching_budget_band",
        "tsp_underbudget_rescue_suite",
        "tsp_underbudget_rescue_holdout",
        "tsp_underbudget_anticase_check",
        "zdt1_default_note_recheck",
        "onemax_control_router_check",
        "onemax_control_switching_check",
        "tsp_seed_fraction_study",
        "tsp_mutation_operator_study",
        "tsp_fixed_stack_coarse",
        "tsp_fixed_stack_holdout",
        "knapsack_rerun_boundary_suite",
        "knapsack_rerun_boundary_holdout",
        "knapsack_repair_vs_restart_confirm",
        "tsp_seed_fraction_ablation_study",
        "tsp_seed_source_ablation_study",
        "tsp_mutation_operator_ablation_study",
        "tsp_canonical_default_confirm",
        "zdt1_default_freeze_recheck",
        "zdt1_threshold_anatomy_study",
        "zdt1_refresh_anatomy_study",
        "zdt1_cooldown_anatomy_study",
        "zdt1_canonical_default_confirm",
        "tsp_budget_frontier_study",
        "tsp_fast_profile_confirm",
        "zdt1_budget_frontier_study",
        "zdt1_fast_profile_confirm",
        "knapsack_rerun_gate_efficiency_study",
        "knapsack_rerun_gate_confirm",
        "onemax_control_budget_check",
        "tsp_default_tiny_freeze_check",
        "onemax_control_tiny_check",
        "onemax_control_fixedstack_check",
    ]

    loaded = [load_local_study(name) for name in manifest_names]
    loaded_by_name = {study.study_name: study for study in loaded}

    assert [study.study_name for study in loaded] == manifest_names
    assert loaded[0].problem == "tsp"
    assert loaded[1].primary_metric == "best_route_distance"
    assert loaded[3].problem == "tsp"
    assert loaded[7].problem == "zdt1"
    assert loaded[11].problem == "knapsack"
    assert loaded_by_name["onemax_control_router_check"].problem == "onemax"
    assert len(loaded_by_name["tsp_hardcase_trigger_suite"].cases) == 3
    assert len(loaded_by_name["tsp_hardcase_holdout_suite"].cases) == 2
    assert loaded_by_name["zdt1_budget_note_freeze"].primary_metric == "hypervolume"
    assert loaded_by_name["tsp_policy_router_train_study"].problem == "tsp"
    assert loaded_by_name["zdt1_default_note_recheck"].problem == "zdt1"
    assert loaded_by_name["tsp_regime_switching_holdout"].problem == "tsp"
    assert loaded_by_name["tsp_underbudget_rescue_holdout"].problem == "tsp"
    assert loaded_by_name["tsp_underbudget_rescue_holdout"].cases[0].group == "rescue_target"
    assert loaded_by_name["tsp_underbudget_rescue_holdout"].cases[-1].group == "anti_case"
    assert loaded_by_name["onemax_control_switching_check"].problem == "onemax"
    fixed_stack_coarse = loaded_by_name["tsp_fixed_stack_coarse"]
    assert fixed_stack_coarse.variant_overrides["fixed_swap_seed50"]["algorithm"] == "hybrid_ga"
    assert loaded_by_name["tsp_fixed_stack_holdout"].cases[-1].group == "anti_case"
    assert loaded_by_name["knapsack_rerun_boundary_suite"].problem == "knapsack"
    assert loaded_by_name["knapsack_rerun_boundary_suite"].cases[0].group == "uncorrelated_small"
    assert (
        loaded_by_name["knapsack_rerun_boundary_holdout"].cases[-1].group
        == "weakly_correlated_small"
    )
    assert loaded_by_name["knapsack_repair_vs_restart_confirm"].variant_overrides[
        "feasibility_aware_mutation_v1"
    ]["algorithm"] == "ga"
    assert loaded_by_name["tsp_seed_fraction_ablation_study"].problem == "tsp"
    assert loaded_by_name["tsp_seed_source_ablation_study"].variant_overrides[
        "hybrid_swap_noseed"
    ]["algorithm"] == "hybrid_ga"
    assert loaded_by_name["tsp_mutation_operator_ablation_study"].variant_overrides[
        "seeded_inversion_seed25"
    ]["mutation"] == "inversion"
    assert loaded_by_name["tsp_canonical_default_confirm"].cases[0].group == "rescue_target"
    assert loaded_by_name["zdt1_default_freeze_recheck"].problem == "zdt1"
    assert loaded_by_name["zdt1_threshold_anatomy_study"].problem == "zdt1"
    assert loaded_by_name["zdt1_refresh_anatomy_study"].problem == "zdt1"
    assert loaded_by_name["zdt1_cooldown_anatomy_study"].problem == "zdt1"
    assert loaded_by_name["zdt1_canonical_default_confirm"].problem == "zdt1"
    assert loaded_by_name["tsp_budget_frontier_study"].problem == "tsp"
    assert loaded_by_name["tsp_fast_profile_confirm"].cases[-1].group == "anti_case"
    assert loaded_by_name["zdt1_budget_frontier_study"].primary_metric == "hypervolume"
    assert loaded_by_name["zdt1_fast_profile_confirm"].problem == "zdt1"
    assert loaded_by_name["knapsack_rerun_gate_efficiency_study"].problem == "knapsack"
    assert loaded_by_name["knapsack_rerun_gate_confirm"].cases[0].group == "boundary_like"
    assert loaded_by_name["onemax_control_budget_check"].problem == "onemax"
    assert loaded_by_name["tsp_default_tiny_freeze_check"].cases[0].group == "rescue_target"
    assert loaded_by_name["onemax_control_tiny_check"].problem == "onemax"
    assert loaded_by_name["onemax_control_fixedstack_check"].problem == "onemax"


def test_local_profile_loads() -> None:
    profile_path = _project_root() / "configs" / "local_profiles" / "zdt1_diversity_injection.json"
    config = load_config(profile_path)

    assert config.problem == "zdt1"
    assert config.algorithm == "nsga2"
    assert config.population_size == 60
    assert config.generations == 80
    assert config.algorithm_options["adaptive_policy"] == "low_diversity_injection"
    assert config.algorithm_options["diversity_threshold"] == 0.55
    assert config.algorithm_options["refresh_fraction"] == 0.1

    tsp_profile = load_config(
        _project_root() / "configs" / "local_profiles" / "tsp_diversity_injection.json"
    )
    assert tsp_profile.problem == "tsp"
    assert tsp_profile.algorithm_options["adaptive_policy"] == "low_diversity_injection"
    assert tsp_profile.algorithm_options["diversity_threshold"] == 0.15
    assert tsp_profile.algorithm_options["refresh_fraction"] == 0.05

    knapsack_profile = load_config(
        _project_root() / "configs" / "local_profiles" / "knapsack_restart_experimental.json"
    )
    assert knapsack_profile.problem == "knapsack"
    assert knapsack_profile.algorithm_options["adaptive_policy"] == "stagnation_restart"
    assert knapsack_profile.algorithm_options["stagnation_window"] == 5

    knapsack_repair_profile = load_config(
        _project_root() / "configs" / "local_profiles" / "knapsack_repair_local_experimental.json"
    )
    assert knapsack_repair_profile.problem == "knapsack"
    assert knapsack_repair_profile.algorithm == "hybrid_ga"
    assert knapsack_repair_profile.algorithm_options["repair_strategy"] == "knapsack_greedy_fill"
    assert knapsack_repair_profile.algorithm_options["seed_fraction"] == 0.0

    tsp_seeded_profile = load_config(
        _project_root() / "configs" / "local_profiles" / "tsp_seeded_swap_local.json"
    )
    assert tsp_seeded_profile.problem == "tsp"
    assert tsp_seeded_profile.algorithm == "hybrid_ga"
    assert tsp_seeded_profile.mutation == "swap"
    assert tsp_seeded_profile.algorithm_options["init_strategy"] == "tsp_nearest_neighbor_mix"
    assert tsp_seeded_profile.algorithm_options["seed_fraction"] == 0.5
    assert tsp_seeded_profile.algorithm_options["local_search_strategy"] == "none"

    tsp_fast_profile = load_config(
        _project_root() / "configs" / "local_profiles" / "tsp_seeded_swap_local_fast.json"
    )
    assert tsp_fast_profile.problem == "tsp"
    assert tsp_fast_profile.population_size == 30
    assert tsp_fast_profile.generations == 45

    zdt_fast_profile = load_config(
        _project_root() / "configs" / "local_profiles" / "zdt1_diversity_injection_fast.json"
    )
    assert zdt_fast_profile.problem == "zdt1"
    assert zdt_fast_profile.population_size == 45
    assert zdt_fast_profile.algorithm_options["refresh_fraction"] == 0.1


def test_tsp_budget_frontier_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "tsp_budget_frontier_smoke.json",
        {
            "study_name": "tsp_budget_frontier_smoke",
            "description": "Tiny TSP budget frontier smoke.",
            "problem": "tsp",
            "base_preset": "tsp_small",
            "shared_overrides": {"log_every": 1},
            "variant_overrides": {
                "current_default": {
                    "algorithm": "hybrid_ga",
                    "mutation": "swap",
                    "population_size": 20,
                    "generations": 12,
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                    },
                },
                "early_stop_plateau": {
                    "algorithm": "hybrid_ga",
                    "mutation": "swap",
                    "population_size": 20,
                    "generations": 12,
                    "algorithm_options": {
                        "adaptive_policy": "none",
                        "init_strategy": "tsp_nearest_neighbor_mix",
                        "seed_fraction": 0.5,
                        "local_search_strategy": "none",
                        "early_stop_policy": "progress_plateau",
                        "early_stop_window": 3,
                        "early_stop_min_generation": 4,
                        "early_stop_epsilon": 100.0,
                    },
                },
            },
            "sweep": {"study_variant": ["current_default", "early_stop_plateau"]},
            "seeds": [1],
            "budget_ceiling": 280,
            "primary_metric": "best_route_distance",
            "plotting": {"history_metric": "best_route_distance"},
            "runtime_budget_note": "Tiny budget frontier smoke.",
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
    early_row = next(row for row in summary_rows if row["study_variant"] == "early_stop_plateau")

    assert Path(str(payload["plots"]["plot_budget_vs_runtime"])).exists()
    assert Path(str(payload["plots"]["plot_budget_vs_regret"])).exists()
    assert Path(str(payload["plots"]["plot_early_stop_vs_quality"])).exists()
    assert float(early_row["early_stop_trigger_rate"]) >= 1.0
    assert float(early_row["actual_evaluations_used_mean"]) < float(early_row["configured_budget"])


def test_knapsack_rerun_gate_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "knapsack_rerun_gate_smoke.json",
        {
            "study_name": "knapsack_rerun_gate_smoke",
            "description": "Tiny knapsack rerun gate smoke.",
            "problem": "knapsack",
            "base_preset": "knapsack_small",
            "cases": [
                {
                    "case_id": "tight_smoke",
                    "group": "boundary_like",
                    "note": "tiny tight-capacity smoke",
                    "overrides": {
                        "genome_length": 12,
                        "problem_options": {
                            "num_items": 12,
                            "instance_name": "tight_smoke",
                            "instance_source": "test_knapsack_rerun_gate",
                            "weights": [2, 3, 5, 7, 11, 13, 4, 6, 8, 9, 10, 12],
                            "values": [3, 4, 9, 13, 22, 25, 7, 10, 13, 14, 16, 18],
                            "capacity": 28.0,
                            "penalty_factor": 20.0
                        }
                    }
                }
            ],
            "shared_overrides": {
                "population_size": 20,
                "generations": 20,
                "log_every": 1,
                "mutation_rate": 0.02
            },
            "variant_overrides": {
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
                        "gate_initial_feasible_threshold": 0.5,
                        "gate_first_feasible_generation_limit": 3
                    }
                }
            },
            "sweep": {"study_variant": ["repair_only", "repair_rerun_gate"]},
            "seeds": [1],
            "budget_ceiling": 440,
            "primary_metric": "best_feasible_fitness",
            "plotting": {"history_metric": "best_fitness"},
            "runtime_budget_note": "Tiny rerun gate smoke."
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
    gate_row = next(row for row in summary_rows if row["study_variant"] == "repair_rerun_gate")

    assert Path(str(payload["plots"]["plot_rerun_gate_vs_regret"])).exists()
    assert Path(str(payload["plots"]["plot_initial_feasible_fraction_vs_rerun_value"])).exists()
    assert float(gate_row["rerun_trigger_rate"]) >= 1.0
    assert float(gate_row["total_actual_evaluations_used_mean"]) >= float(
        gate_row["actual_evaluations_used_mean"]
    )


def test_adaptive_docs_reference_real_commands() -> None:
    readme = (_project_root() / "README.md").read_text(encoding="utf-8")
    guide = (_project_root() / "docs" / "local_experiment_guide.md").read_text(encoding="utf-8")
    examples = (_project_root() / "examples" / "README.md").read_text(encoding="utf-8")

    assert "adaptive" in guide.lower()
    assert "plot_diversity.png" in guide
    assert "plot_stagnation.png" in guide
    assert "plot_trigger_events.png" in guide
    assert "plot_post_trigger_gain.png" in guide
    assert "plot_refresh_schedule_vs_gain.png" in guide
    assert "plot_refresh_volume_vs_gain.png" in guide
    assert "plot_collapse_onset_vs_trigger.png" in guide
    assert "plot_budget_band_vs_gap.png" in guide
    assert "plot_instance_feature_map.png" in guide
    assert "plot_feature_vs_policy_gap.png" in guide
    assert "plot_bridge_score_vs_trigger_value.png" in guide
    assert "plot_anisotropy_vs_decay_value.png" in guide
    assert "plot_router_regret.png" in guide
    assert "plot_budget_band_vs_policy_win.png" in guide
    assert "plot_mode_switch_timeline.png" in guide
    assert "plot_diversity_vs_mode.png" in guide
    assert "plot_collapse_onset_vs_switch.png" in guide
    assert "plot_regret_vs_policy.png" in guide
    assert "plot_budget_band_vs_regret.png" in guide
    assert "plot_rescue_target_vs_anticase_gap.png" in guide
    assert "plot_seed_fraction_vs_gap.png" in guide
    assert "plot_seed_source_vs_gap.png" in guide
    assert "plot_mutation_operator_vs_gap.png" in guide
    assert "plot_initial_quality_vs_final_gap.png" in guide
    assert "plot_family_vs_regret.png" in guide
    assert "plot_seeded_vs_repair_gap.png" in guide
    assert "plot_repair_vs_greedy_gap.png" in guide
    assert "plot_init_feasible_vs_final_gain.png" in guide
    assert "plot_initial_feasible_fraction_vs_gain.png" in guide
    assert "plot_capacity_tightness_vs_gain.png" in guide
    assert "plot_budget_vs_feasible_gain.png" in guide
    assert "plot_budget_vs_runtime.png" in guide
    assert "plot_budget_vs_regret.png" in guide
    assert "plot_early_stop_vs_quality.png" in guide
    assert "plot_budget_vs_spread.png" in guide
    assert "plot_early_stop_vs_hv.png" in guide
    assert "plot_rerun_gate_vs_regret.png" in guide
    assert "plot_initial_feasible_fraction_vs_rerun_value.png" in guide
    assert "plot_mutation_rate.png" in guide
    assert "plot_diversity_vs_distance.png" in guide
    assert "plot_hv_vs_spread.png" in guide
    assert "plot_budget_vs_hv.png" in guide
    assert "plot_threshold_vs_hv.png" in guide
    assert "plot_refresh_vs_hv.png" in guide
    assert "plot_cooldown_vs_hv.png" in guide
    assert "plot_violation.png" in guide
    assert "run_local_sweep.py --study" in readme
    assert "run_tsp_router_analysis.py" in readme
    assert "tsp_hardcase_trigger_suite" in readme
    assert "tsp_hardcase_holdout_suite" in guide
    assert "zdt1_budget_note_freeze" in guide
    assert "tsp_policy_router_train_study" in guide
    assert "tsp_policy_router_holdout_study" in readme
    assert "tsp_policy_router_budget_band_study" in examples
    assert "tsp_regime_switching_coarse" in guide
    assert "tsp_regime_switching_holdout" in readme
    assert "tsp_regime_switching_budget_band" in examples
    assert "tsp_underbudget_rescue_suite" in guide
    assert "tsp_underbudget_rescue_holdout" in readme
    assert "tsp_underbudget_anticase_check" in examples
    assert "zdt1_default_freeze_recheck" in guide
    assert "zdt1_threshold_anatomy_study" in guide
    assert "zdt1_refresh_anatomy_study" in readme
    assert "zdt1_cooldown_anatomy_study" in examples
    assert "zdt1_canonical_default_confirm" in guide
    assert "knapsack_family_suite" in guide
    assert "knapsack_seeded_repair_anatomy_study" in readme
    assert "knapsack_feasibility_control_study" in examples
    assert "knapsack_canonical_experimental_confirm" in guide
    assert "knapsack_rerun_boundary_suite" in guide
    assert "knapsack_rerun_boundary_holdout" in readme
    assert "knapsack_repair_vs_restart_confirm" in examples
    assert "tsp_budget_frontier_study" in readme
    assert "tsp_fast_profile_confirm" in guide
    assert "zdt1_budget_frontier_study" in examples
    assert "zdt1_fast_profile_confirm" in guide
    assert "knapsack_rerun_gate_efficiency_study" in readme
    assert "knapsack_rerun_gate_confirm" in guide
    assert "onemax_control_budget_check" in examples
    assert "tsp_default_tiny_freeze_check" in examples
    assert "onemax_control_tiny_check" in guide
    assert "onemax_control_router_check" in examples
    assert "onemax_control_switching_check" in examples
    assert "onemax_control_hardcase_check" in examples
    assert "tsp_seed_fraction_ablation_study" in examples
    assert "tsp_mutation_operator_ablation_study" in examples
    assert "tsp_fixed_stack_coarse" in guide
    assert "tsp_canonical_default_confirm" in readme
    assert "tsp_seed_fraction_ablation_study" in readme
    assert "tsp_seed_source_ablation_study" in guide
    assert "tsp_mutation_operator_ablation_study" in examples
    assert "tsp_canonical_default_confirm" in readme
    assert "zdt1_default_freeze_recheck" in guide
    assert "onemax_control_fixedstack_check" in guide
    assert "configs/local_profiles/tsp_diversity_injection.json" in guide
    assert "configs/local_profiles/tsp_seeded_swap_local.json" in readme
    assert "configs/local_profiles/tsp_seeded_swap_local_fast.json" in readme
    assert "configs/local_profiles/zdt1_diversity_injection.json" in guide
    assert "configs/local_profiles/zdt1_diversity_injection_fast.json" in readme
    assert "0.55 / 0.10 / 4" in guide
    assert "configs/local_profiles/knapsack_restart_experimental.json" in examples
    assert "configs/local_profiles/knapsack_repair_local_experimental.json" in guide
    assert "run_local_sweep.py --study" in examples
