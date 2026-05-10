from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.tsp_instance_features import extract_tsp_instance_features
from ga_lab.tsp_policy_router import TSPRoutingRule


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


def test_tsp_instance_features_separate_bridge_and_corridor_shapes() -> None:
    bridge = extract_tsp_instance_features(
        "bridge",
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [-4.0, 4.0],
            [5.0, -3.0],
            [60.0, 0.0],
            [64.0, 4.0],
            [58.0, -4.0],
            [30.0, 14.0],
        ],
    )
    corridor = extract_tsp_instance_features(
        "corridor",
        [
            [0.0, 0.0],
            [8.0, 2.0],
            [16.0, -1.0],
            [24.0, 3.0],
            [32.0, -2.0],
            [40.0, 4.0],
            [48.0, -3.0],
            [24.0, 15.0],
        ],
    )

    assert bridge.bridge_score > corridor.bridge_score
    assert corridor.radial_distance_cv > bridge.radial_distance_cv


def test_router_rule_prediction_prefers_trigger_decay_and_none() -> None:
    rule = TSPRoutingRule(
        trigger_min_cities=15,
        trigger_bridge_threshold=3.0,
        trigger_bridge_budget_mode="any",
        trigger_ring_anisotropy_max=2.0,
        trigger_ring_nn_cv_max=0.1,
        decay_anisotropy_min=10.0,
        decay_bridge_max=1.8,
    )

    assert (
        rule.predict(
            {
                "num_cities": 18,
                "budget_band": "reduced",
                "bridge_score": 3.5,
                "pca_anisotropy_ratio": 7.0,
                "nn_distance_cv": 0.4,
            }
        )
        == "low_diversity_injection"
    )
    assert (
        rule.predict(
            {
                "num_cities": 18,
                "budget_band": "reduced",
                "bridge_score": 1.4,
                "pca_anisotropy_ratio": 12.0,
                "nn_distance_cv": 0.2,
            }
        )
        == "decay_mutation"
    )
    assert (
        rule.predict(
            {
                "num_cities": 12,
                "budget_band": "reduced",
                "bridge_score": 2.1,
                "pca_anisotropy_ratio": 5.0,
                "nn_distance_cv": 0.25,
            }
        )
        == "none"
    )


def test_router_analysis_smoke(tmp_path: Path) -> None:
    train_manifest = tmp_path / "tsp_router_train_smoke.json"
    holdout_manifest = tmp_path / "tsp_router_holdout_smoke.json"

    train_manifest.write_text(
        json.dumps(
            {
                "study_name": "tsp_router_train_smoke",
                "description": "Tiny train study for router smoke.",
                "problem": "tsp",
                "base_preset": "tsp_small",
                "cases": [
                    {
                        "case_id": "bridge_case",
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
                                    [15.0, 10.0],
                                ],
                            },
                        },
                    },
                    {
                        "case_id": "corridor_case",
                        "note": "tiny corridor case",
                        "overrides": {
                            "genome_length": 10,
                            "problem_options": {
                                "num_cities": 10,
                                "coordinates": [
                                    [0.0, 0.0],
                                    [6.0, 2.0],
                                    [12.0, -1.0],
                                    [18.0, 3.0],
                                    [24.0, -2.0],
                                    [30.0, 4.0],
                                    [36.0, -3.0],
                                    [42.0, 5.0],
                                    [48.0, -4.0],
                                    [24.0, 14.0],
                                ],
                            },
                        },
                    },
                ],
                "shared_overrides": {
                    "population_size": 20,
                    "algorithm_options": {
                        "diversity_threshold": 0.15,
                        "refresh_fraction": 0.05,
                        "adaptation_cooldown": 4,
                        "decay_end_rate": 0.02,
                    },
                },
                "sweep": {
                    "algorithm_options.adaptive_policy": [
                        "none",
                        "low_diversity_injection",
                        "decay_mutation",
                    ],
                    "generations": [12, 18],
                },
                "seeds": [1],
                "budget_ceiling": 400,
                "primary_metric": "best_route_distance",
                "plotting": {"history_metric": "best_route_distance"},
                "runtime_budget_note": "Tiny train smoke.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    holdout_manifest.write_text(
        train_manifest.read_text(encoding="utf-8").replace("train", "holdout"),
        encoding="utf-8",
    )

    train_payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(train_manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )
    holdout_payload = _run_json_command(
        str(_project_root() / "scripts" / "run_local_sweep.py"),
        "--study",
        str(holdout_manifest),
        "--output-root",
        str(tmp_path / "outputs"),
    )
    router_payload = _run_json_command(
        str(_project_root() / "scripts" / "run_tsp_router_analysis.py"),
        "--train-study-dir",
        str(train_payload["study_dir"]),
        "--holdout-study-dir",
        str(holdout_payload["study_dir"]),
        "--output-root",
        str(tmp_path / "outputs"),
    )

    assert Path(str(router_payload["summary_csv"])).exists()
    assert Path(str(router_payload["summary_md"])).exists()
    assert Path(str(router_payload["router_decision_table"])).exists()
    assert Path(str(router_payload["plots"]["plot_instance_feature_map"])).exists()
    assert Path(str(router_payload["plots"]["plot_router_regret"])).exists()
