from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ga_lab.config import GAConfig
from ga_lab.experiment.comparison import (
    build_baseline_suite_summary,
    collect_comparison_rows,
)
from ga_lab.runner import run_experiment


def test_collect_comparison_rows_reads_standard_run_fields(tmp_path) -> None:
    config = GAConfig(
        run_name="comparison_onemax",
        problem="onemax",
        population_size=24,
        genome_length=16,
        generations=20,
        crossover_rate=0.9,
        mutation_rate=0.02,
        elitism=1,
        tournament_size=3,
        seed=17,
        maximize=True,
        target_fitness=16,
        log_every=1,
    )

    result = run_experiment(config, output_root=tmp_path)
    rows = collect_comparison_rows(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["run_name"] == "comparison_onemax"
    assert row["problem"] == "onemax"
    assert row["seed"] == 17
    assert row["selection"] == "tournament"
    assert row["generations"] == 20
    assert row["best_fitness"] == result.summary["best_fitness"]
    assert row["mean_fitness"] == result.summary["mean_fitness"]
    assert row["config_path"].endswith("config.json")
    assert row["output_dir"] == str(result.output_dir)


def test_build_baseline_suite_summary_groups_runs() -> None:
    rows = [
        {
            "baseline_label": "onemax",
            "run_name": "onemax_seed7",
            "problem": "onemax",
            "algorithm": "ga",
            "representation": "bit",
            "selection": "tournament",
            "crossover": "one_point",
            "mutation": "bit_flip",
            "seed": 7,
            "best_fitness": 64.0,
            "mean_fitness": 63.0,
            "final_generation": 29,
            "runtime_seconds": 0.1,
            "config_path": "configs/onemax_baseline.json",
            "stop_reason": "target_fitness_reached",
        },
        {
            "baseline_label": "onemax",
            "run_name": "onemax_seed8",
            "problem": "onemax",
            "algorithm": "ga",
            "representation": "bit",
            "selection": "tournament",
            "crossover": "one_point",
            "mutation": "bit_flip",
            "seed": 8,
            "best_fitness": 64.0,
            "mean_fitness": 62.5,
            "final_generation": 31,
            "runtime_seconds": 0.2,
            "config_path": "configs/onemax_baseline.json",
            "stop_reason": "target_fitness_reached",
        },
    ]

    summary = build_baseline_suite_summary(
        "core_baselines",
        "configs/baselines/manifest.json",
        rows,
    )

    assert summary["suite_name"] == "core_baselines"
    assert summary["baseline_count"] == 1
    assert summary["run_count"] == 2
    baseline = summary["baselines"][0]
    assert baseline["baseline_label"] == "onemax"
    assert baseline["seed_values"] == [7, 8]
    assert baseline["mean_best_fitness"] == 64.0
    assert baseline["mean_mean_fitness"] == 62.75
    assert baseline["median_best_fitness"] == 64.0
    assert baseline["stdev_mean_fitness"] == pytest.approx(0.3535533905932738)
    assert summary["comparison_groups"][0]["best_by_mean_best_fitness"] == "onemax"


def test_run_baselines_script_writes_suite_outputs(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "smoke_baselines",
                "entries": [
                    {
                        "label": "onemax_baseline",
                        "config": "configs/onemax_baseline.json",
                        "seeds": 2,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_baselines.py"),
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    suite_dirs = list(tmp_path.glob("*_smoke_baselines"))
    assert len(suite_dirs) == 1, completed.stdout
    suite_dir = suite_dirs[0]

    assert (suite_dir / "manifest.json").exists()
    assert (suite_dir / "RUNS.csv").exists()
    assert (suite_dir / "RUNS.jsonl").exists()
    assert (suite_dir / "SUMMARY.md").exists()
    assert (suite_dir / "RETENTION.md").exists()
    assert (suite_dir / "retention_plan.json").exists()
    assert (suite_dir / "onemax_baseline_grid_summary.json").exists()

    suite_summary = json.loads((suite_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert suite_summary["run_count"] == 2
    assert suite_summary["baseline_count"] == 1
    assert suite_summary["baselines"][0]["seed_values"] == [7, 8]

    retention_plan = json.loads((suite_dir / "retention_plan.json").read_text(encoding="utf-8"))
    assert retention_plan["totals"]["run_count"] == 2
    assert retention_plan["totals"]["keep_full_count"] >= 1


def test_run_baselines_script_supports_matrix_manifest_with_10_seed_operator_compare(
    tmp_path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "operator_compare_matrix",
                "matrix": {
                    "base_config": "configs/onemax_baseline.json",
                    "seed_start": 7,
                    "seeds": 10,
                    "label_template": "operator_{selection}",
                    "axes": {
                        "selection": ["tournament", "rank"],
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_baselines.py"),
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    suite_dir = next(tmp_path.glob("*_operator_compare_matrix"))
    suite_summary = json.loads((suite_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert suite_summary["run_count"] == 20
    assert suite_summary["baseline_count"] == 2
    assert sorted(baseline["baseline_label"] for baseline in suite_summary["baselines"]) == [
        "operator_rank",
        "operator_tournament",
    ]
    assert all(baseline["run_count"] == 10 for baseline in suite_summary["baselines"])
    assert (suite_dir / "generated_configs").exists()
    assert (suite_dir / "SUMMARY.md").exists()
    assert (suite_dir / "RETENTION.md").exists()
    assert (suite_dir / "operator_rank_grid_summary.json").exists()
    assert (suite_dir / "operator_tournament_grid_summary.json").exists()


def test_summarize_results_preserves_suite_metadata(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "metadata_suite",
                "entries": [
                    {
                        "label": "onemax_baseline",
                        "config": "configs/onemax_baseline.json",
                        "seeds": 2,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_baselines.py"),
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    suite_dir = next(tmp_path.glob("*_metadata_suite"))
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "summarize_results.py"),
            "--results-dir",
            str(suite_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with (suite_dir / "RUNS.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert {row["suite_name"] for row in rows} == {"metadata_suite"}
    assert {row["baseline_label"] for row in rows} == {"onemax_baseline"}


def test_summarize_results_uses_nested_suite_metadata_for_collection_summary(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "nightly_nested_suite",
                "entries": [
                    {
                        "label": "onemax_baseline",
                        "config": "configs/onemax_baseline.json",
                        "seeds": 2,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_baselines.py"),
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "summarize_results.py"),
            "--results-dir",
            str(tmp_path),
            "--collection-name",
            "nightly_root",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with (tmp_path / "RUNS.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert {row["suite_name"] for row in rows} == {"nightly_nested_suite"}
    assert {row["baseline_label"] for row in rows} == {"onemax_baseline"}
