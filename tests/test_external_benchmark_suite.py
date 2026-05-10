from __future__ import annotations

import json

from ga_lab.benchmarks.external import (
    BENCHMARK_INSTANCES,
    load_benchmark_problem_overrides,
    write_metadata_file,
)
from ga_lab.experiment.external_benchmark_suite import run_manifests


def test_load_benchmark_problem_overrides_from_temp_cache(tmp_path) -> None:
    kplib_path = tmp_path / BENCHMARK_INSTANCES["kplib_uncorrelated_50_s000"]["cache_relpath"]
    kplib_path.parent.mkdir(parents=True, exist_ok=True)
    kplib_path.write_text("\n3\n10\n\n8 4\n7 5\n6 6\n", encoding="utf-8")

    tsp_path = tmp_path / BENCHMARK_INSTANCES["tsplib_ulysses22"]["cache_relpath"]
    tsp_path.parent.mkdir(parents=True, exist_ok=True)
    tsp_path.write_text(
        "\n".join(
            [
                "NAME: tiny3",
                "TYPE: TSP",
                "DIMENSION: 3",
                "EDGE_WEIGHT_TYPE: EUC_2D",
                "NODE_COORD_SECTION",
                "1 0 0",
                "2 3 0",
                "3 3 4",
                "EOF",
            ]
        ),
        encoding="utf-8",
    )

    knapsack_overrides = load_benchmark_problem_overrides(
        "kplib_uncorrelated_50_s000",
        cache_root=tmp_path,
    )
    tsp_overrides = load_benchmark_problem_overrides("tsplib_ulysses22", cache_root=tmp_path)

    assert knapsack_overrides["genome_length"] == 3
    assert tsp_overrides["genome_length"] == 3
    assert tsp_overrides["problem_options"]["distance_matrix"][0][1] == 3.0


def test_run_external_manifest_smoke(tmp_path) -> None:
    manifest_path = tmp_path / "external_smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "external_smoke",
                "default_seeds": 1,
                "entries": [
                    {
                        "entry_id": "leading_ones_smoke",
                        "problem": "onemax",
                        "size": 32,
                        "preset": "configs/presets/onemax_small.json",
                        "instance_or_family": "leading_ones_32",
                        "benchmark_source": "synthetic_bitstring",
                        "validated_internal_range": "32 / 64 / 128",
                        "benchmark": {
                            "synthetic_family": "leading_ones"
                        },
                        "seed_start": 9100,
                        "methods": [
                            {
                                "label": "official_pure_ga",
                                "kind": "base_preset",
                                "solver_family": "pure-ga"
                            },
                            {
                                "label": "hill_climb",
                                "kind": "baseline",
                                "family": "hill_climb",
                                "solver_family": "baseline"
                            }
                        ],
                        "comparisons": [
                            {
                                "comparison_id": "smoke_hill_vs_ga_evals",
                                "left": "hill_climb",
                                "right": "official_pure_ga",
                                "metric": "evaluations_to_target",
                                "objective": "min"
                            }
                        ]
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = run_manifests(
        [manifest_path],
        output_root=tmp_path,
        summary_stem="external_smoke_summary",
        cache_root=tmp_path,
    )

    assert summary["comparison_rows"]
    assert summary["claim_rows"]
    assert (tmp_path / "external_smoke_summary.json").exists()
    assert (tmp_path / "external_smoke_summary.csv").exists()
    assert (tmp_path / "external_smoke_summary.md").exists()


def test_write_metadata_file(tmp_path) -> None:
    metadata_path = write_metadata_file(tmp_path / "metadata.json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert "sources" in payload
    assert "instances" in payload
