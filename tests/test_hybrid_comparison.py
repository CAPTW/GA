from __future__ import annotations

import json
from pathlib import Path

from ga_lab.config import GAConfig, load_config
from ga_lab.experiment.budget_baseline_comparison import (
    TrackedProblem,
    configured_evaluation_budget,
)
from ga_lab.experiment.hybrid_comparison import run_manifests
from ga_lab.factory import build_runtime_context
from ga_lab.utils.seed import make_rng


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_hybrid_ga_respects_budget_smoke() -> None:
    base = load_config(_project_root() / "configs" / "presets" / "knapsack_small.json")
    data = base.to_dict()
    data["algorithm"] = "hybrid_ga"
    data["algorithm_options"] = {
        "init_strategy": "knapsack_greedy_mix",
        "seed_fraction": 0.25,
        "repair_strategy": "knapsack_greedy_fill",
        "local_search_strategy": "knapsack_ratio_swap",
        "local_search_interval": 5,
        "local_search_candidates": 2,
        "local_search_steps": 6,
    }
    config = GAConfig.from_dict(data)
    runtime = build_runtime_context(config)
    tracked_problem = TrackedProblem(runtime.problem, config)

    summary, _history = runtime.algorithm_fn(
        config=config,
        problem=tracked_problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=make_rng(config.seed),
    )

    assert summary["configured_evaluation_budget"] == configured_evaluation_budget(config)
    assert summary["actual_evaluations_used"] <= summary["configured_evaluation_budget"]
    assert tracked_problem.evaluation_count == summary["actual_evaluations_used"]
    assert summary["hybrid_extra_evaluations"] >= 0


def test_run_hybrid_manifests_smoke(tmp_path) -> None:
    manifest_path = tmp_path / "hybrid_smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "hybrid_smoke",
                "suite_kind": "comparison",
                "default_seeds": 1,
                "entries": [
                    {
                        "problem": "knapsack",
                        "size": 20,
                        "preset": "configs/presets/knapsack_small.json",
                        "baselines": ["random_sampling", "greedy_local_search"],
                        "hybrids": [
                            {
                                "label": "hybrid_seeded_repair",
                                "overrides": {
                                    "algorithm": "hybrid_ga",
                                    "algorithm_options": {
                                        "init_strategy": "knapsack_greedy_mix",
                                        "seed_fraction": 0.25,
                                        "repair_strategy": "knapsack_greedy_fill",
                                    },
                                },
                            }
                        ],
                        "seed_start": 9101,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = run_manifests(
        [manifest_path],
        output_root=tmp_path,
        summary_stem="hybrid_smoke_summary",
    )

    labels = {
        row["label"]
        for row in summary["aggregate_rows"]
        if row["suite_kind"] == "comparison" and row["problem"] == "knapsack"
    }
    assert labels == {
        "recommended_preset",
        "random_sampling",
        "greedy_local_search",
        "hybrid_seeded_repair",
    }
