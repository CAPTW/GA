from __future__ import annotations

import json
from pathlib import Path

from ga_lab.config import load_config
from ga_lab.experiment.budget_baseline_comparison import (
    configured_evaluation_budget,
    run_manifests,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_configured_evaluation_budget_for_ga_and_nsga2() -> None:
    onemax = load_config(_project_root() / "configs" / "presets" / "onemax_small.json")
    zdt1 = load_config(_project_root() / "configs" / "presets" / "zdt1_small.json")

    assert configured_evaluation_budget(onemax) == 40 * (50 + 2)
    assert configured_evaluation_budget(zdt1) == 80 * ((3 * 120) + 2)


def test_run_manifests_smoke(tmp_path) -> None:
    manifest_path = tmp_path / "baseline_smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "baseline_smoke",
                "suite_kind": "comparison",
                "default_seeds": 1,
                "entries": [
                    {
                        "problem": "onemax",
                        "size": 32,
                        "preset": "configs/presets/onemax_small.json",
                        "baselines": ["random_search", "hill_climb"],
                        "seed_start": 9001,
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
        summary_stem="baseline_smoke_summary",
    )

    labels = {
        row["label"]
        for row in summary["aggregate_rows"]
        if row["suite_kind"] == "comparison" and row["problem"] == "onemax"
    }
    assert labels == {"recommended_preset", "random_search", "hill_climb"}
