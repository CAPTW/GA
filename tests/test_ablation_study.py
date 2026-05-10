from __future__ import annotations

import json

from ga_lab.experiment.ablation_study import _paired_test, _rank_biserial, run_manifests


def test_ablation_stats_helpers() -> None:
    differences = [1.0] * 10
    test_name, p_value = _paired_test(differences)

    assert test_name in {"wilcoxon_signed_rank", "sign_test"}
    assert p_value < 0.05
    assert _rank_biserial(differences) == 1.0


def test_run_ablation_manifests_smoke(tmp_path) -> None:
    manifest_path = tmp_path / "ablation_smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "ablation_smoke",
                "suite_kind": "comparison",
                "default_seeds": 1,
                "entries": [
                    {
                        "entry_id": "onemax_smoke",
                        "problem": "onemax",
                        "size": 32,
                        "preset": "configs/presets/onemax_small.json",
                        "seed_start": 9511,
                        "methods": [
                            {
                                "label": "pure_ga",
                                "kind": "base_preset",
                                "family": "ga",
                            },
                            {
                                "label": "hill_climb",
                                "kind": "baseline",
                                "family": "hill_climb",
                            },
                        ],
                        "comparisons": [
                            {
                                "comparison_id": "onemax_smoke_hill_vs_ga_success",
                                "left": "hill_climb",
                                "right": "pure_ga",
                                "metric": "success_to_target",
                                "objective": "max",
                            }
                        ],
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
        summary_stem="ablation_smoke_summary",
    )

    assert summary["comparison_rows"]
    assert summary["classification_rows"]
    assert (tmp_path / "ablation_smoke_summary.json").exists()
    assert (tmp_path / "ablation_smoke_summary.csv").exists()
    assert (tmp_path / "ablation_smoke_summary.md").exists()
