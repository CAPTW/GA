from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.experiment.external_mo_comparators import run_internal_nsga2
from ga_lab.experiment.nsga2_candidate_variants import (
    apply_candidate_variant,
    candidate_n_low_g_tail_mutation_light,
    candidate_o_spread_preserving_variation_light,
    candidate_variant_metadata,
)


def _base_config() -> GAConfig:
    return GAConfig(
        run_name="spread_preserving_phase0_test",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=15,
        genome_length=6,
        generations=6,
        crossover_rate=0.9,
        mutation_rate=0.15,
        elitism=1,
        tournament_size=2,
        seed=101,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"hypervolume_reference_point": [1.05, 10.5]},
    )


def test_candidate_o_metadata_is_phase0_only() -> None:
    variant = candidate_o_spread_preserving_variation_light()
    metadata = candidate_variant_metadata(variant)

    assert metadata["candidate_id"] == "candidate_o_spread_preserving_variation_light"
    assert metadata["base_candidate_id"] == "candidate_n_low_g_tail_mutation_light"
    assert metadata["base_candidate"] == "candidate_n_low_g_tail_mutation_light"
    assert metadata["mechanism"] == "spread_preserving_variation_light"
    assert metadata["default_changed"] is False
    assert metadata["promotion_status"] == "phase0_sanity"
    assert metadata["allowed_use"] == "phase0_sanity_only"
    assert metadata["disallowed_use"] == "default_replacement"
    assert metadata["zdt1_specific_warning"]
    assert metadata["candidate_parameters"]["spread_preserving_probability"] == 0.25


def test_candidate_o_apply_keeps_default_path_unchanged() -> None:
    base = _base_config()
    variant = candidate_o_spread_preserving_variation_light()
    candidate_config = apply_candidate_variant(base, variant)

    assert base.crossover == "arithmetic"
    assert base.selection_options == {}
    assert base.algorithm_options == {"hypervolume_reference_point": [1.05, 10.5]}
    assert candidate_config.crossover == "uniform"
    assert candidate_config.algorithm_options["nsga2_low_g_tail_mutation_light"] is True
    assert candidate_config.algorithm_options["nsga2_spread_preserving_variation_light"] is True
    assert candidate_config.algorithm_options["nsga2_spread_preserving_zdt1_only"] is True


def test_default_nsga2_has_no_candidate_o_metadata(tmp_path: Path) -> None:
    result = run_internal_nsga2(_base_config(), seed=102, output_root=str(tmp_path))

    assert result.success is True
    assert "candidate_id" not in result.metadata
    assert "default_changed" not in result.metadata


def test_candidate_o_config_does_not_change_candidate_n_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidate_n = json.loads(
        (
            repo_root
            / "configs"
            / "candidates"
            / "nsga2_low_g_tail_mutation_candidate_n.json"
        ).read_text(encoding="utf-8")
    )
    candidate_o = json.loads(
        (
            repo_root
            / "configs"
            / "candidates"
            / "nsga2_spread_preserving_variation_candidate_o.json"
        ).read_text(encoding="utf-8")
    )

    assert candidate_n["candidate_id"] == "candidate_n_low_g_tail_mutation_light"
    assert candidate_n["base_candidate"] == "candidate_j_h_lite_retry2"
    assert candidate_o["candidate_id"] == "candidate_o_spread_preserving_variation_light"
    assert candidate_o["base_candidate"] == "candidate_n_low_g_tail_mutation_light"
    assert candidate_o["no_pymoo_operator_clone"] is True


def test_candidate_o_non_zdt1_path_is_safe_noop_like(tmp_path: Path) -> None:
    candidate_config = apply_candidate_variant(_base_config(), candidate_o_spread_preserving_variation_light())
    candidate_config.problem = "zdt2"
    candidate_config.run_name = "candidate_o_non_zdt1_safety"

    result = run_internal_nsga2(candidate_config, seed=103, output_root=str(tmp_path))

    assert result.success is True
    low_g_stats = result.metadata["low_g_tail_mutation_stats"]
    spread_stats = result.metadata["spread_preserving_variation_stats"]
    assert low_g_stats["enabled"] is True
    assert spread_stats["enabled"] is True
    assert low_g_stats["problem_applied"] == "zdt2"
    assert spread_stats["problem_applied"] == "zdt2"
    assert low_g_stats["skipped_non_zdt1_problem_count"] > 0
    assert spread_stats["skipped_non_zdt1_problem_count"] > 0


def test_spread_preserving_phase0_runner_artifact_suffix_and_summaries(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "spread-phase0-artifacts"
    output_root = tmp_path / "spread-phase0-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_nsga2_spread_preserving_phase0.py",
            "--problem",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            "25101",
            "--budget",
            "300",
            "--artifact-suffix",
            "candidate_o_test",
            "--artifact-root",
            str(artifact_root),
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    results_json = Path(str(payload["results_json"]))
    report_md = Path(str(payload["report_md"]))
    result_payload = json.loads(results_json.read_text(encoding="utf-8"))

    assert results_json.name == "nsga2_spread_preserving_phase0_results_candidate_o_test.json"
    assert report_md.name == "nsga2_spread_preserving_phase0_report_candidate_o_test.md"
    assert "fairness_summary" in result_payload
    assert result_payload["fairness"]["status"] in {"pass", "warning", "fail"}
    assert result_payload["spread_rows"]
    assert result_payload["zdt1_component_rows"]
    assert result_payload["operator_supply_rows"]

    candidate_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"] == "candidate_o_spread_preserving_variation_light"
    ]
    assert candidate_rows
    for row in candidate_rows:
        assert row["metadata"]["candidate_id"] == "candidate_o_spread_preserving_variation_light"
        assert row["metadata"]["base_candidate_id"] == "candidate_n_low_g_tail_mutation_light"
        assert row["metadata"]["default_changed"] is False
        assert row["metadata"]["promotion_status"] == "phase0_sanity"
        assert row["requested_budget"] == 300
        assert row["actual_evaluations"] == 300
        assert row["spread_parity_diagnostics_success"] is True
        assert row["zdt1_component_diagnostics_success"] is True
        assert row["success"] is True

    default_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert default_rows
    for row in default_rows:
        assert "candidate_id" not in row["metadata"]
        assert "default_changed" not in row["metadata"]


def test_candidate_o_keeps_candidate_n_variant_unchanged() -> None:
    base = _base_config()
    candidate_n_config = apply_candidate_variant(base, candidate_n_low_g_tail_mutation_light())
    candidate_o_config = apply_candidate_variant(base, candidate_o_spread_preserving_variation_light())

    assert "nsga2_spread_preserving_variation_light" not in candidate_n_config.algorithm_options
    assert candidate_o_config.algorithm_options["nsga2_spread_preserving_variation_light"] is True
