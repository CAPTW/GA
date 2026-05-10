from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.experiment.external_mo_comparators import run_internal_nsga2
from ga_lab.experiment.nsga2_candidate_suite import safe_artifact_path
from ga_lab.experiment.nsga2_candidate_variants import (
    apply_candidate_variant,
    candidate_j_h_lite_retry2,
    candidate_n_low_g_tail_mutation_light,
    candidate_variant_metadata,
)


def _base_config() -> GAConfig:
    return GAConfig(
        run_name="operator_quality_phase0_test",
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
        seed=91,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"hypervolume_reference_point": [1.05, 10.5]},
    )


def test_candidate_n_metadata_is_phase0_only() -> None:
    variant = candidate_n_low_g_tail_mutation_light()
    metadata = candidate_variant_metadata(variant)

    assert metadata["candidate_id"] == "candidate_n_low_g_tail_mutation_light"
    assert metadata["base_candidate_id"] == "candidate_j_h_lite_retry2"
    assert metadata["base_candidate"] == "candidate_j_h_lite_retry2"
    assert metadata["mechanism"] == "low_g_tail_mutation_light"
    assert metadata["default_changed"] is False
    assert metadata["promotion_status"] == "phase0_sanity"
    assert metadata["allowed_use"] == "phase0_sanity_only"
    assert metadata["disallowed_use"] == "default_replacement"
    assert metadata["zdt1_specific_warning"]
    assert metadata["candidate_parameters"]["zdt1_only"] is True


def test_candidate_n_apply_keeps_default_path_unchanged() -> None:
    base = _base_config()
    variant = candidate_n_low_g_tail_mutation_light()
    candidate_config = apply_candidate_variant(base, variant)

    assert base.crossover == "arithmetic"
    assert base.selection_options == {}
    assert base.algorithm_options == {"hypervolume_reference_point": [1.05, 10.5]}
    assert candidate_config.crossover == "uniform"
    assert candidate_config.algorithm_options["nsga2_duplicate_retry_count"] == 2
    assert candidate_config.algorithm_options["nsga2_duplicate_retry_mutation_scale"] == 1.5
    assert candidate_config.algorithm_options["nsga2_low_g_tail_mutation_light"] is True
    assert candidate_config.algorithm_options["nsga2_low_g_tail_zdt1_only"] is True


def test_default_nsga2_has_no_candidate_n_metadata(tmp_path: Path) -> None:
    result = run_internal_nsga2(_base_config(), seed=92, output_root=str(tmp_path))

    assert result.success is True
    assert "candidate_id" not in result.metadata
    assert "default_changed" not in result.metadata


def test_candidate_n_config_does_not_change_candidate_j_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidate_j = json.loads(
        (repo_root / "configs" / "candidates" / "nsga2_h_lite_candidate_j.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_n = json.loads(
        (
            repo_root
            / "configs"
            / "candidates"
            / "nsga2_low_g_tail_mutation_candidate_n.json"
        ).read_text(encoding="utf-8")
    )

    assert candidate_j["candidate_id"] == "candidate_j_h_lite_retry2"
    assert candidate_j["default_changed"] is False
    assert candidate_n["candidate_id"] == "candidate_n_low_g_tail_mutation_light"
    assert candidate_n["base_candidate"] == "candidate_j_h_lite_retry2"


def test_candidate_n_non_zdt1_path_is_safe_noop_like(tmp_path: Path) -> None:
    candidate_config = apply_candidate_variant(_base_config(), candidate_n_low_g_tail_mutation_light())
    candidate_config.problem = "zdt2"
    candidate_config.run_name = "candidate_n_non_zdt1_safety"

    result = run_internal_nsga2(candidate_config, seed=93, output_root=str(tmp_path))

    assert result.success is True
    stats = result.metadata["low_g_tail_mutation_stats"]
    assert stats["enabled"] is True
    assert stats["problem_applied"] == "zdt2"
    assert stats["skipped_non_zdt1_problem_count"] > 0


def test_operator_quality_phase0_runner_artifact_suffix_and_component_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "operator-artifacts"
    output_root = tmp_path / "operator-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_nsga2_operator_quality_phase0.py",
            "--problem",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            "19101",
            "--budget",
            "300",
            "--artifact-suffix",
            "candidate_n_test",
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

    assert results_json.name == "nsga2_operator_quality_phase0_results_candidate_n_test.json"
    assert report_md.name == "nsga2_operator_quality_phase0_report_candidate_n_test.md"
    assert "fairness_summary" in result_payload
    assert result_payload["fairness"]["status"] in {"pass", "warning", "fail"}
    assert result_payload["zdt1_component_rows"]
    assert result_payload["operator_supply_rows"]

    candidate_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"] == "candidate_n_low_g_tail_mutation_light"
    ]
    assert candidate_rows
    for row in candidate_rows:
        assert row["metadata"]["candidate_id"] == "candidate_n_low_g_tail_mutation_light"
        assert row["metadata"]["base_candidate_id"] == "candidate_j_h_lite_retry2"
        assert row["metadata"]["default_changed"] is False
        assert row["metadata"]["promotion_status"] == "phase0_sanity"
        assert row["requested_budget"] == 300
        assert row["actual_evaluations"] == 300
        assert row["zdt1_component_diagnostics_success"] is True
        assert row["success"] is True

    default_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert default_rows
    for row in default_rows:
        assert "candidate_id" not in row["metadata"]
        assert "default_changed" not in row["metadata"]


def test_operator_quality_phase0_safe_artifact_suffix_path(tmp_path: Path) -> None:
    path = safe_artifact_path(
        tmp_path,
        "nsga2_operator_quality_phase0_results",
        "candidate_n_phase0",
        ".json",
    )
    assert path.name == "nsga2_operator_quality_phase0_results_candidate_n_phase0.json"


def test_candidate_n_keeps_candidate_j_variant_unchanged() -> None:
    base = _base_config()
    candidate_j_config = apply_candidate_variant(base, candidate_j_h_lite_retry2())
    candidate_n_config = apply_candidate_variant(base, candidate_n_low_g_tail_mutation_light())

    assert "nsga2_low_g_tail_mutation_light" not in candidate_j_config.algorithm_options
    assert candidate_n_config.algorithm_options["nsga2_low_g_tail_mutation_light"] is True
