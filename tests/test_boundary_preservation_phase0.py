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
    candidate_m_boundary_preservation_light,
    candidate_variant_metadata,
)


def _base_config() -> GAConfig:
    return GAConfig(
        run_name="boundary_phase0_test",
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
        seed=71,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"hypervolume_reference_point": [1.05, 10.5]},
    )


def test_candidate_m_metadata_is_phase0_only() -> None:
    variant = candidate_m_boundary_preservation_light()
    metadata = candidate_variant_metadata(variant)

    assert metadata["candidate_id"] == "candidate_m_boundary_preservation_light"
    assert metadata["base_candidate_id"] == "candidate_j_h_lite_retry2"
    assert metadata["mechanism"] == "boundary_preservation_light"
    assert metadata["default_changed"] is False
    assert metadata["promotion_status"] == "phase0_sanity"
    assert metadata["allowed_use"] == "phase0_sanity_only"
    assert metadata["disallowed_use"] == "default_replacement"


def test_candidate_m_apply_keeps_default_path_unchanged() -> None:
    base = _base_config()
    variant = candidate_m_boundary_preservation_light()
    candidate_config = apply_candidate_variant(base, variant)

    assert base.crossover == "arithmetic"
    assert base.selection_options == {}
    assert base.algorithm_options == {"hypervolume_reference_point": [1.05, 10.5]}
    assert candidate_config.crossover == "uniform"
    assert candidate_config.algorithm_options["nsga2_duplicate_retry_count"] == 2
    assert candidate_config.algorithm_options["nsga2_duplicate_retry_mutation_scale"] == 1.5
    assert candidate_config.algorithm_options["nsga2_partial_front_strategy"] == "boundary_preservation_light"


def test_default_nsga2_has_no_candidate_m_metadata(tmp_path: Path) -> None:
    result = run_internal_nsga2(_base_config(), seed=72, output_root=str(tmp_path))

    assert result.success is True
    assert "candidate_id" not in result.metadata
    assert "default_changed" not in result.metadata


def test_candidate_m_config_does_not_change_candidate_j_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidate_j = json.loads(
        (repo_root / "configs" / "candidates" / "nsga2_h_lite_candidate_j.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_m = json.loads(
        (
            repo_root
            / "configs"
            / "candidates"
            / "nsga2_boundary_preservation_candidate_m.json"
        ).read_text(encoding="utf-8")
    )

    assert candidate_j["candidate_id"] == "candidate_j_h_lite_retry2"
    assert candidate_j["default_changed"] is False
    assert candidate_m["candidate_id"] == "candidate_m_boundary_preservation_light"
    assert candidate_m["base_candidate"] == "candidate_j_h_lite_retry2"


def test_boundary_phase0_runner_artifact_suffix_and_fairness_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "boundary-artifacts"
    output_root = tmp_path / "boundary-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_nsga2_boundary_phase0.py",
            "--problems",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            "11301",
            "--budget",
            "300",
            "--artifact-suffix",
            "boundary_test",
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

    assert results_json.name == "nsga2_boundary_preservation_phase0_results_boundary_test.json"
    assert report_md.name == "nsga2_boundary_preservation_phase0_report_boundary_test.md"
    assert "fairness_summary" in result_payload
    assert result_payload["fairness"]["status"] in {"pass", "warning", "fail"}

    candidate_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"] == "candidate_m_boundary_preservation_light"
    ]
    assert candidate_rows
    for row in candidate_rows:
        assert row["metadata"]["candidate_id"] == "candidate_m_boundary_preservation_light"
        assert row["metadata"]["default_changed"] is False
        assert row["metadata"]["base_candidate_id"] == "candidate_j_h_lite_retry2"
        assert row["requested_budget"] == 300
        assert row["actual_evaluations"] == 300

    default_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert default_rows
    for row in default_rows:
        assert "candidate_id" not in row["metadata"]
        assert "default_changed" not in row["metadata"]


def test_boundary_phase0_safe_artifact_suffix_path(tmp_path: Path) -> None:
    path = safe_artifact_path(
        tmp_path,
        "nsga2_boundary_preservation_phase0_results",
        "boundary_phase0_m",
        ".json",
    )
    assert path.name == "nsga2_boundary_preservation_phase0_results_boundary_phase0_m.json"
