from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.experiment.external_mo_comparators import (
    ExternalLibraryStatus,
    run_internal_nsga2,
    run_pymoo_nsga2,
)
from ga_lab.experiment.external_operator_parity import (
    ExternalOperatorParityConfig,
    summarize_final_decision_distribution,
    summarize_final_zdt1_component_distribution,
    summarize_operator_parameter_summary,
    summarize_parity_gap,
)
from ga_lab.problems.zdt1 import ZDT1Problem


def _base_config() -> GAConfig:
    return GAConfig(
        run_name="external_operator_parity_test",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=20,
        genome_length=6,
        generations=12,
        crossover_rate=0.9,
        mutation_rate=0.15,
        elitism=1,
        tournament_size=2,
        seed=61,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"hypervolume_reference_point": [1.05, 10.5]},
    )


def test_external_operator_parity_config_defaults_to_disabled() -> None:
    config = ExternalOperatorParityConfig()

    assert config.external_parity_trace_enabled is False
    assert config.segment_count == 6
    assert config.tail_low_threshold == 0.2
    assert config.low_g_threshold == 1.1


def test_final_decision_distribution_is_json_serializable() -> None:
    summary = summarize_final_decision_distribution(
        [
            [0.1, 0.2, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.1, 0.2, 0.2, 0.2, 0.2],
        ],
        tail_low_threshold=0.25,
    )

    assert summary["decision_count"] == 2
    assert summary["unique_decision_count"] == 2
    assert summary["tail_low_rate"] is not None
    json.dumps(summary)


def test_zdt1_component_summary_matches_repository_evaluator() -> None:
    problem = ZDT1Problem("zdt1")
    decision_vectors = [
        [0.05, 0.10, 0.10, 0.20, 0.20, 0.10],
        [0.15, 0.30, 0.20, 0.20, 0.25, 0.25],
    ]
    objective_vectors = [list(problem.fitness(vector)) for vector in decision_vectors]

    summary = summarize_final_zdt1_component_distribution(
        decision_vectors,
        objective_vectors,
        bins=6,
        low_g_threshold=1.5,
    )
    expected_g_values = [1.0 + 9.0 * (sum(vector[1:]) / len(vector[1:])) for vector in decision_vectors]

    assert abs(float(summary["g_mean"]) - (sum(expected_g_values) / len(expected_g_values))) < 1e-12
    assert summary["segment0_count"] >= 1
    assert summary["distance_mean"] is not None


def test_parity_gap_summary_includes_candidate_j_n_and_pymoo_values() -> None:
    rows = summarize_parity_gap(
        {
            "candidate_j_h_lite_retry2": {
                "g_mean": 3.8,
                "segment0_g_mean": 3.7,
                "segment0_distance_mean": 2.7,
                "segment0_low_g_count": 0.3,
                "occupied_bins": 8,
                "spacing": 0.12,
                "nondominated_count": 8.0,
                "hypervolume_2d": 0.61,
                "inverted_generational_distance": 0.11,
                "runtime_seconds": 0.8,
            },
            "candidate_n_low_g_tail_mutation_light": {
                "g_mean": 3.7,
                "segment0_g_mean": 3.6,
                "segment0_distance_mean": 2.6,
                "segment0_low_g_count": 0.45,
                "occupied_bins": 7,
                "spacing": 0.13,
                "nondominated_count": 7.5,
                "hypervolume_2d": 0.62,
                "inverted_generational_distance": 0.10,
                "runtime_seconds": 0.9,
            },
            "pymoo_nsga2": {
                "g_mean": 3.4,
                "segment0_g_mean": 3.2,
                "segment0_distance_mean": 2.3,
                "segment0_low_g_count": 0.8,
                "occupied_bins": 10,
                "spacing": 0.08,
                "nondominated_count": 10.0,
                "hypervolume_2d": 0.64,
                "inverted_generational_distance": 0.08,
                "runtime_seconds": 0.4,
            },
        }
    )

    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["g_mean"]["candidate_j"] == 3.8
    assert by_metric["g_mean"]["candidate_n"] == 3.7
    assert by_metric["g_mean"]["pymoo"] == 3.4
    assert "candidate_n_vs_j_delta" in by_metric["occupied_bins"]


def test_pymoo_missing_gracefully_skips(monkeypatch) -> None:
    import ga_lab.experiment.external_mo_comparators as comparators

    monkeypatch.setattr(
        comparators,
        "optional_library_status",
        lambda name: ExternalLibraryStatus(
            library=name,
            installed=False,
            version=None,
            reason=f"{name} unavailable for test",
        ),
    )

    result = run_pymoo_nsga2(_base_config(), seed=61, budget=80)

    assert result.status == "skipped"
    assert result.success is False
    assert "unavailable" in str(result.error_message)


def test_operator_parameter_summary_and_default_path_stay_separate(tmp_path: Path) -> None:
    result = run_internal_nsga2(_base_config(), seed=61, output_root=str(tmp_path / "baseline"))
    row = {
        "algorithm": result.algorithm_name,
        "requested_budget": result.requested_budget,
        "actual_evaluations": result.evaluations,
        "metadata": dict(result.metadata),
    }

    summary = summarize_operator_parameter_summary(row)

    assert result.success is True
    assert summary["algorithm"] == "internal_nsga2"
    assert "candidate_id" not in result.metadata
    assert "external_parity_trace_enabled" not in result.metadata


def test_external_operator_parity_runner_writes_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "parity-artifacts"
    output_root = tmp_path / "parity-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_external_operator_parity.py",
            "--problem",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            "23001",
            "--budget",
            "80",
            "--segment-count",
            "6",
            "--artifact-suffix",
            "op_parity_test",
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
    fairness_md = Path(str(payload["fairness_md"]))
    result_payload = json.loads(results_json.read_text(encoding="utf-8"))

    assert results_json.name == "nsga2_external_operator_parity_results_op_parity_test.json"
    assert report_md.name == "nsga2_external_operator_parity_report_op_parity_test.md"
    assert fairness_md.name == "nsga2_external_operator_parity_fairness_report_op_parity_test.md"
    assert result_payload["selected_problems"] == ["zdt1"]
    assert result_payload["segment_count"] == 6
    assert result_payload["external_parity_trace_enabled"] is True
    assert "fairness" in result_payload
    assert "fairness_summary" in result_payload
    assert result_payload["decision_distribution_rows"]
    assert result_payload["component_distribution_rows"]
    assert result_payload["objective_segment_rows"]
    assert result_payload["parity_gap_rows"]
    assert result_payload["operator_parameter_rows"]

    candidate_n_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"] == "candidate_n_low_g_tail_mutation_light"
    ]
    assert len(candidate_n_rows) == 1
    candidate_n_row = candidate_n_rows[0]
    assert candidate_n_row["metadata"]["candidate_id"] == "candidate_n_low_g_tail_mutation_light"
    assert candidate_n_row["metadata"]["default_changed"] is False
    assert "final_decision_distribution" in candidate_n_row
    assert "final_zdt1_component_distribution" in candidate_n_row
    assert "final_objective_segment_distribution" in candidate_n_row
    assert candidate_n_row["actual_evaluations"] == 80

    internal_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert len(internal_rows) == 1
    assert "candidate_id" not in internal_rows[0]["metadata"]
    assert "external_parity_trace_enabled" not in internal_rows[0]["metadata"]

    assert any(row["metric"] == "g_mean" for row in result_payload["parity_gap_rows"])
    issue_types = {issue["issue_type"] for issue in result_payload["fairness"]["issues"]}
    assert "external_operator_family_difference" in issue_types or "evaluation_budget_fail" in issue_types

    pymoo_rows = [row for row in result_payload["raw_rows"] if row["algorithm"] == "pymoo_nsga2"]
    assert len(pymoo_rows) == 1
    pymoo_row = pymoo_rows[0]
    if pymoo_row["success"]:
        assert pymoo_row["final_decision_distribution"]["decision_count"] >= 1
    else:
        assert pymoo_row["status"] in {"skipped", "failed"}
