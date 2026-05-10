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
from ga_lab.experiment.nsga2_diagnostics import compute_zdt1_components
from ga_lab.experiment.spread_parity_diagnostics import (
    SpreadParityConfig,
    summarize_decision_to_segment_mapping,
    summarize_nondominated_distribution,
    summarize_occupancy_uniformity,
    summarize_parity_spread_gap,
    summarize_segment_allocation,
    summarize_segment_spacing_contribution,
)
from ga_lab.problems.zdt1 import ZDT1Problem


def _base_config() -> GAConfig:
    return GAConfig(
        run_name="spread_parity_test",
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
        seed=73,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"hypervolume_reference_point": [1.05, 10.5]},
    )


def test_spread_parity_config_defaults_to_disabled() -> None:
    config = SpreadParityConfig()

    assert config.spread_parity_trace_enabled is False
    assert config.segment_count == 6
    assert config.low_g_threshold == 1.1


def test_segment_allocation_summary_is_json_serializable() -> None:
    problem = ZDT1Problem("zdt1")
    decisions = [
        [0.05, 0.10, 0.10, 0.20, 0.20, 0.10],
        [0.35, 0.20, 0.20, 0.20, 0.25, 0.30],
        [0.75, 0.05, 0.05, 0.06, 0.07, 0.08],
    ]
    objectives = [list(problem.fitness(vector)) for vector in decisions]
    summary = summarize_segment_allocation(
        decisions,
        objectives,
        objectives[:2],
        [False, False],
        bins=6,
    )

    assert len(summary["segment_rows"]) == 6
    assert any(row["point_count"] > 0 for row in summary["segment_rows"])
    json.dumps(summary)


def test_segment_spacing_contribution_handles_singleton_segment() -> None:
    summary = summarize_segment_spacing_contribution(
        [[0.1, 0.9]],
        [False, False],
        bins=6,
    )

    assert summary["segment_rows"]
    assert "segment_spacing_singleton_or_empty_front" in summary["warnings"]
    json.dumps(summary)


def test_occupancy_uniformity_handles_empty_segments() -> None:
    summary = summarize_occupancy_uniformity(
        [[0.1, 0.9], [0.15, 0.8]],
        [False, False],
        bins=6,
    )

    assert summary["empty_bins"] >= 0
    assert summary["segment_load_gini"] is not None
    json.dumps(summary)


def test_nondominated_distribution_generates_segment_counts() -> None:
    objective_vectors = [
        [0.1, 0.9],
        [0.15, 0.85],
        [0.4, 0.5],
        [0.6, 0.35],
    ]
    front_vectors = objective_vectors[:3]
    summary = summarize_nondominated_distribution(
        objective_vectors,
        front_vectors,
        [False, False],
        bins=6,
    )

    assert summary["total_nondominated_count"] == 3
    assert len(summary["segment_rows"]) == 6
    assert sum(row["segment_nondominated_count"] for row in summary["segment_rows"]) >= 3


def test_decision_to_segment_mapping_matches_zdt1_components() -> None:
    problem = ZDT1Problem("zdt1")
    decision = [0.08, 0.12, 0.14, 0.18, 0.16, 0.15]
    objective = list(problem.fitness(decision))
    expected = compute_zdt1_components(decision, objective, bins=6)

    summary = summarize_decision_to_segment_mapping(
        [decision],
        [objective],
        [False, False],
        bins=6,
    )
    populated_rows = [row for row in summary["segment_rows"] if row["point_count"] > 0]

    assert len(populated_rows) == 1
    row = populated_rows[0]
    assert abs(float(row["g_mean"]) - float(expected["g"])) < 1e-12
    assert abs(float(row["x0_mean"]) - float(expected["x0"])) < 1e-12


def test_parity_spread_gap_summary_includes_candidate_j_n_and_pymoo_values() -> None:
    rows = summarize_parity_spread_gap(
        {
            "candidate_j_h_lite_retry2": {
                "occupied_bins": 6.6,
                "empty_bins": 29.4,
                "point_count_entropy": 2.2,
                "segment_load_std": 1.3,
                "segment_load_gini": 0.25,
                "spacing": 0.086,
                "total_nondominated_count": 8.7,
                "segment_point_counts": {"0": 3.0, "1": 3.0, "2": 3.0},
                "segment_nondominated_counts": {"0": 2.0, "1": 2.0, "2": 1.0},
                "weakest_segment_id": 0,
            },
            "candidate_n_low_g_tail_mutation_light": {
                "occupied_bins": 6.8,
                "empty_bins": 29.2,
                "point_count_entropy": 2.4,
                "segment_load_std": 1.0,
                "segment_load_gini": 0.20,
                "spacing": 0.081,
                "total_nondominated_count": 9.8,
                "segment_point_counts": {"0": 4.0, "1": 3.0, "2": 3.0},
                "segment_nondominated_counts": {"0": 3.0, "1": 2.0, "2": 2.0},
                "weakest_segment_id": 1,
            },
            "pymoo_nsga2": {
                "occupied_bins": 8.5,
                "empty_bins": 27.5,
                "point_count_entropy": 2.9,
                "segment_load_std": 0.5,
                "segment_load_gini": 0.09,
                "spacing": 0.030,
                "total_nondominated_count": 18.2,
                "segment_point_counts": {"0": 3.0, "1": 3.0, "2": 3.0, "3": 3.0},
                "segment_nondominated_counts": {"0": 3.0, "1": 3.0, "2": 3.0, "3": 3.0},
                "weakest_segment_id": 3,
            },
        }
    )

    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["occupied_bins"]["candidate_j"] == 6.6
    assert by_metric["occupied_bins"]["candidate_n"] == 6.8
    assert by_metric["occupied_bins"]["pymoo"] == 8.5
    assert by_metric["total_nondominated_count"]["gap_segment"] is not None


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

    result = run_pymoo_nsga2(_base_config(), seed=73, budget=80)

    assert result.status == "skipped"
    assert result.success is False
    assert "unavailable" in str(result.error_message)


def test_default_internal_path_stays_free_of_spread_metadata(tmp_path: Path) -> None:
    result = run_internal_nsga2(_base_config(), seed=73, output_root=str(tmp_path / "baseline"))

    assert result.success is True
    assert "candidate_id" not in result.metadata
    assert "spread_parity_trace_enabled" not in result.metadata


def test_spread_parity_runner_writes_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "spread-artifacts"
    output_root = tmp_path / "spread-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_spread_parity.py",
            "--problem",
            "zdt1",
            "--seeds",
            "1",
            "--seed-start",
            "23101",
            "--budget",
            "80",
            "--segment-count",
            "6",
            "--artifact-suffix",
            "spread_parity_test",
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

    assert results_json.name == "nsga2_spread_parity_results_spread_parity_test.json"
    assert report_md.name == "nsga2_spread_parity_report_spread_parity_test.md"
    assert fairness_md.name == "nsga2_spread_parity_fairness_report_spread_parity_test.md"
    assert result_payload["selected_problems"] == ["zdt1"]
    assert result_payload["segment_count"] == 6
    assert result_payload["spread_parity_trace_enabled"] is True
    assert "fairness" in result_payload
    assert "fairness_summary" in result_payload
    assert result_payload["segment_allocation_rows"]
    assert result_payload["segment_spacing_rows"]
    assert result_payload["occupancy_uniformity_aggregate_rows"]
    assert result_payload["nondominated_distribution_rows"]
    assert result_payload["decision_to_segment_rows"]
    assert result_payload["spread_gap_rows"]

    candidate_n_rows = [
        row
        for row in result_payload["raw_rows"]
        if row["algorithm"] == "candidate_n_low_g_tail_mutation_light"
    ]
    assert len(candidate_n_rows) == 1
    candidate_n_row = candidate_n_rows[0]
    assert candidate_n_row["metadata"]["candidate_id"] == "candidate_n_low_g_tail_mutation_light"
    assert candidate_n_row["metadata"]["default_changed"] is False
    assert "segment_allocation_summary" in candidate_n_row
    assert "segment_spacing_contribution" in candidate_n_row
    assert "occupancy_uniformity_summary" in candidate_n_row
    assert "nondominated_distribution_summary" in candidate_n_row
    assert "decision_to_segment_mapping" in candidate_n_row

    internal_rows = [
        row for row in result_payload["raw_rows"] if row["algorithm"] == "internal_nsga2"
    ]
    assert len(internal_rows) == 1
    assert "candidate_id" not in internal_rows[0]["metadata"]
    assert "spread_parity_trace_enabled" not in internal_rows[0]["metadata"]

    assert any(row["metric"] == "occupied_bins" for row in result_payload["spread_gap_rows"])
    issue_types = {issue["issue_type"] for issue in result_payload["fairness"]["issues"]}
    assert "external_operator_family_difference" in issue_types or "evaluation_budget_fail" in issue_types
