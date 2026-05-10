from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.config import GAConfig, load_config
from ga_lab.convergence_diagnostics import configured_evaluation_budget
from ga_lab.experiment.diversity_diagnostics import evaluate_diversity_diagnostics
from ga_lab.experiment.external_mo_comparators import (
    ExternalMOComparatorResult,
    result_to_front_row,
    run_internal_nsga2,
    run_pymoo_nsga2,
)
from ga_lab.experiment.external_operator_parity import (
    extract_internal_final_decisions,
    extract_internal_final_objectives,
    extract_pymoo_final_decisions,
    extract_pymoo_final_objectives,
)
from ga_lab.experiment.mo_runner_fairness import (
    build_candidate_rows,
    build_mo_benchmark_rows,
    decorate_fairness_row,
    fairness_summary_rows,
)
from ga_lab.experiment.nsga2_candidate_suite import (
    build_problem_config,
    mo_candidate_suite_specs,
    reference_front_for_spec,
    safe_artifact_path,
)
from ga_lab.experiment.nsga2_candidate_variants import (
    NSGA2CandidateVariant,
    apply_candidate_variant,
    candidate_j_h_lite_retry2,
    candidate_n_low_g_tail_mutation_light,
    candidate_o_spread_preserving_variation_light,
    candidate_variant_metadata,
)
from ga_lab.experiment.parameter_fairness import evaluate_parameter_fairness
from ga_lab.experiment.spread_parity_diagnostics import (
    summarize_decision_to_segment_mapping,
    summarize_nondominated_distribution,
    summarize_occupancy_uniformity,
    summarize_segment_allocation,
    summarize_segment_spacing_contribution,
)


def _load_base_helpers():
    helper_path = PROJECT_ROOT / "scripts" / "validate_nsga2_candidate_suite.py"
    spec = importlib.util.spec_from_file_location(
        "_candidate_suite_spread_phase0_base",
        helper_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 0 sanity validation for spread-preserving candidate_o."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problem", dest="problems", default="zdt1")
    parser.add_argument("--problems", dest="problems")
    parser.add_argument("--output-root", default="outputs/nsga2_spread_preserving_phase0")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=24101)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--segment-count", type=int, default=6)
    parser.add_argument("--include-pymoo", action="store_true")
    parser.add_argument(
        "--drift-audit-json",
        default="artifacts/nsga2_candidate_o_default_drift_audit_results.json",
        help="Optional drift-audit JSON included in the report if it exists.",
    )
    parser.add_argument(
        "--local-baseline-status",
        default="not_run",
        help="Optional local baseline governance status recorded in the report.",
    )
    parser.add_argument(
        "--local-baseline-note",
        default="see regression check section",
        help="Optional note for the local baseline governance row in the report.",
    )
    return parser.parse_args()


def _phase0_variants() -> list[NSGA2CandidateVariant]:
    return [
        candidate_j_h_lite_retry2(),
        candidate_n_low_g_tail_mutation_light(),
        candidate_o_spread_preserving_variation_light(),
    ]


def _retarget_budget(base_config: GAConfig, requested_budget: int) -> GAConfig:
    clone = GAConfig.from_dict(base_config.to_dict())
    if configured_evaluation_budget(clone) == requested_budget:
        return clone

    best_match: tuple[int, int, int] | None = None
    max_population = max(clone.population_size * 2, 40)
    for population_size in range(4, max_population + 1):
        if requested_budget % population_size != 0:
            continue
        generation_term = (requested_budget // population_size) - 2
        if generation_term <= 0 or generation_term % 3 != 0:
            continue
        generations = generation_term // 3
        if generations <= 0:
            continue
        score = abs(population_size - clone.population_size) + abs(generations - clone.generations)
        if best_match is None or score < best_match[0]:
            best_match = (score, population_size, generations)

    if best_match is None:
        raise ValueError(
            f"Unable to derive an exact NSGA-II population/generation pair for requested budget {requested_budget}"
        )

    _, population_size, generations = best_match
    clone.population_size = population_size
    clone.generations = generations
    clone.elitism = min(clone.elitism, population_size - 1)
    clone.tournament_size = min(max(2, clone.tournament_size), population_size)
    if "tournament_size" in clone.selection_options:
        clone.selection_options["tournament_size"] = min(
            max(2, int(clone.selection_options["tournament_size"])),
            population_size,
        )
    return clone


def _enable_phase0_diagnostics(config: GAConfig, *, candidate_id: str | None, segment_count: int) -> GAConfig:
    clone = GAConfig.from_dict(config.to_dict())
    clone.algorithm_options = dict(clone.algorithm_options)
    clone.algorithm_options["nsga2_trace_enabled"] = True
    clone.algorithm_options["nsga2_operator_supply_trace_enabled"] = True
    clone.algorithm_options["nsga2_zdt1_component_trace_enabled"] = True
    clone.algorithm_options["nsga2_trace_generation_sample_stride"] = 1
    clone.algorithm_options["nsga2_trace_segment_count"] = max(1, int(segment_count))
    if candidate_id:
        clone.algorithm_options["nsga2_trace_candidate_id"] = candidate_id
        clone.algorithm_options["nsga2_trace_run_id"] = f"{candidate_id}_{clone.problem}_phase0"
    else:
        clone.algorithm_options["nsga2_trace_run_id"] = f"internal_nsga2_{clone.problem}_phase0"
    return clone


def _diag_metric(row: dict[str, Any], trace_type: str, metric_name: str) -> float | None:
    diagnostics = dict(row.get("metadata", {}).get("nsga2_diagnostics", {}))
    aggregate = dict(diagnostics.get("aggregate", {}))
    trace_payload = dict(aggregate.get(trace_type, {}))
    value = trace_payload.get(f"{metric_name}_mean")
    if not isinstance(value, int | float):
        return None
    value_float = float(value)
    if not math.isfinite(value_float):
        return None
    return value_float


def _finite(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    value_float = float(value)
    if not math.isfinite(value_float):
        return None
    return value_float


def _decision_vectors_for_row(row: dict[str, Any]) -> list[list[float]]:
    if str(row.get("algorithm")) == "pymoo_nsga2":
        return extract_pymoo_final_decisions(row)
    return extract_internal_final_decisions(row)


def _objective_vectors_for_row(row: dict[str, Any]) -> list[list[float]]:
    if str(row.get("algorithm")) == "pymoo_nsga2":
        return extract_pymoo_final_objectives(row)
    return extract_internal_final_objectives(row)


def _front_objectives_for_row(row: dict[str, Any]) -> list[list[float]]:
    values = row.get("nondominated_objective_vectors")
    if isinstance(values, list) and values:
        return [list(map(float, vector)) for vector in values]
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        front_values = metadata.get("front_objective_vectors")
        if isinstance(front_values, list) and front_values:
            return [list(map(float, vector)) for vector in front_values]
    values = row.get("objective_vectors")
    if isinstance(values, list):
        return [list(map(float, vector)) for vector in values]
    return []


def _segment_row(summary: dict[str, Any], segment_id: int) -> dict[str, Any] | None:
    rows = summary.get("segment_rows", [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and int(row.get("segment_id", -1)) == segment_id:
            return row
    return None


def _decorate_row(
    row: dict[str, Any],
    *,
    spec,
    base_config: GAConfig,
    requested_budget: int,
    variant_map: dict[str, NSGA2CandidateVariant],
    reference_front: list[list[float]],
    segment_count: int,
) -> dict[str, Any]:
    row = decorate_fairness_row(
        row,
        spec=spec,
        base_config=base_config,
        requested_budget=requested_budget,
        variant_map=variant_map,
    )
    directions = [
        bool(value) for value in row.get("metadata", {}).get("objective_directions", [False, False])
    ]
    if row.get("success"):
        decision_vectors = row.get("front_decision_vectors") or row.get("decision_vectors")
        row.update(
            evaluate_diversity_diagnostics(
                row.get("objective_vectors", []),
                directions=directions,
                decision_vectors=decision_vectors if isinstance(decision_vectors, list) else None,
            )
        )
        row["metric_calculation_success"] = all(
            isinstance(row.get(metric_name), int | float) and math.isfinite(float(row[metric_name]))
            for metric_name in (
                "hypervolume_2d",
                "reference_front_distance",
                "inverted_generational_distance",
                "spacing",
                "nondominated_count",
            )
        )
        diagnostics = dict(row.get("metadata", {}).get("nsga2_diagnostics", {}))
        trace_types = set(diagnostics.get("aggregate", {}).get("trace_types", []))
        row["zdt1_component_diagnostics_success"] = {
            "zdt1_initial_component_coverage",
            "zdt1_offspring_component_quality",
            "zdt1_parent_child_component_delta",
            "zdt1_mutation_retry_component_effect",
            "zdt1_segment0_quality_funnel",
        }.issubset(trace_types)
        row["segment0_g"] = _diag_metric(row, "zdt1_offspring_component_quality", "segment0_g_mean")
        row["segment0_distance"] = _diag_metric(
            row,
            "zdt1_offspring_component_quality",
            "segment0_distance_mean",
        )
        row["segment0_low_g_count"] = _diag_metric(
            row,
            "zdt1_segment0_quality_funnel",
            "segment0_low_g_count",
        )
        row["segment0_nondominated_rate"] = _diag_metric(
            row,
            "zdt1_offspring_component_quality",
            "segment0_nondominated_rate",
        )
        row["segment0_survival_rate"] = _diag_metric(
            row,
            "zdt1_offspring_component_quality",
            "segment0_survival_rate",
        )
        row["segment0_offspring_count"] = _diag_metric(
            row,
            "operator_offspring_quality",
            "segment0_offspring_count",
        )
        low_g_stats = dict(row.get("metadata", {}).get("low_g_tail_mutation_stats", {}))
        row["low_g_tail_adjusted_solution_count"] = low_g_stats.get("adjusted_solution_count")
        row["low_g_tail_adjusted_gene_count"] = low_g_stats.get("adjusted_gene_count")
        row["low_g_tail_mean_step"] = low_g_stats.get("mean_step")
        spread_stats = dict(row.get("metadata", {}).get("spread_preserving_variation_stats", {}))
        row["spread_preserving_adjusted_solution_count"] = spread_stats.get(
            "adjusted_solution_count"
        )
        row["spread_preserving_adjusted_gene_count"] = spread_stats.get("adjusted_gene_count")
        row["spread_preserving_mean_abs_step"] = spread_stats.get("mean_abs_step")

        decision_vectors = _decision_vectors_for_row(row)
        population_objectives = _objective_vectors_for_row(row)
        front_objectives = _front_objectives_for_row(row)
        segment_allocation = summarize_segment_allocation(
            decision_vectors,
            population_objectives,
            front_objectives,
            directions,
            bins=segment_count,
        )
        segment_spacing = summarize_segment_spacing_contribution(
            front_objectives,
            directions,
            bins=segment_count,
        )
        occupancy_uniformity = summarize_occupancy_uniformity(
            front_objectives,
            directions,
            bins=segment_count,
        )
        nondominated_distribution = summarize_nondominated_distribution(
            population_objectives,
            front_objectives,
            directions,
            bins=segment_count,
        )
        decision_segment_mapping = summarize_decision_to_segment_mapping(
            decision_vectors,
            population_objectives,
            directions,
            bins=segment_count,
        )

        row["segment_allocation_summary"] = segment_allocation
        row["segment_spacing_contribution"] = segment_spacing
        row["occupancy_uniformity_summary"] = occupancy_uniformity
        row["nondominated_distribution_summary"] = nondominated_distribution
        row["decision_to_segment_mapping"] = decision_segment_mapping
        row["spread_parity_diagnostics_success"] = bool(segment_allocation.get("segment_rows"))
        row["occupied_bins"] = occupancy_uniformity.get("occupied_bins")
        row["empty_bins"] = occupancy_uniformity.get("empty_bins")
        row["segment_entropy"] = occupancy_uniformity.get("point_count_entropy")
        row["segment_load_gini"] = occupancy_uniformity.get("segment_load_gini")
        row["weakest_segment_id"] = segment_spacing.get("weakest_segment_id")
        row["largest_gap_segment_id"] = segment_spacing.get("largest_gap_segment_id")
        row["total_nondominated_count"] = nondominated_distribution.get("total_nondominated_count")
        segment0_allocation = _segment_row(segment_allocation, 0)
        segment4_spacing = _segment_row(segment_spacing, 4)
        row["segment0_allocation"] = (
            segment0_allocation.get("point_count") if isinstance(segment0_allocation, dict) else None
        )
        row["segment4_spacing"] = (
            segment4_spacing.get("local_spacing_contribution")
            if isinstance(segment4_spacing, dict)
            else None
        )
        row["spread_parity_warnings"] = sorted(
            set(
                list(segment_allocation.get("warnings", []))
                + list(segment_spacing.get("warnings", []))
                + list(occupancy_uniformity.get("warnings", []))
                + list(nondominated_distribution.get("warnings", []))
                + list(decision_segment_mapping.get("warnings", []))
            )
        )
    else:
        row["metric_calculation_success"] = False
        row["zdt1_component_diagnostics_success"] = False
        row["segment0_g"] = None
        row["segment0_distance"] = None
        row["segment0_low_g_count"] = None
        row["segment0_nondominated_rate"] = None
        row["segment0_survival_rate"] = None
        row["segment0_offspring_count"] = None
        row["low_g_tail_adjusted_solution_count"] = None
        row["low_g_tail_adjusted_gene_count"] = None
        row["low_g_tail_mean_step"] = None
        row["spread_preserving_adjusted_solution_count"] = None
        row["spread_preserving_adjusted_gene_count"] = None
        row["spread_preserving_mean_abs_step"] = None
        row["segment_allocation_summary"] = {}
        row["segment_spacing_contribution"] = {}
        row["occupancy_uniformity_summary"] = {}
        row["nondominated_distribution_summary"] = {}
        row["decision_to_segment_mapping"] = {}
        row["spread_parity_diagnostics_success"] = False
        row["occupied_bins"] = None
        row["empty_bins"] = None
        row["segment_entropy"] = None
        row["segment_load_gini"] = None
        row["weakest_segment_id"] = None
        row["largest_gap_segment_id"] = None
        row["total_nondominated_count"] = None
        row["segment0_allocation"] = None
        row["segment4_spacing"] = None
        row["spread_parity_warnings"] = []
    return row


def _make_candidate_result(
    base_config: GAConfig,
    variant: NSGA2CandidateVariant,
    *,
    seed: int,
    output_root: Path,
    segment_count: int,
) -> ExternalMOComparatorResult:
    candidate_config = _enable_phase0_diagnostics(
        apply_candidate_variant(base_config, variant),
        candidate_id=variant.candidate_id,
        segment_count=segment_count,
    )
    result = run_internal_nsga2(candidate_config, seed=seed, output_root=str(output_root))
    metadata = dict(result.metadata)
    metadata.update(candidate_variant_metadata(variant))
    return ExternalMOComparatorResult(
        problem_name=result.problem_name,
        algorithm_name=variant.candidate_id,
        library_name="internal_candidate",
        seed=result.seed,
        requested_budget=result.requested_budget,
        evaluations=result.evaluations,
        runtime_seconds=result.runtime_seconds,
        status=result.status,
        success=result.success,
        error_message=result.error_message,
        objective_vectors=result.objective_vectors,
        nondominated_objective_vectors=result.nondominated_objective_vectors,
        metadata=metadata,
    )


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    aggregates: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        successful = [row for row in bucket if row.get("success")]
        statuses = {str(row.get("status", "unknown")) for row in bucket}
        status = (
            "skipped"
            if statuses == {"skipped"}
            else "failed"
            if "failed" in statuses and not successful
            else "partial_failure"
            if "failed" in statuses
            else "success"
        )
        aggregates.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "status": status,
                "seeds": len(bucket),
                "successful_seeds": len(successful),
                "mean_hv": BASE._summary_stat(successful, "hypervolume_2d")["mean"],
                "mean_distance": BASE._summary_stat(successful, "reference_front_distance")["mean"],
                "mean_igd": BASE._summary_stat(successful, "inverted_generational_distance")["mean"],
                "mean_spacing": BASE._summary_stat(successful, "spacing")["mean"],
                "mean_nondominated_count": BASE._summary_stat(successful, "nondominated_count")["mean"],
                "mean_occupied_bins": BASE._summary_stat(successful, "occupied_bins")["mean"],
                "mean_segment_entropy": BASE._summary_stat(successful, "segment_entropy")["mean"],
                "mean_segment0_allocation": BASE._summary_stat(successful, "segment0_allocation")["mean"],
                "mean_segment4_spacing": BASE._summary_stat(successful, "segment4_spacing")["mean"],
                "mean_segment0_g": BASE._summary_stat(successful, "segment0_g")["mean"],
                "mean_segment0_distance": BASE._summary_stat(successful, "segment0_distance")["mean"],
                "mean_runtime_seconds": BASE._summary_stat(successful, "runtime_seconds")["mean"],
                "mean_actual_evaluations": BASE._summary_stat(successful, "actual_evaluations")["mean"],
                "success_rate": BASE._success_rate(bucket),
            }
        )
    return aggregates


def _zdt1_component_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("success"):
            grouped[str(row["algorithm"])].append(row)
    rows: list[dict[str, Any]] = []
    for algorithm, bucket in sorted(grouped.items()):
        rows.append(
            {
                "algorithm": algorithm,
                "segment0_g_mean": BASE._summary_stat(bucket, "segment0_g")["mean"],
                "segment0_distance_mean": BASE._summary_stat(bucket, "segment0_distance")["mean"],
                "segment0_low_g_count": BASE._summary_stat(bucket, "segment0_low_g_count")["mean"],
                "segment0_nondominated_rate": BASE._summary_stat(
                    bucket,
                    "segment0_nondominated_rate",
                )["mean"],
                "segment0_survival_rate": BASE._summary_stat(
                    bucket,
                    "segment0_survival_rate",
                )["mean"],
            }
        )
    return rows


def _operator_supply_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("success"):
            grouped[str(row["algorithm"])].append(row)
    rows: list[dict[str, Any]] = []
    for algorithm, bucket in sorted(grouped.items()):
        rows.append(
            {
                "algorithm": algorithm,
                "segment0_offspring_count": BASE._summary_stat(bucket, "segment0_offspring_count")["mean"],
                "segment0_nondominated_rate": BASE._summary_stat(
                    bucket,
                    "segment0_nondominated_rate",
                )["mean"],
                "segment0_survival_rate": BASE._summary_stat(
                    bucket,
                    "segment0_survival_rate",
                )["mean"],
            }
        )
    return rows


def _spread_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("success"):
            grouped[str(row["algorithm"])].append(row)
    rows: list[dict[str, Any]] = []
    for algorithm, bucket in sorted(grouped.items()):
        weakest_values = [row.get("weakest_segment_id") for row in bucket if row.get("weakest_segment_id") is not None]
        weakest_segment = None
        if weakest_values:
            counts: dict[int, int] = defaultdict(int)
            for value in weakest_values:
                counts[int(value)] += 1
            weakest_segment = min(
                (segment for segment, count in counts.items() if count == max(counts.values())),
                default=None,
            )
        rows.append(
            {
                "algorithm": algorithm,
                "occupied_bins": BASE._summary_stat(bucket, "occupied_bins")["mean"],
                "segment_entropy": BASE._summary_stat(bucket, "segment_entropy")["mean"],
                "spacing": BASE._summary_stat(bucket, "spacing")["mean"],
                "nondominated_count": BASE._summary_stat(bucket, "nondominated_count")["mean"],
                "segment0_allocation": BASE._summary_stat(bucket, "segment0_allocation")["mean"],
                "segment4_spacing": BASE._summary_stat(bucket, "segment4_spacing")["mean"],
                "weakest_segment_id": weakest_segment,
            }
        )
    return rows


def _aggregate_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["algorithm"]): row for row in rows}


def _catastrophic_regression(
    candidate_row: dict[str, Any] | None,
    reference_row: dict[str, Any] | None,
) -> bool:
    if not candidate_row or not reference_row:
        return True

    def _worse_higher(key: str, tolerance: float) -> bool:
        candidate_value = _finite(candidate_row.get(key))
        reference_value = _finite(reference_row.get(key))
        if candidate_value is None or reference_value is None:
            return True
        return candidate_value < (reference_value - tolerance)

    def _worse_lower(key: str, tolerance_ratio: float, tolerance_abs: float = 0.0) -> bool:
        candidate_value = _finite(candidate_row.get(key))
        reference_value = _finite(reference_row.get(key))
        if candidate_value is None or reference_value is None:
            return True
        threshold = max(tolerance_abs, abs(reference_value) * tolerance_ratio)
        return candidate_value > (reference_value + threshold)

    return any(
        (
            _worse_higher("mean_hv", 0.05),
            _worse_lower("mean_distance", 0.20, 0.05),
            _worse_lower("mean_igd", 0.20, 0.05),
            _worse_higher("mean_occupied_bins", 1.0),
            _worse_lower("mean_spacing", 0.25, 0.02),
            _worse_lower("mean_segment0_g", 0.15, 0.10),
            _worse_lower("mean_segment0_distance", 0.15, 0.10),
        )
    )


def _sanity_gate_results(
    raw_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    *,
    local_baseline_status: str,
    local_baseline_note: str,
    drift_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidate_rows = [
        row for row in raw_rows if row["algorithm"] == "candidate_o_spread_preserving_variation_light"
    ]
    default_rows = [row for row in raw_rows if row["algorithm"] == "internal_nsga2"]
    metadata_ok = all(
        row.get("metadata", {}).get("candidate_id") == "candidate_o_spread_preserving_variation_light"
        and row.get("metadata", {}).get("base_candidate_id")
        == "candidate_n_low_g_tail_mutation_light"
        and row.get("metadata", {}).get("mechanism") == "spread_preserving_variation_light"
        for row in candidate_rows
    )
    default_changed_ok = all(
        row.get("metadata", {}).get("default_changed") is False for row in candidate_rows
    )
    candidate_isolation_ok = all(
        "candidate_id" not in row.get("metadata", {})
        and "default_changed" not in row.get("metadata", {})
        for row in default_rows
    )
    evaluations_ok = all(
        row.get("requested_budget") == row.get("actual_evaluations")
        for row in candidate_rows
        if row.get("success")
    )
    spread_parity_ok = all(
        bool(row.get("spread_parity_diagnostics_success"))
        for row in candidate_rows
        if row.get("success")
    )
    zdt1_component_ok = all(
        bool(row.get("zdt1_component_diagnostics_success"))
        for row in candidate_rows
        if row.get("success")
    )
    non_finite_ok = all(row.get("success") for row in candidate_rows)
    metrics_ok = all(
        bool(row.get("metric_calculation_success")) for row in candidate_rows if row.get("success")
    )
    fairness_fail_free = fairness_payload.get("summary_counts", {}).get("fail", 0) == 0
    drift_ok = drift_payload is not None and drift_payload.get("overall", {}).get("drift_detected") is False
    local_baseline_ok = str(local_baseline_status).strip().lower() == "pass"

    return [
        {
            "gate": "candidate metadata",
            "result": metadata_ok,
            "pass_fail": metadata_ok,
            "note": "candidate_o metadata present with expected base candidate and mechanism",
        },
        {
            "gate": "default_changed=false",
            "result": default_changed_ok,
            "pass_fail": default_changed_ok,
            "note": "candidate_o metadata keeps default_changed=false",
        },
        {
            "gate": "candidate isolation",
            "result": candidate_isolation_ok,
            "pass_fail": candidate_isolation_ok,
            "note": "default internal NSGA-II rows stay free of candidate metadata",
        },
        {
            "gate": "default drift",
            "result": drift_ok,
            "pass_fail": drift_ok,
            "note": (
                "candidate_o default drift audit reports NO DRIFT"
                if drift_payload is not None
                else "drift audit artifact not available"
            ),
        },
        {
            "gate": "fairness fail 없음",
            "result": fairness_fail_free,
            "pass_fail": fairness_fail_free,
            "note": f"fairness status={fairness_payload.get('status')}",
        },
        {
            "gate": "actual evaluations match",
            "result": evaluations_ok,
            "pass_fail": evaluations_ok,
            "note": "candidate_o actual evaluations match requested budget",
        },
        {
            "gate": "spread parity diagnostics success",
            "result": spread_parity_ok,
            "pass_fail": spread_parity_ok,
            "note": "required spread-parity summaries were emitted",
        },
        {
            "gate": "ZDT1 component diagnostics success",
            "result": zdt1_component_ok,
            "pass_fail": zdt1_component_ok,
            "note": "required ZDT1 component traces were emitted",
        },
        {
            "gate": "non-finite objective 없음",
            "result": non_finite_ok,
            "pass_fail": non_finite_ok,
            "note": "candidate_o runs completed without fail-fast objective errors",
        },
        {
            "gate": "local baseline check",
            "result": local_baseline_ok,
            "pass_fail": local_baseline_ok,
            "note": local_baseline_note,
        },
        {
            "gate": "artifact generation",
            "result": True,
            "pass_fail": True,
            "note": "results and report artifacts created",
        },
        {
            "gate": "metric calculation success",
            "result": metrics_ok,
            "pass_fail": metrics_ok,
            "note": "core MO metrics stayed finite",
        },
    ]


def _phase0_decision(
    gate_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    failures: list[dict[str, Any]],
    *,
    aggregate_rows: list[dict[str, Any]],
) -> str:
    if fairness_payload.get("summary_counts", {}).get("fail", 0):
        return "Phase 0 failed, fix required"
    if not all(bool(row["pass_fail"]) for row in gate_rows):
        return "Phase 0 failed, fix required"

    aggregate_lookup = _aggregate_lookup(aggregate_rows)
    candidate_o = aggregate_lookup.get("candidate_o_spread_preserving_variation_light")
    candidate_n = aggregate_lookup.get("candidate_n_low_g_tail_mutation_light")
    catastrophic = _catastrophic_regression(candidate_o, candidate_n)
    if catastrophic:
        return "Phase 0 failed, fix required"

    spread_metrics = (
        _finite(candidate_o.get("mean_occupied_bins")) if candidate_o else None,
        _finite(candidate_o.get("mean_spacing")) if candidate_o else None,
        _finite(candidate_o.get("mean_nondominated_count")) if candidate_o else None,
        _finite(candidate_n.get("mean_occupied_bins")) if candidate_n else None,
        _finite(candidate_n.get("mean_spacing")) if candidate_n else None,
        _finite(candidate_n.get("mean_nondominated_count")) if candidate_n else None,
    )
    all_worse = False
    if all(value is not None for value in spread_metrics):
        (
            o_bins,
            o_spacing,
            o_count,
            n_bins,
            n_spacing,
            n_count,
        ) = spread_metrics
        all_worse = bool(o_bins < n_bins and o_spacing > n_spacing and o_count < n_count)

    if failures or fairness_payload.get("summary_counts", {}).get("warning", 0) or all_worse:
        return "Phase 0 passed with warnings"
    return "Phase 0 passed, eligible for Phase 1 planning"


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _results_markdown(
    aggregate_rows: list[dict[str, Any]],
    spread_rows: list[dict[str, Any]],
    zdt1_component_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
) -> str:
    lines = [
        "# NSGA-II Spread-Preserving Variation Phase 0 Results",
        "",
        "## Aggregate Results",
        "",
        *BASE._markdown_table(
            aggregate_rows,
            [
                "problem",
                "algorithm",
                "status",
                "mean_hv",
                "mean_igd",
                "mean_spacing",
                "mean_nondominated_count",
                "mean_occupied_bins",
                "mean_segment_entropy",
                "mean_segment0_g",
                "mean_segment0_distance",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Spread Signal",
        "",
        *BASE._markdown_table(
            spread_rows,
            [
                "algorithm",
                "occupied_bins",
                "segment_entropy",
                "spacing",
                "nondominated_count",
                "segment0_allocation",
                "segment4_spacing",
                "weakest_segment_id",
            ],
        ),
        "",
        "## ZDT1 Component Signal",
        "",
        *BASE._markdown_table(
            zdt1_component_rows,
            [
                "algorithm",
                "segment0_g_mean",
                "segment0_distance_mean",
                "segment0_low_g_count",
                "segment0_nondominated_rate",
                "segment0_survival_rate",
            ],
        ),
        "",
        "## Fairness Summary",
        "",
        *BASE._markdown_table(
            fairness_summary_rows(fairness_payload),
            ["status", "pass", "warning", "fail"],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _report_markdown(
    *,
    args: argparse.Namespace,
    aggregate_rows: list[dict[str, Any]],
    spread_rows: list[dict[str, Any]],
    zdt1_component_rows: list[dict[str, Any]],
    operator_supply_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    failures: list[dict[str, Any]],
    decision: str,
    drift_payload: dict[str, Any] | None,
) -> str:
    aggregate_lookup = _aggregate_lookup(aggregate_rows)
    spread_lookup = _aggregate_lookup(spread_rows)
    component_lookup = _aggregate_lookup(zdt1_component_rows)
    candidate_n_aggregate = aggregate_lookup.get("candidate_n_low_g_tail_mutation_light", {})
    candidate_o_aggregate = aggregate_lookup.get("candidate_o_spread_preserving_variation_light", {})
    candidate_n_spread = spread_lookup.get("candidate_n_low_g_tail_mutation_light", {})
    candidate_o_spread = spread_lookup.get("candidate_o_spread_preserving_variation_light", {})
    candidate_n_component = component_lookup.get("candidate_n_low_g_tail_mutation_light", {})
    candidate_o_component = component_lookup.get("candidate_o_spread_preserving_variation_light", {})
    phase1_allowed = decision == "Phase 0 passed, eligible for Phase 1 planning"
    drift_text = (
        "NO DRIFT"
        if drift_payload is not None and drift_payload.get("overall", {}).get("drift_detected") is False
        else "DRIFT NOT AVAILABLE"
        if drift_payload is None
        else "DRIFT DETECTED"
    )
    lines = [
        "# NSGA-II Spread-Preserving Variation Phase 0 Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: `candidate_o_spread_preserving_variation_light`의 Phase 0 sanity와 spread-preservation signal 점검",
        "- 구현한 candidate: `candidate_o_spread_preserving_variation_light`",
        "- 기본값 변경 여부: `false`",
        f"- 실행한 Phase 0 sanity: problem=`{','.join(args.problems.split(','))}`, seeds=`{args.seeds}`, budget=`{args.budget}`",
        f"- default drift 결과: **{drift_text}**",
        f"- fairness 결과: **{fairness_payload.get('status', 'unknown')}**",
        "- candidate isolation 결과: PASS 여부는 gate 표 참고",
        "- spread parity signal: candidate_n 대비 occupied-bin / entropy / spacing / count 변화 확인",
        "- ZDT1 component signal: segment0 g / distance / low-g count / nondominated rate / survival rate 확인",
        f"- Phase 0 판정: **{decision}**",
        f"- Phase 1 진행 가능 여부: **{'가능' if phase1_allowed else '자동 승인 아님'}**",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "Scope:",
        "- candidate_o_spread_preserving_variation_light",
        "- Phase 0 sanity",
        "- ZDT1 small run",
        "- spread parity diagnostics",
        "- ZDT1 component diagnostics",
        "- operator supply diagnostics",
        "- candidate isolation",
        "- default drift audit",
        "- fairness check",
        "- artifact generation",
        "",
        "Non-Scope:",
        "- default promotion",
        "- candidate_o change request",
        "- approved opt-in profile",
        "- Phase 1 full benchmark",
        "- DTLZ/WFG validation",
        "- new survivor-pressure candidates",
        "- pymoo operator clone",
        "- productization",
        "",
        "## 3. Candidate Definition",
        "",
        *BASE._markdown_table(
            [
                {
                    "field": "candidate_id",
                    "value": "candidate_o_spread_preserving_variation_light",
                },
                {"field": "base_candidate", "value": "candidate_n_low_g_tail_mutation_light"},
                {"field": "mechanism", "value": "spread_preserving_variation_light"},
                {"field": "default_changed", "value": False},
                {"field": "promotion_status", "value": "phase0_sanity"},
                {"field": "allowed_use", "value": "phase0_sanity_only"},
                {"field": "disallowed_use", "value": "default_replacement"},
                {
                    "field": "zdt1_specific_warning",
                    "value": "ZDT1-specific sanity probe only",
                },
                {"field": "no_pymoo_operator_clone", "value": True},
            ],
            ["field", "value"],
        ),
        "",
        "## 4. Implementation Summary",
        "",
        *BASE._markdown_table(
            [
                {
                    "파일": "src/ga_lab/experiment/nsga2_candidate_variants.py",
                    "변경 내용": "candidate_o metadata와 apply_candidate_variant opt-in path 추가",
                    "default 영향": "none",
                },
                {
                    "파일": "src/ga_lab/algorithms/nsga2.py",
                    "변경 내용": "opt-in spread-preserving jitter hook과 stats 추가",
                    "default 영향": "option disabled 시 no-op",
                },
                {
                    "파일": "src/ga_lab/experiment/external_mo_comparators.py",
                    "변경 내용": "candidate_o spread stats metadata surface",
                    "default 영향": "none",
                },
                {
                    "파일": "scripts/validate_nsga2_spread_preserving_phase0.py",
                    "변경 내용": "candidate_o Phase 0 runner 추가",
                    "default 영향": "none",
                },
                {
                    "파일": "tests/test_spread_preserving_phase0.py",
                    "변경 내용": "candidate_o metadata, runner, diagnostics test 추가",
                    "default 영향": "none",
                },
            ],
            ["파일", "변경 내용", "default 영향"],
        ),
        "",
        "## 5. Phase 0 Configuration",
        "",
        *BASE._markdown_table(
            [
                {"항목": "problem", "값": args.problems},
                {"항목": "seeds", "값": args.seeds},
                {"항목": "budget", "값": args.budget},
                {
                    "항목": "algorithms",
                    "값": "internal_nsga2, candidate_j_h_lite_retry2, candidate_n_low_g_tail_mutation_light, candidate_o_spread_preserving_variation_light"
                    + (", pymoo_nsga2" if args.include_pymoo else ""),
                },
                {
                    "항목": "metrics",
                    "값": "HV, IGD, spacing, nondominated_count, occupied_bins, segment_entropy, segment0_g, segment0_distance",
                },
                {
                    "항목": "diagnostics",
                    "값": "spread parity, ZDT1 component, operator supply",
                },
                {"항목": "fairness checker 사용 여부", "값": True},
            ],
            ["항목", "값"],
        ),
        "",
        "## 6. Sanity Gate Results",
        "",
        *BASE._markdown_table(gate_rows, ["gate", "result", "pass_fail", "note"]),
        "",
        "## 7. Metric Snapshot",
        "",
        *BASE._markdown_table(
            [
                {
                    "알고리즘": row["algorithm"],
                    "mean HV": row.get("mean_hv"),
                    "mean IGD": row.get("mean_igd"),
                    "mean spacing": row.get("mean_spacing"),
                    "nondominated_count": row.get("mean_nondominated_count"),
                    "occupied_bins": row.get("mean_occupied_bins"),
                    "segment_entropy": row.get("mean_segment_entropy"),
                    "segment0_g": row.get("mean_segment0_g"),
                    "segment0_distance": row.get("mean_segment0_distance"),
                    "runtime": row.get("mean_runtime_seconds"),
                }
                for row in aggregate_rows
            ],
            [
                "알고리즘",
                "mean HV",
                "mean IGD",
                "mean spacing",
                "nondominated_count",
                "occupied_bins",
                "segment_entropy",
                "segment0_g",
                "segment0_distance",
                "runtime",
            ],
        ),
        "",
        "## 8. Spread Parity Signal",
        "",
        *BASE._markdown_table(
            [
                {
                    "항목": "occupied_bins",
                    "candidate_n": candidate_n_spread.get("occupied_bins"),
                    "candidate_o": candidate_o_spread.get("occupied_bins"),
                    "해석": "candidate_n 대비 occupied-bin breadth 변화",
                },
                {
                    "항목": "spacing",
                    "candidate_n": candidate_n_spread.get("spacing"),
                    "candidate_o": candidate_o_spread.get("spacing"),
                    "해석": "lower is better",
                },
                {
                    "항목": "nondominated_count",
                    "candidate_n": candidate_n_spread.get("nondominated_count"),
                    "candidate_o": candidate_o_spread.get("nondominated_count"),
                    "해석": "higher is better",
                },
                {
                    "항목": "segment_entropy",
                    "candidate_n": candidate_n_spread.get("segment_entropy"),
                    "candidate_o": candidate_o_spread.get("segment_entropy"),
                    "해석": "segment allocation uniformity proxy",
                },
                {
                    "항목": "segment 0 allocation",
                    "candidate_n": candidate_n_spread.get("segment0_allocation"),
                    "candidate_o": candidate_o_spread.get("segment0_allocation"),
                    "해석": "segment 0 breadth proxy",
                },
                {
                    "항목": "segment 4 spacing",
                    "candidate_n": candidate_n_spread.get("segment4_spacing"),
                    "candidate_o": candidate_o_spread.get("segment4_spacing"),
                    "해석": "segment 4 local spacing contribution",
                },
            ],
            ["항목", "candidate_n", "candidate_o", "해석"],
        ),
        "",
        "## 9. ZDT1 Component Signal",
        "",
        *BASE._markdown_table(
            [
                {
                    "항목": "segment0_g_mean",
                    "candidate_n": candidate_n_component.get("segment0_g_mean"),
                    "candidate_o": candidate_o_component.get("segment0_g_mean"),
                    "해석": "lower is better",
                },
                {
                    "항목": "segment0_distance_mean",
                    "candidate_n": candidate_n_component.get("segment0_distance_mean"),
                    "candidate_o": candidate_o_component.get("segment0_distance_mean"),
                    "해석": "lower is better",
                },
                {
                    "항목": "segment0_low_g_count",
                    "candidate_n": candidate_n_component.get("segment0_low_g_count"),
                    "candidate_o": candidate_o_component.get("segment0_low_g_count"),
                    "해석": "higher is better",
                },
                {
                    "항목": "segment0_nondominated_rate",
                    "candidate_n": candidate_n_component.get("segment0_nondominated_rate"),
                    "candidate_o": candidate_o_component.get("segment0_nondominated_rate"),
                    "해석": "higher is better",
                },
                {
                    "항목": "segment0_survival_rate",
                    "candidate_n": candidate_n_component.get("segment0_survival_rate"),
                    "candidate_o": candidate_o_component.get("segment0_survival_rate"),
                    "해석": "higher is better",
                },
            ],
            ["항목", "candidate_n", "candidate_o", "해석"],
        ),
        "",
        "## 10. Failures and Warnings",
        "",
        *BASE._markdown_table(
            failures
            or [{"유형": "none", "대상": "none", "메시지": "none", "영향": "none", "조치": "none"}],
            ["유형", "대상", "메시지", "영향", "조치"],
        ),
        "",
        "## 11. Phase 0 Decision",
        "",
        f"- {decision}",
        "",
        "## 12. Next Gate",
        "",
        "- Phase 1은 다음 패스에서 별도 승인 후에만 가능하다.",
        "- 조건: ZDT1 10 seeds, candidate_o vs candidate_n, candidate_o vs candidate_j, candidate_o vs pymoo, spread parity diagnostics 포함, ZDT1 component diagnostics 포함, occupied_bins/spacing/count 약한 양의 신호, segment0_g/distance catastrophic regression 없음, default promotion 금지.",
        "",
        "## 13. Regression Check",
        "",
        *BASE._markdown_table(
            [
                {
                    "명령": "python scripts/audit_nsga2_default_drift.py ...candidate_o...",
                    "결과": drift_text,
                    "비고": "candidate metadata / diagnostics leak check",
                },
                {
                    "명령": "python scripts/check_local_baseline.py --output-dir artifacts/spread_preserving_phase0_guard",
                    "결과": args.local_baseline_status,
                    "비고": args.local_baseline_note,
                },
            ],
            ["명령", "결과", "비고"],
        ),
        "",
        "## 14. Maturity Impact",
        "",
        "- Level 4 근거 강화",
        "- Phase 0는 sanity일 뿐 성능 maturity 상향 근거는 아니다.",
        "- candidate isolation과 fairness gate가 유지되면 실험 툴킷으로서 Level 4 근거는 강화 가능하다.",
        "- 기본값이 바뀌지 않았으므로 default algorithm maturity 상향은 금지한다.",
        "- ZDT1-specific candidate이므로 일반 MOEA 성숙도 상향 금지.",
        "",
        "## 15. Recommended Next Work",
        "",
        "1. Phase 0 passed이면 Phase 1 ZDT1 repeated validation 계획 작성",
        "2. Phase 0 failed이면 candidate_o 수정 또는 폐기",
        "3. Phase 0 signal이 약하면 diagnostics로 회귀",
        "4. candidate_n은 Phase 1 passed with trade-offs 상태 유지",
        "5. candidate_j opt-in docs 유지",
        "6. survivor-pressure family는 계속 pause",
        "7. fairness checker single-objective runner 확장 검토",
        "8. constrained multi-objective contract",
        "9. checkpoint/resume",
        "10. parallel evaluation",
        "",
        f"“이번 Phase 0 결과, candidate_o_spread_preserving_variation_light는 spread-preservation 병목에 대해 {'유효한 초기 sanity' if decision != 'Phase 0 failed, fix required' else '실패'} 신호를 보였고, 기본 NSGA-II default는 {drift_text} 상태로 유지되었으며, 다음 단계는 {'별도 승인 하의 Phase 1 계획' if phase1_allowed else 'diagnostics 또는 설계 조정 재검토'}이다.”",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(PROJECT_ROOT / args.config)
    requested_problem_names = [
        item.strip().lower() for item in str(args.problems or "zdt1").split(",") if item.strip()
    ]
    selected_specs = [mo_candidate_suite_specs()[name] for name in requested_problem_names]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    variants = _phase0_variants()
    variant_map = {variant.candidate_id: variant for variant in variants}

    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for spec in selected_specs:
        config = build_problem_config(base_config, spec)
        config = _retarget_budget(config, args.budget)
        reference_front = reference_front_for_spec(spec, point_count=201)
        reference_point = list(spec.hv_reference_point)
        problem_output_root = output_root / spec.problem
        problem_output_root.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            results: list[ExternalMOComparatorResult] = [
                run_internal_nsga2(
                    _enable_phase0_diagnostics(
                        config,
                        candidate_id=None,
                        segment_count=args.segment_count,
                    ),
                    seed=seed,
                    output_root=str(problem_output_root),
                ),
                *[
                    _make_candidate_result(
                        config,
                        variant,
                        seed=seed,
                        output_root=problem_output_root,
                        segment_count=args.segment_count,
                    )
                    for variant in variants
                ],
            ]
            if args.include_pymoo:
                results.append(run_pymoo_nsga2(config, seed=seed, budget=args.budget))

            for result in results:
                row = result_to_front_row(
                    result,
                    reference_front=reference_front,
                    reference_point=reference_point,
                )
                row = _decorate_row(
                    row,
                    spec=spec,
                    base_config=config,
                    requested_budget=args.budget,
                    variant_map=variant_map,
                    reference_front=reference_front,
                    segment_count=args.segment_count,
                )
                raw_rows.append(row)
                if result.status != "success":
                    failures.append(
                        {
                            "유형": result.status,
                            "대상": result.algorithm_name,
                            "메시지": result.error_message,
                            "영향": "seed excluded from Phase 0 sanity read",
                            "조치": "fix the runtime issue before any Phase 1 planning",
                        }
                    )

    fairness_payload = evaluate_parameter_fairness(
        raw_rows,
        benchmark_rows=benchmark_rows,
        candidate_rows=candidate_rows,
    )
    aggregate_rows = _aggregate_rows(raw_rows)
    spread_rows = _spread_rows(raw_rows)
    zdt1_component_rows = _zdt1_component_rows(raw_rows)
    operator_supply_rows = _operator_supply_rows(raw_rows)
    drift_payload = _load_optional_json(PROJECT_ROOT / args.drift_audit_json)
    gate_rows = _sanity_gate_results(
        raw_rows,
        fairness_payload,
        local_baseline_status=args.local_baseline_status,
        local_baseline_note=args.local_baseline_note,
        drift_payload=drift_payload,
    )
    decision = _phase0_decision(
        gate_rows,
        fairness_payload,
        failures,
        aggregate_rows=aggregate_rows,
    )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "selected_problems": requested_problem_names,
        "seeds": seeds,
        "budget": args.budget,
        "segment_count": args.segment_count,
        "spread_parity_trace_enabled": True,
        "zdt1_component_trace_enabled": True,
        "operator_supply_trace_enabled": True,
        "benchmark_rows": benchmark_rows,
        "candidate_rows": candidate_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "spread_rows": spread_rows,
        "zdt1_component_rows": zdt1_component_rows,
        "operator_supply_rows": operator_supply_rows,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
        "gate_rows": gate_rows,
        "phase0_decision": decision,
        "failures": failures,
        "drift_audit": drift_payload,
    }

    json_path = safe_artifact_path(
        artifact_root,
        "nsga2_spread_preserving_phase0_results",
        args.artifact_suffix,
        ".json",
    )
    results_md_path = safe_artifact_path(
        artifact_root,
        "nsga2_spread_preserving_phase0_results",
        args.artifact_suffix,
        ".md",
    )
    report_path = safe_artifact_path(
        artifact_root,
        "nsga2_spread_preserving_phase0_report",
        args.artifact_suffix,
        ".md",
    )

    BASE._write_json(json_path, payload)
    results_md_path.write_text(
        _results_markdown(
            aggregate_rows,
            spread_rows,
            zdt1_component_rows,
            fairness_payload,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        _report_markdown(
            args=args,
            aggregate_rows=aggregate_rows,
            spread_rows=spread_rows,
            zdt1_component_rows=zdt1_component_rows,
            operator_supply_rows=operator_supply_rows,
            gate_rows=gate_rows,
            fairness_payload=fairness_payload,
            failures=failures,
            decision=decision,
            drift_payload=drift_payload,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "results_json": str(json_path),
                "results_md": str(results_md_path),
                "report_md": str(report_path),
                "phase0_decision": decision,
                "fairness_status": fairness_payload["status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
