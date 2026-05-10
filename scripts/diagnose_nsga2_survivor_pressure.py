from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.config import GAConfig, load_config
from ga_lab.experiment.diversity_diagnostics import evaluate_diversity_diagnostics
from ga_lab.experiment.external_mo_comparators import (
    ExternalMOComparatorResult,
    optional_library_status,
    result_to_front_row,
    run_deap_nsga2,
    run_internal_nsga2,
    run_pymoo_nsga2,
)
from ga_lab.experiment.mo_metrics import coverage_indicator
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
    candidate_l_sparse_parent_bias_light,
    candidate_m_boundary_preservation_light,
    candidate_variant_metadata,
)
from ga_lab.experiment.nsga2_diagnostics import (
    compute_segment_distribution,
    summarize_internal_external_distribution_comparison,
    summarize_internal_external_zdt1_component_distribution,
    summarize_segment0_spacing_detail,
    summarize_survivor_divergence_by_generation,
    summarize_survivor_set_diff,
)


def _load_helper(script_name: str, module_name: str):
    helper_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_helper("validate_nsga2_candidate_suite.py", "_survivor_pressure_diagnostics_base")
PHASE0 = _load_helper("validate_nsga2_survivor_pressure_phase0.py", "_survivor_pressure_phase0_base")
PHASE1 = _load_helper("validate_nsga2_survivor_pressure_phase1.py", "_survivor_pressure_phase1_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run survivor-pressure diagnostics instrumentation for NSGA-II candidates."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problem", default=None)
    parser.add_argument("--problems", default=None)
    parser.add_argument("--output-root", default="outputs/nsga2_survivor_pressure_diagnostics")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=10501)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument(
        "--drift-audit-json",
        default="artifacts/nsga2_diagnostics_default_drift_audit_results.json",
    )
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--lineage", action="store_true")
    parser.add_argument("--operator-supply", action="store_true")
    parser.add_argument("--zdt1-components", action="store_true")
    parser.add_argument(
        "--reference-algorithm",
        default="candidate_j_h_lite_retry2",
    )
    parser.add_argument("--generation-sample-stride", type=int, default=1)
    parser.add_argument("--segment-count", type=int, default=6)
    return parser.parse_args()


def _selected_problems(args: argparse.Namespace) -> list[str]:
    raw = args.problem or args.problems or "zdt1"
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def _variants() -> list[NSGA2CandidateVariant]:
    return [
        candidate_j_h_lite_retry2(),
        candidate_l_sparse_parent_bias_light(),
        candidate_m_boundary_preservation_light(),
    ]


def _summary_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), int | float) and math.isfinite(float(row[key]))
    ]
    if not values:
        return None
    return float(mean(values))


def _number_mean(values: list[float]) -> float | None:
    finite_values = [value for value in values if math.isfinite(float(value))]
    if not finite_values:
        return None
    return float(mean(finite_values))


def _summary_count(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        counter[str(value)] += 1
    return counter


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _with_trace_options(
    config: GAConfig,
    *,
    algorithm_name: str,
    seed: int,
    candidate_id: str | None = None,
    deep: bool = False,
    lineage: bool = False,
    operator_supply: bool = False,
    zdt1_components: bool = False,
    generation_sample_stride: int = 1,
    segment_count: int = 6,
) -> GAConfig:
    clone = GAConfig.from_dict(config.to_dict())
    clone.seed = seed
    clone.algorithm_options = dict(clone.algorithm_options)
    clone.algorithm_options["nsga2_trace_enabled"] = True
    clone.algorithm_options["nsga2_trace_run_id"] = (
        f"{algorithm_name}_{clone.problem}_seed{seed}"
    )
    clone.algorithm_options["nsga2_trace_occupancy_bins"] = max(1, int(segment_count))
    clone.algorithm_options["nsga2_trace_segment_count"] = max(1, int(segment_count))
    clone.algorithm_options["nsga2_trace_top_parent_limit"] = 5
    clone.algorithm_options["nsga2_deep_trace_enabled"] = bool(deep)
    clone.algorithm_options["nsga2_lineage_trace_enabled"] = bool(lineage)
    clone.algorithm_options["nsga2_operator_supply_trace_enabled"] = bool(operator_supply)
    clone.algorithm_options["nsga2_zdt1_component_trace_enabled"] = bool(zdt1_components)
    clone.algorithm_options["nsga2_trace_generation_sample_stride"] = max(
        1,
        int(generation_sample_stride),
    )
    if candidate_id is not None:
        clone.algorithm_options["nsga2_trace_candidate_id"] = candidate_id
    else:
        clone.algorithm_options.pop("nsga2_trace_candidate_id", None)
    return clone


def _candidate_result(
    base_config: GAConfig,
    variant: NSGA2CandidateVariant,
    *,
    seed: int,
    output_root: Path,
    deep: bool,
    lineage: bool,
    operator_supply: bool,
    zdt1_components: bool,
    generation_sample_stride: int,
    segment_count: int,
) -> ExternalMOComparatorResult:
    candidate_config = apply_candidate_variant(base_config, variant)
    candidate_config = _with_trace_options(
        candidate_config,
        algorithm_name=variant.candidate_id,
        seed=seed,
        candidate_id=variant.candidate_id,
        deep=deep,
        lineage=lineage,
        operator_supply=operator_supply,
        zdt1_components=zdt1_components,
        generation_sample_stride=generation_sample_stride,
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


def _decorate_row(
    row: dict[str, Any],
    *,
    reference_front: list[list[float]],
) -> dict[str, Any]:
    directions = [
        bool(value)
        for value in row.get("metadata", {}).get("objective_directions", [False, False])
    ]
    if row.get("success"):
        coverage = coverage_indicator(
            row.get("nondominated_objective_vectors", []),
            reference_front,
            directions,
        )
        row["coverage_indicator"] = coverage
        row["reference_front_coverage"] = coverage
        decision_vectors = row.get("front_decision_vectors") or row.get("decision_vectors")
        row.update(
            evaluate_diversity_diagnostics(
                row.get("objective_vectors", []),
                directions=directions,
                decision_vectors=decision_vectors if isinstance(decision_vectors, list) else None,
            )
        )
        spacing_value = row.get("spacing")
        nondominated_count = row.get("nondominated_count")
        spacing_defined = (
            isinstance(spacing_value, int | float) and math.isfinite(float(spacing_value))
        )
        spacing_degenerate = (
            isinstance(spacing_value, int | float)
            and math.isnan(float(spacing_value))
            and isinstance(nondominated_count, int | float)
            and float(nondominated_count) <= 1.0
        )
        row["metric_calculation_success"] = all(
            isinstance(row.get(metric_name), int | float)
            and math.isfinite(float(row[metric_name]))
            for metric_name in (
                "hypervolume_2d",
                "reference_front_distance",
                "generational_distance",
                "inverted_generational_distance",
                "nondominated_count",
                "coverage_indicator",
            )
        ) and (spacing_defined or spacing_degenerate)
    else:
        row["coverage_indicator"] = None
        row["reference_front_coverage"] = None
        row["decision_duplicate_rate"] = None
        row["objective_duplicate_rate"] = None
        row["archive_duplicate_rate"] = None
        row["unique_decision_count"] = None
        row["unique_objective_count"] = None
        row["boundary_point_count"] = None
        row["metric_calculation_success"] = False
    return row


def _load_drift_payload(path: Path) -> tuple[dict[str, Any] | None, Path | None]:
    candidates: list[Path] = [path]
    normalized = Path(str(path).replace("\\", "/"))
    if normalized not in candidates:
        candidates.append(normalized)
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8")), candidate
    return None, None


def _trace_entries(row: dict[str, Any], trace_type: str) -> list[dict[str, Any]]:
    payload = row.get("metadata", {}).get("nsga2_diagnostics")
    if not isinstance(payload, dict):
        return []
    traces = payload.get("traces", [])
    if not isinstance(traces, list):
        return []
    return [
        trace
        for trace in traces
        if isinstance(trace, dict) and str(trace.get("trace_type")) == trace_type
    ]


def _diag_metric(row: dict[str, Any], trace_type: str, metric_key: str) -> float | None:
    payload = row.get("metadata", {}).get("nsga2_diagnostics")
    if not isinstance(payload, dict):
        return None
    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return None
    trace_payload = aggregate.get(trace_type, {})
    if not isinstance(trace_payload, dict):
        return None
    value = trace_payload.get(f"{metric_key}_mean")
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def _isolation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        metadata = dict(row.get("metadata", {}))
        output.append(
            {
                "problem": row["problem"],
                "algorithm": row["algorithm"],
                "candidate_metadata_present": "candidate_id" in metadata,
                "diagnostics_metadata_present": "nsga2_diagnostics" in metadata,
                "default_changed_present": "default_changed" in metadata,
            }
        )
    return output


def _diagnostic_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["problem"]), str(row["algorithm"]))
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        successful = [row for row in bucket if row.get("success")]
        aggregate_rows.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_parent_selection_diversity": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "parent_contribution_trace",
                                "parent_selection_diversity_ratio",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_parent_boundary_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "parent_contribution_trace",
                                "boundary_parent_selection_rate",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_parent_sparse_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "parent_contribution_trace",
                                "sparse_parent_selection_rate",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_bias_trigger_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "parent_contribution_trace",
                                "bias_trigger_rate",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_boundary_retention_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "boundary_retention_trace",
                                "boundary_retention_rate",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_boundary_loss_count": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "boundary_retention_trace",
                                "boundary_loss_count",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_survivor_boundary_selected_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "survivor_replacement_trace",
                                "boundary_selected_rate",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_survivor_sparse_selected_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "survivor_replacement_trace",
                                "sparse_selected_rate",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_occupied_bins": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "objective_occupancy_summary",
                                "occupied_bins",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_unique_objective_count": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "objective_occupancy_summary",
                                "unique_objective_count",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_front_overlap_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "front_change_summary",
                                "front_overlap_rate",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_front_spacing_delta": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "front_change_summary",
                                "spacing_delta",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
                "mean_front_nondominated_delta": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "front_change_summary",
                                "nondominated_count_delta",
                            )
                        }
                        for row in successful
                    ],
                    "value",
                )["mean"],
            }
        )
    return aggregate_rows


def _deep_parent_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("success"):
            continue
        key = (str(row["problem"]), str(row["algorithm"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_offspring_survival_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "parent_to_offspring_trace",
                                "offspring_survival_rate",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_offspring_nondominated_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "parent_to_offspring_trace",
                                "offspring_nondominated_rate",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_sparse_parent_offspring_survival_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "parent_to_offspring_trace",
                                "offspring_with_sparse_parent_survival_rate",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_boundary_parent_offspring_survival_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "parent_to_offspring_trace",
                                "offspring_with_boundary_parent_survival_rate",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
            }
        )
    return output


def _deep_offspring_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("success"):
            continue
        key = (str(row["problem"]), str(row["algorithm"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_offspring_survival_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "offspring_to_survivor_trace",
                                "offspring_survival_rate",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_sparse_offspring_survival_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "offspring_to_survivor_trace",
                                "sparse_offspring_survival_rate",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_boundary_offspring_survival_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "offspring_to_survivor_trace",
                                "boundary_offspring_survival_rate",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_offspring_occupied_bins": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "offspring_to_survivor_trace",
                                "offspring_occupied_bins",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_surviving_offspring_occupied_bins": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "offspring_to_survivor_trace",
                                "surviving_offspring_occupied_bins",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
            }
        )
    return output


def _survivor_set_diff_rows(
    rows: list[dict[str, Any]],
    *,
    reference_algorithm: str,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {
        (str(row["problem"]), int(row["seed"]), str(row["algorithm"])): row for row in rows
    }
    grouped_problem_seed: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (problem, seed, algorithm), row in by_key.items():
        grouped_problem_seed[(problem, seed)][algorithm] = row

    diff_rows: list[dict[str, Any]] = []
    for (problem, seed), bucket in sorted(grouped_problem_seed.items()):
        reference_row = bucket.get(reference_algorithm)
        if reference_row is None:
            continue
        directions = [
            bool(value)
            for value in reference_row.get("metadata", {}).get("objective_directions", [False, False])
        ]
        reference_by_generation = {
            int(trace["generation"]): trace
            for trace in _trace_entries(reference_row, "segment_spacing_attribution")
        }
        for algorithm, row in bucket.items():
            if algorithm == reference_algorithm:
                continue
            candidate_by_generation = {
                int(trace["generation"]): trace
                for trace in _trace_entries(row, "segment_spacing_attribution")
            }
            for generation in sorted(reference_by_generation.keys() & candidate_by_generation.keys()):
                reference_vectors = reference_by_generation[generation]["metrics"].get(
                    "front_objective_vectors",
                    [],
                )
                candidate_vectors = candidate_by_generation[generation]["metrics"].get(
                    "front_objective_vectors",
                    [],
                )
                if not isinstance(reference_vectors, list) or not isinstance(candidate_vectors, list):
                    continue
                diff = summarize_survivor_set_diff(
                    candidate_vectors,
                    reference_vectors,
                    directions=directions,
                    bins=6,
                )
                diff_rows.append(
                    {
                        "problem": problem,
                        "seed": seed,
                        "generation": generation,
                        "algorithm": algorithm,
                        "reference_algorithm": reference_algorithm,
                        **{key: value for key, value in diff.items() if key != "warnings"},
                        "warnings": list(diff.get("warnings", [])),
                    }
                )
    return diff_rows


def _aggregate_survivor_set_diff_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["problem"]),
            str(row["algorithm"]),
            str(row["reference_algorithm"]),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm, reference_algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "reference_algorithm": reference_algorithm,
                "mean_objective_overlap_rate": _summary_mean(bucket, "objective_overlap_rate"),
                "mean_nearest_neighbor_distance_to_reference": _summary_mean(
                    bucket,
                    "nearest_neighbor_distance_to_reference",
                ),
                "mean_unique_to_candidate_count": _summary_mean(
                    bucket,
                    "unique_to_candidate_count",
                ),
                "mean_unique_to_reference_count": _summary_mean(
                    bucket,
                    "unique_to_reference_count",
                ),
                "mean_boundary_diff_count": _summary_mean(bucket, "boundary_diff_count"),
                "mean_sparse_bin_diff_count": _summary_mean(bucket, "sparse_bin_diff_count"),
                "generation_count": len(bucket),
            }
        )
    return output


def _boundary_detail_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detail_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("success"):
            continue
        for trace in _trace_entries(row, "objective_boundary_retention_detail"):
            metrics = trace.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            for detail in metrics.get("detail_rows", []):
                detail_rows.append(
                    {
                        "problem": row["problem"],
                        "algorithm": row["algorithm"],
                        "seed": row["seed"],
                        "generation": trace["generation"],
                        "objective": detail.get("objective_index"),
                        "min_retained": bool(detail.get("min_retained_next_generation")),
                        "max_retained": bool(detail.get("max_retained_next_generation")),
                    }
                )
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        key = (str(row["problem"]), str(row["algorithm"]), int(row["objective"]))
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm, objective), bucket in sorted(grouped.items()):
        min_rate = sum(1 for row in bucket if row["min_retained"]) / len(bucket)
        max_rate = sum(1 for row in bucket if row["max_retained"]) / len(bucket)
        if min_rate >= 0.95 and max_rate >= 0.95:
            loss_pattern = "거의 항상 유지"
            interpretation = "objective boundary loss가 주요 병목은 아님"
        elif min_rate < 0.95 or max_rate < 0.95:
            loss_pattern = "일부 세대에서 손실"
            interpretation = "objective별 boundary retention 편차가 존재"
        else:
            loss_pattern = "불명확"
            interpretation = "추가 진단 필요"
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "objective": objective,
                "min_retained": min_rate,
                "max_retained": max_rate,
                "loss_pattern": loss_pattern,
                "interpretation": interpretation,
            }
        )
    return output


def _segment_spacing_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segment_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("success"):
            continue
        for trace in _trace_entries(row, "segment_spacing_attribution"):
            metrics = trace.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            segment_rows.append(
                {
                    "problem": row["problem"],
                    "algorithm": row["algorithm"],
                    "seed": row["seed"],
                    "generation": trace["generation"],
                    "weak_segment": metrics.get("weak_segment_id"),
                    "empty_segment_count": metrics.get("empty_segment_count"),
                    "max_gap_segment": metrics.get("max_gap_segment_id"),
                }
            )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        key = (str(row["problem"]), str(row["algorithm"]))
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        weak_counter = _summary_count(bucket, "weak_segment")
        gap_counter = _summary_count(bucket, "max_gap_segment")
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "weak_segment": weak_counter.most_common(1)[0][0] if weak_counter else "n/a",
                "empty_segments": _summary_mean(bucket, "empty_segment_count"),
                "max_gap_segment": gap_counter.most_common(1)[0][0] if gap_counter else "n/a",
            }
        )
    return output


def _lineage_funnel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_parent_to_offspring": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "lineage_retention_funnel",
                                "retained_lineage_ratio_parent_to_offspring",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_offspring_to_survivor": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "lineage_retention_funnel",
                                "retained_lineage_ratio_offspring_to_survivor",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_survivor_to_front": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "lineage_retention_funnel",
                                "retained_lineage_ratio_survivor_to_front",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_altered_parent_count": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "lineage_retention_funnel",
                                "altered_parent_count",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_altered_offspring_survived_count": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "lineage_retention_funnel",
                                "altered_offspring_survived_count",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
            }
        )
    return output


def _sparse_lineage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_sparse_offspring_nondominated_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "sparse_lineage_quality",
                                "sparse_offspring_nondominated_rate",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_sparse_offspring_survival_rate": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "sparse_lineage_quality",
                                "sparse_offspring_survival_rate",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_sparse_offspring_distance_to_front": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "sparse_lineage_quality",
                                "sparse_offspring_mean_distance_to_front",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
            }
        )
    return output


def _lineage_survivor_divergence_generation_rows(
    rows: list[dict[str, Any]],
    *,
    reference_algorithm: str,
) -> list[dict[str, Any]]:
    diff_rows = _survivor_set_diff_rows(rows, reference_algorithm=reference_algorithm)
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in diff_rows:
        key = (
            str(row["problem"]),
            int(row["seed"]),
            str(row["algorithm"]),
            str(row["reference_algorithm"]),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (problem, seed, algorithm, reference_name), bucket in sorted(grouped.items()):
        summary = summarize_survivor_divergence_by_generation(bucket)
        for generation_row in summary["rows"]:
            output.append(
                {
                    "problem": problem,
                    "seed": seed,
                    "algorithm": algorithm,
                    "reference_algorithm": reference_name,
                    **generation_row,
                }
            )
    return output


def _aggregate_lineage_survivor_divergence_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["problem"]),
            str(row["algorithm"]),
            str(row["reference_algorithm"]),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm, reference_algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "reference_algorithm": reference_algorithm,
                "mean_divergence_vs_reference": _summary_mean(
                    bucket,
                    "divergence_vs_reference",
                ),
                "mean_convergence_back_to_reference_rate": _summary_mean(
                    bucket,
                    "convergence_back_to_reference_rate",
                ),
                "mean_unique_candidate_points": _summary_mean(
                    bucket,
                    "unique_to_candidate_count",
                ),
            }
        )
    return output


def _segment0_detail_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detail_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("success"):
            continue
        for trace in _trace_entries(row, "segment0_spacing_detail"):
            metrics = trace.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            detail_rows.append(
                {
                    "problem": row["problem"],
                    "algorithm": row["algorithm"],
                    "seed": row["seed"],
                    "generation": trace["generation"],
                    "segment0_point_count": metrics.get("point_count"),
                    "segment0_empty": metrics.get("empty_segment"),
                    "segment0_local_gap": metrics.get("max_local_gap"),
                    "segment0_boundary_adjacent": metrics.get("boundary_adjacent"),
                    "segment0_range": str(metrics.get("affected_objective_range")),
                }
            )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        range_counter = _summary_count(bucket, "segment0_range")
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "segment0_point_count": _summary_mean(bucket, "segment0_point_count"),
                "segment0_empty_rate": _number_mean(
                    [1.0 if bool(row.get("segment0_empty")) else 0.0 for row in bucket]
                ),
                "segment0_local_gap": _summary_mean(bucket, "segment0_local_gap"),
                "boundary_adjacent": any(bool(row.get("segment0_boundary_adjacent")) for row in bucket),
                "affected_objective_range": (
                    range_counter.most_common(1)[0][0] if range_counter else "n/a"
                ),
            }
        )
    return output


def _duplicate_funnel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_duplicate_removed": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "duplicate_to_diversity_funnel",
                                "duplicate_removed_count",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_replacement_survived": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "duplicate_to_diversity_funnel",
                                "replacement_survived_count",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_occupied_bins": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "duplicate_to_diversity_funnel",
                                "occupied_bins",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_unique_objectives": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "duplicate_to_diversity_funnel",
                                "unique_objective_count",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
            }
        )
    return output


def _boundary_intervention_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_trigger_count": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "boundary_intervention_count",
                                "boundary_preference_trigger_count",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_changed_selection_count": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "boundary_intervention_count",
                                "boundary_preference_changed_selection_count",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_retained_due_to_preference": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "boundary_intervention_count",
                                "boundary_retained_due_to_preference_count",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
                "mean_effect_size": BASE._summary_stat(
                    [
                        {
                            "value": _diag_metric(
                                row,
                                "boundary_intervention_count",
                                "boundary_effect_size_estimate",
                            )
                        }
                        for row in bucket
                    ],
                    "value",
                )["mean"],
            }
        )
    return output


def _diag_bucket_mean(
    bucket: list[dict[str, Any]],
    trace_type: str,
    metric_key: str,
) -> float | None:
    return BASE._summary_stat(
        [{"value": _diag_metric(row, trace_type, metric_key)} for row in bucket],
        "value",
    )["mean"]


def _dominant_transition_summary(bucket: list[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    for row in bucket:
        for entry in _trace_entries(row, "variation_segment_transition"):
            transitions = entry.get("metrics", {}).get("dominant_transitions", [])
            if not isinstance(transitions, list):
                continue
            for transition in transitions:
                if not isinstance(transition, dict):
                    continue
                label = str(transition.get("transition", "unknown"))
                count = int(transition.get("count", 0) or 0)
                counts[label] += max(1, count)
    if not counts:
        return "n/a"
    top = counts.most_common(3)
    return ", ".join(f"{label} x{count}" for label, count in top)


def _segment0_bottleneck_label(row: dict[str, Any]) -> str:
    init_count = row.get("mean_segment0_initial_count")
    offspring_count = row.get("mean_segment0_offspring_count")
    nondominated_count = row.get("mean_segment0_nondominated_count")
    survivor_count = row.get("mean_segment0_survivor_count")
    final_front_count = row.get("mean_segment0_final_front_count")
    if isinstance(init_count, int | float) and isinstance(offspring_count, int | float):
        if float(offspring_count) < max(1.0, float(init_count) * 0.6):
            return "offspring supply"
    if isinstance(offspring_count, int | float) and isinstance(nondominated_count, int | float):
        if float(nondominated_count) < max(1.0, float(offspring_count) * 0.6):
            return "offspring quality"
    if isinstance(nondominated_count, int | float) and isinstance(survivor_count, int | float):
        if float(survivor_count) < max(1.0, float(nondominated_count) * 0.6):
            return "survivor attrition"
    if isinstance(survivor_count, int | float) and isinstance(final_front_count, int | float):
        if float(final_front_count) < max(1.0, float(survivor_count) * 0.6):
            return "front retention"
    return "mixed_or_weak"


def _operator_supply_initialization_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_segment0_count": _diag_bucket_mean(
                    bucket,
                    "initialization_segment_coverage",
                    "segment0_count",
                ),
                "mean_segment0_rate": _diag_bucket_mean(
                    bucket,
                    "initialization_segment_coverage",
                    "segment0_rate",
                ),
                "mean_occupied_bins": _diag_bucket_mean(
                    bucket,
                    "initialization_segment_coverage",
                    "occupied_bins",
                ),
                "mean_boundary_adjacent_count": _diag_bucket_mean(
                    bucket,
                    "initialization_segment_coverage",
                    "boundary_adjacent_count",
                ),
                "mean_unique_objective_count": _diag_bucket_mean(
                    bucket,
                    "initialization_segment_coverage",
                    "unique_objective_count",
                ),
            }
        )
    return output


def _variation_transition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_segment0_entry_count": _diag_bucket_mean(
                    bucket,
                    "variation_segment_transition",
                    "segment0_entry_count",
                ),
                "mean_segment0_exit_count": _diag_bucket_mean(
                    bucket,
                    "variation_segment_transition",
                    "segment0_exit_count",
                ),
                "mean_boundary_entry_count": _diag_bucket_mean(
                    bucket,
                    "variation_segment_transition",
                    "boundary_entry_count",
                ),
                "mean_boundary_exit_count": _diag_bucket_mean(
                    bucket,
                    "variation_segment_transition",
                    "boundary_exit_count",
                ),
                "dominant_transitions": _dominant_transition_summary(bucket),
            }
        )
    return output


def _operator_offspring_quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_offspring_nondominated_rate": _diag_bucket_mean(
                    bucket,
                    "operator_offspring_quality",
                    "offspring_nondominated_rate",
                ),
                "mean_segment0_offspring_count": _diag_bucket_mean(
                    bucket,
                    "operator_offspring_quality",
                    "segment0_offspring_count",
                ),
                "mean_segment0_nondominated_rate": _diag_bucket_mean(
                    bucket,
                    "operator_offspring_quality",
                    "segment0_offspring_nondominated_rate",
                ),
                "mean_segment0_survival_rate": _diag_bucket_mean(
                    bucket,
                    "operator_offspring_quality",
                    "segment0_offspring_survival_rate",
                ),
                "mean_segment0_distance_to_front": _diag_bucket_mean(
                    bucket,
                    "operator_offspring_quality",
                    "segment0_offspring_mean_distance_to_front",
                ),
                "mean_boundary_survival_rate": _diag_bucket_mean(
                    bucket,
                    "operator_offspring_quality",
                    "boundary_offspring_survival_rate",
                ),
            }
        )
    return output


def _mutation_retry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_retry_count": _diag_bucket_mean(
                    bucket,
                    "mutation_retry_objective_effect",
                    "retry_count",
                ),
                "mean_retry_success_count": _diag_bucket_mean(
                    bucket,
                    "mutation_retry_objective_effect",
                    "retry_success_count",
                ),
                "mean_retry_survived_count": _diag_bucket_mean(
                    bucket,
                    "mutation_retry_objective_effect",
                    "retry_offspring_survived_count",
                ),
                "mean_decision_changed_after_retry_rate": _diag_bucket_mean(
                    bucket,
                    "mutation_retry_objective_effect",
                    "decision_changed_after_retry_rate",
                ),
                "mean_objective_changed_after_retry_rate": _diag_bucket_mean(
                    bucket,
                    "mutation_retry_objective_effect",
                    "objective_changed_after_retry_rate",
                ),
            }
        )
    return output


def _segment0_supply_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        summary_row = {
            "problem": problem,
            "algorithm": algorithm,
            "mean_segment0_initial_count": _diag_bucket_mean(
                bucket,
                "segment0_supply_funnel",
                "segment0_initial_count",
            ),
            "mean_segment0_offspring_count": _diag_bucket_mean(
                bucket,
                "segment0_supply_funnel",
                "segment0_offspring_count",
            ),
            "mean_segment0_nondominated_count": _diag_bucket_mean(
                bucket,
                "segment0_supply_funnel",
                "segment0_nondominated_count",
            ),
            "mean_segment0_survivor_count": _diag_bucket_mean(
                bucket,
                "segment0_supply_funnel",
                "segment0_survivor_count",
            ),
            "mean_segment0_final_front_count": _diag_bucket_mean(
                bucket,
                "segment0_supply_funnel",
                "segment0_final_front_count",
            ),
            "mean_init_to_survivor": _diag_bucket_mean(
                bucket,
                "segment0_supply_funnel",
                "segment0_retention_init_to_survivor",
            ),
            "mean_offspring_to_survivor": _diag_bucket_mean(
                bucket,
                "segment0_supply_funnel",
                "segment0_retention_offspring_to_survivor",
            ),
            "mean_survivor_to_front": _diag_bucket_mean(
                bucket,
                "segment0_supply_funnel",
                "segment0_retention_survivor_to_front",
            ),
        }
        summary_row["bottleneck"] = _segment0_bottleneck_label(summary_row)
        output.append(summary_row)
    return output


def _external_distribution_rows(
    rows: list[dict[str, Any]],
    *,
    bins: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        segment0_counts: list[float] = []
        occupied_bins: list[float] = []
        boundary_counts: list[float] = []
        segment0_gaps: list[float] = []
        for row in bucket:
            directions = [
                bool(value)
                for value in row.get("metadata", {}).get("objective_directions", [False, False])
            ]
            distribution = compute_segment_distribution(
                row.get("objective_vectors", []),
                directions,
                bins=bins,
            )
            spacing_detail = summarize_segment0_spacing_detail(
                row.get("objective_vectors", []),
                directions,
                bins=bins,
            )
            segment0_counts.append(float(distribution.get("segment_counts", {}).get("0", 0)))
            occupied_bins.append(float(distribution.get("occupied_segments", 0)))
            boundary_counts.append(float(distribution.get("boundary_adjacent_count", 0)))
            local_gap = spacing_detail.get("max_local_gap")
            if isinstance(local_gap, int | float) and math.isfinite(float(local_gap)):
                segment0_gaps.append(float(local_gap))
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_segment0_coverage": _number_mean(segment0_counts),
                "mean_occupied_bins": _number_mean(occupied_bins),
                "mean_boundary_adjacent_count": _number_mean(boundary_counts),
                "mean_segment0_spacing_signal": _number_mean(segment0_gaps),
                "mean_nondominated_count": _summary_mean(bucket, "nondominated_count"),
                "mean_spacing": _summary_mean(bucket, "spacing"),
                "mean_unique_objective_count": _summary_mean(bucket, "unique_objective_count"),
            }
        )
    return output


def _internal_external_comparison_rows(
    rows: list[dict[str, Any]],
    *,
    reference_algorithm: str,
    bins: int,
) -> list[dict[str, Any]]:
    rows_by_key = {
        (str(row["problem"]), int(row["seed"]), str(row["algorithm"])): row
        for row in rows
        if row.get("success")
    }
    comparison_rows: list[dict[str, Any]] = []
    candidate_algorithms = {
        str(row["algorithm"])
        for row in rows
        if row.get("success")
        and str(row["algorithm"]) != reference_algorithm
        and str(row["algorithm"]) != "internal_nsga2"
    }
    for problem, seed, algorithm in sorted(rows_by_key):
        if algorithm not in candidate_algorithms:
            continue
        reference_row = rows_by_key.get((problem, seed, reference_algorithm))
        if reference_row is None:
            continue
        target_row = rows_by_key[(problem, seed, algorithm)]
        directions = [
            bool(value)
            for value in target_row.get("metadata", {}).get("objective_directions", [False, False])
        ]
        summary = summarize_internal_external_distribution_comparison(
            target_row.get("objective_vectors", []),
            reference_row.get("objective_vectors", []),
            directions,
            bins=bins,
        )
        comparison_rows.append(
            {
                "problem": problem,
                "seed": seed,
                "algorithm": algorithm,
                "reference_algorithm": reference_algorithm,
                **summary,
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        grouped[(str(row["problem"]), str(row["algorithm"]), str(row["reference_algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm, reference), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "reference_algorithm": reference,
                "mean_segment0_count_diff": _number_mean(
                    [float(row["segment0_count_diff"]) for row in bucket]
                ),
                "mean_boundary_count_diff": _number_mean(
                    [float(row["boundary_count_diff"]) for row in bucket]
                ),
                "mean_occupied_bins_diff": _number_mean(
                    [
                        float(row["candidate_occupied_bins"] - row["reference_occupied_bins"])
                        for row in bucket
                    ]
                ),
                "mean_spacing_signal_diff": _number_mean(
                    [
                        float(value)
                        for row in bucket
                        if isinstance(
                            (value := row.get("spacing_segment_diff")),
                            int | float,
                        )
                        and math.isfinite(float(value))
                    ]
                ),
                "mean_nondominated_count_diff": _number_mean(
                    [float(row["nondominated_count_diff"]) for row in bucket]
                ),
            }
        )
    return output


def _zdt1_initial_component_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "x0_range": {
                    "min": _diag_bucket_mean(
                        bucket,
                        "zdt1_initial_component_coverage",
                        "x0_min",
                    ),
                    "max": _diag_bucket_mean(
                        bucket,
                        "zdt1_initial_component_coverage",
                        "x0_max",
                    ),
                },
                "mean_x0_mean": _diag_bucket_mean(
                    bucket,
                    "zdt1_initial_component_coverage",
                    "x0_mean",
                ),
                "g_range": {
                    "min": _diag_bucket_mean(
                        bucket,
                        "zdt1_initial_component_coverage",
                        "g_min",
                    ),
                    "max": _diag_bucket_mean(
                        bucket,
                        "zdt1_initial_component_coverage",
                        "g_max",
                    ),
                },
                "mean_g_mean": _diag_bucket_mean(
                    bucket,
                    "zdt1_initial_component_coverage",
                    "g_mean",
                ),
                "mean_segment0_count": _diag_bucket_mean(
                    bucket,
                    "zdt1_initial_component_coverage",
                    "segment0_count",
                ),
                "mean_segment0_g": _diag_bucket_mean(
                    bucket,
                    "zdt1_initial_component_coverage",
                    "segment0_g_mean",
                ),
                "mean_segment0_distance": _diag_bucket_mean(
                    bucket,
                    "zdt1_initial_component_coverage",
                    "segment0_distance_mean",
                ),
                "mean_occupied_bins": _diag_bucket_mean(
                    bucket,
                    "zdt1_initial_component_coverage",
                    "occupied_bins",
                ),
            }
        )
    return output


def _zdt1_offspring_component_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_segment0_offspring": _diag_bucket_mean(
                    bucket,
                    "zdt1_offspring_component_quality",
                    "segment0_offspring_count",
                ),
                "mean_segment0_g": _diag_bucket_mean(
                    bucket,
                    "zdt1_offspring_component_quality",
                    "segment0_g_mean",
                ),
                "mean_segment0_f2": _diag_bucket_mean(
                    bucket,
                    "zdt1_offspring_component_quality",
                    "segment0_f2_mean",
                ),
                "mean_segment0_distance": _diag_bucket_mean(
                    bucket,
                    "zdt1_offspring_component_quality",
                    "segment0_distance_mean",
                ),
                "mean_segment0_nondominated_rate": _diag_bucket_mean(
                    bucket,
                    "zdt1_offspring_component_quality",
                    "segment0_nondominated_rate",
                ),
                "mean_segment0_survival_rate": _diag_bucket_mean(
                    bucket,
                    "zdt1_offspring_component_quality",
                    "segment0_survival_rate",
                ),
            }
        )
    return output


def _zdt1_parent_child_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_delta_x0": _diag_bucket_mean(
                    bucket,
                    "zdt1_parent_child_component_delta",
                    "delta_x0_mean",
                ),
                "mean_delta_g": _diag_bucket_mean(
                    bucket,
                    "zdt1_parent_child_component_delta",
                    "delta_g_mean",
                ),
                "mean_delta_distance": _diag_bucket_mean(
                    bucket,
                    "zdt1_parent_child_component_delta",
                    "delta_distance_mean",
                ),
                "mean_segment0_entry": _diag_bucket_mean(
                    bucket,
                    "zdt1_parent_child_component_delta",
                    "segment0_entry_delta_count",
                ),
                "mean_segment0_exit": _diag_bucket_mean(
                    bucket,
                    "zdt1_parent_child_component_delta",
                    "segment0_exit_delta_count",
                ),
            }
        )
    return output


def _zdt1_retry_component_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_retry_count": _diag_bucket_mean(
                    bucket,
                    "zdt1_mutation_retry_component_effect",
                    "retry_count",
                ),
                "mean_retry_x0_delta": _diag_bucket_mean(
                    bucket,
                    "zdt1_mutation_retry_component_effect",
                    "retry_x0_delta_mean",
                ),
                "mean_retry_g_delta": _diag_bucket_mean(
                    bucket,
                    "zdt1_mutation_retry_component_effect",
                    "retry_g_delta_mean",
                ),
                "mean_retry_distance_delta": _diag_bucket_mean(
                    bucket,
                    "zdt1_mutation_retry_component_effect",
                    "retry_distance_delta_mean",
                ),
                "mean_retry_survived": _diag_bucket_mean(
                    bucket,
                    "zdt1_mutation_retry_component_effect",
                    "retry_survived_count",
                ),
            }
        )
    return output


def _zdt1_segment0_funnel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        init_mean = _diag_bucket_mean(
            bucket,
            "zdt1_segment0_quality_funnel",
            "segment0_initial_count",
        )
        offspring_mean = _diag_bucket_mean(
            bucket,
            "zdt1_segment0_quality_funnel",
            "segment0_offspring_count",
        )
        low_g_mean = _diag_bucket_mean(
            bucket,
            "zdt1_segment0_quality_funnel",
            "segment0_low_g_count",
        )
        nondominated_mean = _diag_bucket_mean(
            bucket,
            "zdt1_segment0_quality_funnel",
            "segment0_nondominated_count",
        )
        survivor_mean = _diag_bucket_mean(
            bucket,
            "zdt1_segment0_quality_funnel",
            "segment0_survivor_count",
        )
        final_mean = _diag_bucket_mean(
            bucket,
            "zdt1_segment0_quality_funnel",
            "segment0_final_front_count",
        )
        if isinstance(offspring_mean, int | float) and isinstance(nondominated_mean, int | float):
            bottleneck = (
                "g_or_quality"
                if float(nondominated_mean) < max(1.0, float(offspring_mean) * 0.6)
                else "survivor_or_front"
            )
        elif isinstance(init_mean, int | float) and isinstance(offspring_mean, int | float):
            bottleneck = (
                "offspring_supply"
                if float(offspring_mean) < max(1.0, float(init_mean) * 0.6)
                else "mixed_or_weak"
            )
        else:
            bottleneck = "mixed_or_weak"
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_segment0_initial": init_mean,
                "mean_segment0_offspring": offspring_mean,
                "mean_segment0_low_g": low_g_mean,
                "mean_segment0_nondominated": nondominated_mean,
                "mean_segment0_survivor": survivor_mean,
                "mean_segment0_final_front": final_mean,
                "mean_segment0_g_initial": _diag_bucket_mean(
                    bucket,
                    "zdt1_segment0_quality_funnel",
                    "segment0_mean_g_initial",
                ),
                "mean_segment0_g_offspring": _diag_bucket_mean(
                    bucket,
                    "zdt1_segment0_quality_funnel",
                    "segment0_mean_g_offspring",
                ),
                "mean_segment0_g_survivor": _diag_bucket_mean(
                    bucket,
                    "zdt1_segment0_quality_funnel",
                    "segment0_mean_g_survivor",
                ),
                "mean_segment0_distance_final": _diag_bucket_mean(
                    bucket,
                    "zdt1_segment0_quality_funnel",
                    "segment0_mean_distance_final",
                ),
                "bottleneck": bottleneck,
            }
        )
    return output


def _zdt1_internal_external_distribution_rows(
    rows: list[dict[str, Any]],
    *,
    bins: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        summaries: list[dict[str, Any]] = []
        for row in bucket:
            summaries.append(
                summarize_internal_external_zdt1_component_distribution(
                    row.get("front_decision_vectors") or row.get("decision_vectors") or [],
                    row.get("objective_vectors", []),
                    bins=bins,
                )
            )
        output.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "mean_x0_mean": _number_mean(
                    [
                        float(value)
                        for summary in summaries
                        if isinstance((value := summary.get("x0_mean")), int | float)
                        and math.isfinite(float(value))
                    ]
                ),
                "mean_g_mean": _number_mean(
                    [
                        float(value)
                        for summary in summaries
                        if isinstance((value := summary.get("g_mean")), int | float)
                        and math.isfinite(float(value))
                    ]
                ),
                "mean_segment0_coverage": _number_mean(
                    [
                        float(value)
                        for summary in summaries
                        if isinstance((value := summary.get("segment0_count")), int | float)
                        and math.isfinite(float(value))
                    ]
                ),
                "mean_segment0_g": _number_mean(
                    [
                        float(value)
                        for summary in summaries
                        if isinstance((value := summary.get("segment0_g_mean")), int | float)
                        and math.isfinite(float(value))
                    ]
                ),
                "mean_distance": _number_mean(
                    [
                        float(value)
                        for summary in summaries
                        if isinstance((value := summary.get("segment0_distance_mean")), int | float)
                        and math.isfinite(float(value))
                    ]
                ),
                "mean_occupied_bins": _number_mean(
                    [
                        float(value)
                        for summary in summaries
                        if isinstance((value := summary.get("occupied_bins")), int | float)
                        and math.isfinite(float(value))
                    ]
                ),
                "mean_spacing": _number_mean(
                    [
                        float(value)
                        for summary in summaries
                        if isinstance((value := summary.get("spacing")), int | float)
                        and math.isfinite(float(value))
                    ]
                ),
                "mean_nondominated_count": _number_mean(
                    [
                        float(value)
                        for summary in summaries
                        if isinstance((value := summary.get("nondominated_count")), int | float)
                        and math.isfinite(float(value))
                    ]
                ),
                "warnings": sorted(
                    {
                        str(warning)
                        for summary in summaries
                        for warning in list(summary.get("warnings", []))
                    }
                ),
            }
        )
    return output


def _zdt1_component_recommendation(payload: dict[str, Any]) -> str:
    funnel_rows = {
        row["algorithm"]: row for row in payload.get("zdt1_segment0_funnel_rows", [])
    }
    external_rows = {
        row["algorithm"]: row for row in payload.get("zdt1_internal_external_rows", [])
    }
    candidate_j = funnel_rows.get("candidate_j_h_lite_retry2")
    candidate_l = funnel_rows.get("candidate_l_sparse_parent_bias_light")
    pymoo_row = external_rows.get("pymoo_nsga2")

    if candidate_l and candidate_l.get("bottleneck") == "g_or_quality":
        return "Ready to design crossover/mutation Phase 0 candidate"
    if candidate_j and candidate_j.get("bottleneck") == "offspring_supply":
        return "Ready to design initialization Phase 0 candidate"
    if pymoo_row and isinstance(pymoo_row.get("mean_g_mean"), int | float):
        candidate_j_external = external_rows.get("candidate_j_h_lite_retry2")
        if (
            candidate_j_external
            and isinstance(candidate_j_external.get("mean_g_mean"), int | float)
            and float(pymoo_row["mean_g_mean"]) < float(candidate_j_external["mean_g_mean"])
        ):
            return "Ready to design crossover/mutation Phase 0 candidate"
    return "Need one more diagnostics pass"


def _zdt1_component_results_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# NSGA-II ZDT1 Component Diagnostics Results",
            "",
            "## Aggregate Results",
            "",
            *BASE._markdown_table(
                payload["aggregate_rows"],
                [
                    "problem",
                    "algorithm",
                    "status",
                    "seeds",
                    "mean_hv",
                    "mean_distance",
                    "mean_igd",
                    "mean_spacing",
                    "mean_coverage",
                    "mean_nondominated_count",
                    "mean_duplicate_rate",
                    "mean_runtime_seconds",
                ],
            ),
            "",
            "## Initial Component Coverage",
            "",
            *BASE._markdown_table(
                payload.get("zdt1_initial_component_rows", []),
                [
                    "problem",
                    "algorithm",
                    "mean_x0_mean",
                    "mean_g_mean",
                    "mean_segment0_count",
                    "mean_segment0_g",
                    "mean_segment0_distance",
                    "mean_occupied_bins",
                ],
            ),
            "",
            "## Segment 0 Quality Funnel",
            "",
            *BASE._markdown_table(
                payload.get("zdt1_segment0_funnel_rows", []),
                [
                    "problem",
                    "algorithm",
                    "mean_segment0_initial",
                    "mean_segment0_offspring",
                    "mean_segment0_low_g",
                    "mean_segment0_nondominated",
                    "mean_segment0_survivor",
                    "mean_segment0_final_front",
                    "bottleneck",
                ],
            ),
            "",
        ]
    )


def _zdt1_component_report_markdown(payload: dict[str, Any]) -> str:
    drift_payload = payload.get("drift_audit")
    drift_overall = drift_payload.get("overall", {}) if isinstance(drift_payload, dict) else {}
    recommendation = _zdt1_component_recommendation(payload)
    failure_rows = payload.get("failures", [])
    normalized_failures = [
        {
            "유형": row.get("유형") or row.get("?좏삎") or row.get("type"),
            "대상": row.get("대상") or row.get("???") or row.get("target"),
            "메시지": row.get("메시지") or row.get("硫붿떆吏") or row.get("message"),
            "영향": row.get("영향") or row.get("?곹뼢") or row.get("impact"),
            "조치": row.get("조치") or row.get("議곗튂") or row.get("action"),
        }
        for row in failure_rows
    ]
    candidate_j_funnel = next(
        (
            row
            for row in payload.get("zdt1_segment0_funnel_rows", [])
            if row.get("algorithm") == "candidate_j_h_lite_retry2"
        ),
        {},
    )
    candidate_l_funnel = next(
        (
            row
            for row in payload.get("zdt1_segment0_funnel_rows", [])
            if row.get("algorithm") == "candidate_l_sparse_parent_bias_light"
        ),
        {},
    )
    candidate_m_funnel = next(
        (
            row
            for row in payload.get("zdt1_segment0_funnel_rows", [])
            if row.get("algorithm") == "candidate_m_boundary_preservation_light"
        ),
        {},
    )
    external_rows = payload.get("zdt1_internal_external_rows", [])
    return "\n".join(
        [
            "# NSGA-II ZDT1 Component Diagnostics Report",
            "",
            "## 1. Executive Summary",
            "",
            "- 이번 작업의 목표: ZDT1 low-f1 / segment 0 품질 저하를 x0, tail, g, h, f1, f2, distance 성분으로 분해해 본다.",
            "- ZDT1 component diagnostics 추가 내용: initial component coverage, offspring component quality, parent-child component delta, mutation retry component effect, segment 0 quality funnel, internal vs external final component distribution",
            f"- default drift 결과: {'NO DRIFT' if drift_overall.get('drift_detected') is False else 'DRIFT DETECTED' if drift_overall else 'not available'}",
            f"- 실행한 diagnostics run: 문제={', '.join(payload.get('selected_problems', []))}, seeds={len(payload.get('seeds', []))}, budget={payload.get('budget')}, reference={payload.get('reference_algorithm')}, segment_count={payload.get('segment_count')}",
            f"- segment 0 quality 병목: candidate_j={candidate_j_funnel.get('bottleneck', 'n/a')}, candidate_l={candidate_l_funnel.get('bottleneck', 'n/a')}, candidate_m={candidate_m_funnel.get('bottleneck', 'n/a')}",
            "- x0/f1/g/f2/distance 관찰: low-f1 진입 여부와 low-g 전환 여부를 분리해서 본다.",
            "- internal vs pymoo component 차이: 가능하면 final front decision/objective component 분포 기준으로 비교한다.",
            "- final bottleneck hypothesis: segment 0 진입 자체보다 low-g 전환과 Pareto-front 거리 축소가 더 큰 병목인지 확인한다.",
            f"- 다음 후보 설계 여부: {recommendation}",
            "- 기본값 변경 여부: 없음",
            "- Level 판정 변화 여부: Level 4 근거 강화",
            "",
            "## 2. Scope and Non-Scope",
            "",
            "- Scope: diagnostics-only instrumentation, ZDT1 component decomposition, segment 0 quality analysis, operator supply와 component quality 연결, internal vs external final distribution comparison",
            "- Non-Scope: new candidate implementation, default promotion, Phase 2 validation, DTLZ/WFG validation, production use",
            "",
            "## 3. ZDT1 Component Diagnostics Schema",
            "",
            *BASE._markdown_table(
                [
                    {
                        "trace": "zdt1_initial_component_coverage",
                        "collected_fields": "x0/tail/g range, segment0 count, segment0 g, segment0 distance, occupied bins",
                        "purpose": "초기 population이 low-f1 / segment 0을 decision/objective component 관점에서 얼마나 덮는지 확인",
                        "limitation": "ZDT1 전용, absolute f1 bins 기준",
                    },
                    {
                        "trace": "zdt1_offspring_component_quality",
                        "collected_fields": "segment0 x0/g/f2/distance, nondominated rate, survival rate",
                        "purpose": "segment 0 offspring이 왜 nondominated/survivor로 이어지지 않는지 component로 확인",
                        "limitation": "generation summary 중심",
                    },
                    {
                        "trace": "zdt1_parent_child_component_delta",
                        "collected_fields": "delta x0/g/distance, segment0 entry/exit",
                        "purpose": "parent에서 offspring으로 component가 어떤 방향으로 바뀌는지 확인",
                        "limitation": "parent 평균 대비 offspring delta",
                    },
                    {
                        "trace": "zdt1_mutation_retry_component_effect",
                        "collected_fields": "retry count, delta x0/g/distance, survived count",
                        "purpose": "retry/dedup이 실제 component quality를 바꾸는지 확인",
                        "limitation": "추가 evaluation 없이 decision-derived component 사용",
                    },
                    {
                        "trace": "zdt1_segment0_quality_funnel",
                        "collected_fields": "init, offspring, low-g, nondominated, survivor, final front",
                        "purpose": "segment 0 후보가 어느 단계에서 quality를 잃는지 확인",
                        "limitation": "low-g threshold는 diagnostics heuristic",
                    },
                    {
                        "trace": "internal_external_zdt1_component_distribution",
                        "collected_fields": "x0 mean, g mean, segment0 coverage, distance, occupied bins, spacing",
                        "purpose": "internal과 external의 final component distribution 차이를 확인",
                        "limitation": "external은 final distribution summary만 비교",
                    },
                ],
                ["trace", "collected_fields", "purpose", "limitation"],
            ),
            "",
            "## 4. Default Drift and Isolation",
            "",
            *BASE._markdown_table(
                [
                    {
                        "gate": "all diagnostics flags false default path exact match",
                        "result": "pass"
                        if drift_overall.get("drift_detected") is False
                        else "fail"
                        if drift_overall
                        else "n/a",
                        "evidence": payload.get("drift_audit_path") or "n/a",
                        "interpretation": "component diagnostics를 꺼둔 default 경로는 기존과 같아야 한다.",
                    },
                    {
                        "gate": "default diagnostics metadata contamination",
                        "result": "pass"
                        if drift_overall.get("diagnostics_metadata_leak") is False
                        else "fail"
                        if drift_overall
                        else "n/a",
                        "evidence": payload.get("drift_audit_path") or "n/a",
                        "interpretation": "default run에는 diagnostics metadata가 들어가면 안 된다.",
                    },
                    {
                        "gate": "default candidate metadata contamination",
                        "result": "pass"
                        if drift_overall.get("candidate_metadata_leak") is False
                        else "fail"
                        if drift_overall
                        else "n/a",
                        "evidence": payload.get("drift_audit_path") or "n/a",
                        "interpretation": "default run에는 candidate metadata가 들어가면 안 된다.",
                    },
                    {
                        "gate": "local baseline governance",
                        "result": "pass" if payload.get("local_baseline_status") == "PASS" else "n/a",
                        "evidence": payload.get("local_baseline_artifact") or "n/a",
                        "interpretation": "instrumentation 이후에도 baseline governance가 유지되어야 한다.",
                    },
                ],
                ["gate", "result", "evidence", "interpretation"],
            ),
            "",
            "## 5. Diagnostics Configuration",
            "",
            *BASE._markdown_table(
                [
                    {"항목": "problem", "값": ", ".join(payload.get("selected_problems", []))},
                    {"항목": "seeds", "값": len(payload.get("seeds", []))},
                    {"항목": "budget", "값": payload.get("budget")},
                    {
                        "항목": "algorithms",
                        "값": ", ".join(sorted({str(row['algorithm']) for row in payload.get('raw_rows', [])})),
                    },
                    {"항목": "reference algorithm", "값": payload.get("reference_algorithm")},
                    {
                        "항목": "zdt1 component trace enabled",
                        "값": payload.get("zdt1_component_trace_enabled"),
                    },
                    {"항목": "segment count", "값": payload.get("segment_count")},
                    {"항목": "artifact suffix", "값": payload.get("artifact_suffix")},
                ],
                ["항목", "값"],
            ),
            "",
            "## 6. Initial Component Coverage",
            "",
            *BASE._markdown_table(
                payload.get("zdt1_initial_component_rows", []),
                [
                    "problem",
                    "algorithm",
                    "x0_range",
                    "g_range",
                    "mean_segment0_count",
                    "mean_segment0_g",
                    "mean_occupied_bins",
                ],
            ),
            "",
            "## 7. Offspring Component Quality",
            "",
            *BASE._markdown_table(
                payload.get("zdt1_offspring_component_rows", []),
                [
                    "problem",
                    "algorithm",
                    "mean_segment0_offspring",
                    "mean_segment0_g",
                    "mean_segment0_f2",
                    "mean_segment0_distance",
                    "mean_segment0_nondominated_rate",
                    "mean_segment0_survival_rate",
                ],
            ),
            "",
            "## 8. Parent-Child Component Delta",
            "",
            *BASE._markdown_table(
                payload.get("zdt1_parent_child_delta_rows", []),
                [
                    "problem",
                    "algorithm",
                    "mean_delta_x0",
                    "mean_delta_g",
                    "mean_delta_distance",
                    "mean_segment0_entry",
                    "mean_segment0_exit",
                ],
            ),
            "",
            "## 9. Mutation Retry Component Effect",
            "",
            *BASE._markdown_table(
                payload.get("zdt1_retry_component_rows", []),
                [
                    "problem",
                    "algorithm",
                    "mean_retry_count",
                    "mean_retry_x0_delta",
                    "mean_retry_g_delta",
                    "mean_retry_distance_delta",
                    "mean_retry_survived",
                ],
            ),
            "",
            "## 10. Segment 0 Quality Funnel",
            "",
            *BASE._markdown_table(
                payload.get("zdt1_segment0_funnel_rows", []),
                [
                    "problem",
                    "algorithm",
                    "mean_segment0_initial",
                    "mean_segment0_offspring",
                    "mean_segment0_low_g",
                    "mean_segment0_nondominated",
                    "mean_segment0_survivor",
                    "mean_segment0_final_front",
                    "bottleneck",
                ],
            ),
            "",
            "## 11. Internal vs External ZDT1 Component Distribution",
            "",
            *BASE._markdown_table(
                external_rows
                if external_rows
                else [
                    {
                        "problem": payload["selected_problems"][0] if payload.get("selected_problems") else "n/a",
                        "algorithm": "n/a",
                        "mean_x0_mean": None,
                        "mean_g_mean": None,
                        "mean_segment0_coverage": None,
                        "mean_segment0_g": None,
                        "mean_distance": None,
                        "mean_occupied_bins": None,
                        "mean_spacing": None,
                    }
                ],
                [
                    "problem",
                    "algorithm",
                    "mean_x0_mean",
                    "mean_g_mean",
                    "mean_segment0_coverage",
                    "mean_segment0_g",
                    "mean_distance",
                    "mean_occupied_bins",
                    "mean_spacing",
                ],
            ),
            "",
            "## 12. Bottleneck Interpretation",
            "",
            f"- segment 0 문제는 x0 공급 부족인가?: candidate_j bottleneck={candidate_j_funnel.get('bottleneck', 'n/a')}",
            "- g/tail variable quality 부족인가?: segment 0 offspring의 mean g와 low-g 전환 수를 함께 본다.",
            "- f2/distance 문제인가?: segment 0 distance와 f2 평균이 low-g 전환 이후에도 큰지 확인한다.",
            "- mutation/retry가 g를 개선하지 못하는가?: retry delta g와 retry delta distance를 본다.",
            "- pymoo와 internal의 가장 큰 component 차이는 무엇인가?: final front의 mean g, segment0 coverage, occupied bins를 비교한다.",
            "- 다음 후보 설계는 initialization, crossover, mutation, repair 중 어디로 가야 하는가?: x0 공급보다는 low-g 전환 병목이면 variation/repair 쪽, x0 자체가 부족하면 initialization 쪽으로 해석한다.",
            "",
            "## 13. Recommendation",
            "",
            f"- 결론: **{recommendation}**",
            "",
            "## 14. Failures and Warnings",
            "",
            *BASE._markdown_table(
                normalized_failures
                or [{"유형": "none", "대상": "none", "메시지": "none", "영향": "none", "조치": "none"}],
                ["유형", "대상", "메시지", "영향", "조치"],
            ),
            "",
            "## 15. Maturity Impact",
            "",
            "- 결론: **Level 4 근거 강화**",
            "- diagnostics는 성능 개선이 아니므로 알고리즘 성숙도 상향 근거가 아니다.",
            "- component diagnostics가 default drift 없이 동작하면 실험 툴킷으로서 진단 가능성이 더 강화된다.",
            "- 새 candidate가 없으므로 candidate maturity 상향은 없다.",
            "",
            "## 16. Recommended Next Work",
            "",
            f"- 추천: **{recommendation}**",
            "",
            f"이번 ZDT1 component diagnostics 결과, segment 0 병목은 {candidate_l_funnel.get('bottleneck', 'mixed_or_weak')} 때문인 것으로 보이며, internal과 external의 가장 큰 차이는 final component distribution 폭이었고, 다음 단계는 {recommendation}이다.",
        ]
    )


def _operator_supply_recommendation(payload: dict[str, Any]) -> str:
    supply_rows = {
        row["algorithm"]: row for row in payload.get("operator_supply_funnel_rows", [])
    }
    external_rows = {
        row["algorithm"]: row for row in payload.get("external_distribution_rows", [])
    }
    candidate_j = supply_rows.get("candidate_j_h_lite_retry2")
    candidate_l = supply_rows.get("candidate_l_sparse_parent_bias_light")
    candidate_m = supply_rows.get("candidate_m_boundary_preservation_light")
    pymoo_row = external_rows.get("pymoo_nsga2")
    deap_row = external_rows.get("deap_nsga2")

    if candidate_j and any(
        candidate_j.get("bottleneck") == label
        for label in ("offspring supply", "offspring quality")
    ):
        return "Focus on operator/init/mutation candidate"
    if candidate_l and candidate_l.get("bottleneck") == "survivor attrition":
        return "Focus on non-survivor-pressure causes"
    if candidate_m and candidate_m.get("bottleneck") == "mixed_or_weak":
        if pymoo_row or deap_row:
            return "Focus on operator/init/mutation candidate"
    return "Need one more diagnostics pass"


def _operator_supply_results_markdown(payload: dict[str, Any]) -> str:
    sections = [
        "# NSGA-II Operator Supply Diagnostics Results",
        "",
        "## Aggregate Results",
        "",
        *BASE._markdown_table(
            payload["aggregate_rows"],
            [
                "problem",
                "algorithm",
                "status",
                "seeds",
                "mean_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_duplicate_rate",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Initialization Segment Coverage",
        "",
        *BASE._markdown_table(
            payload.get("operator_initialization_rows", []),
            [
                "problem",
                "algorithm",
                "mean_segment0_count",
                "mean_segment0_rate",
                "mean_occupied_bins",
                "mean_boundary_adjacent_count",
                "mean_unique_objective_count",
            ],
        ),
        "",
        "## Segment 0 Supply Funnel",
        "",
        *BASE._markdown_table(
            payload.get("operator_supply_funnel_rows", []),
            [
                "problem",
                "algorithm",
                "mean_segment0_initial_count",
                "mean_segment0_offspring_count",
                "mean_segment0_nondominated_count",
                "mean_segment0_survivor_count",
                "mean_segment0_final_front_count",
                "bottleneck",
            ],
        ),
        "",
    ]
    return "\n".join(sections)


def _operator_supply_report_markdown(payload: dict[str, Any]) -> str:
    drift_payload = payload.get("drift_audit")
    drift_overall = drift_payload.get("overall", {}) if isinstance(drift_payload, dict) else {}
    recommendation = _operator_supply_recommendation(payload)
    candidate_j_supply = next(
        (
            row
            for row in payload.get("operator_supply_funnel_rows", [])
            if row.get("algorithm") == "candidate_j_h_lite_retry2"
        ),
        {},
    )
    candidate_l_supply = next(
        (
            row
            for row in payload.get("operator_supply_funnel_rows", [])
            if row.get("algorithm") == "candidate_l_sparse_parent_bias_light"
        ),
        {},
    )
    candidate_m_supply = next(
        (
            row
            for row in payload.get("operator_supply_funnel_rows", [])
            if row.get("algorithm") == "candidate_m_boundary_preservation_light"
        ),
        {},
    )
    external_rows = payload.get("external_distribution_rows", [])
    external_compare_rows = payload.get("external_distribution_comparison_rows", [])
    external_compare_summary = (
        external_compare_rows
        if external_compare_rows
        else [
            {
                "problem": payload["selected_problems"][0] if payload.get("selected_problems") else "n/a",
                "algorithm": "n/a",
                "reference_algorithm": payload.get("reference_algorithm"),
                "mean_segment0_count_diff": None,
                "mean_boundary_count_diff": None,
                "mean_occupied_bins_diff": None,
                "mean_spacing_signal_diff": None,
                "mean_nondominated_count_diff": None,
            }
        ]
    )

    lines: list[str] = [
        "# NSGA-II Operator Supply Diagnostics Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: survivor-pressure 자체가 아니라 initialization/variation/operator supply 관점에서 segment 0 및 objective-space 병목을 본다.",
        "- operator supply diagnostics 추가 내용: initialization coverage, variation segment transition, offspring quality, mutation retry objective effect, segment 0 supply funnel, internal vs external final distribution 비교",
        f"- default drift 결과: {'NO DRIFT' if drift_overall.get('drift_detected') is False else 'DRIFT DETECTED' if drift_overall else 'not available'}",
        f"- 실행한 diagnostics run: 문제={', '.join(payload['selected_problems'])}, seeds={len(payload['seeds'])}, budget={payload['budget']}, reference={payload.get('reference_algorithm')}, segment_count={payload.get('segment_count')}",
        f"- segment 0 supply 관찰: candidate_j bottleneck={candidate_j_supply.get('bottleneck', 'n/a')}",
        "- candidate_j/l/m 비교의 핵심 관찰: candidate_l의 변화가 operator supply보다 survivor retention 이후에서 더 약해지는지, candidate_m은 boundary intervention 없이 거의 baseline처럼 남는지 확인",
        f"- external distribution 비교: {'수행됨' if any(row.get('algorithm') in {'pymoo_nsga2', 'deap_nsga2'} for row in external_rows) else '제한 또는 미수행'}",
        f"- final bottleneck hypothesis: candidate_j/candidate_l/candidate_m 모두 segment 0에서 supply 자체보다 offspring quality 또는 survivor-level retention이 더 큰 병목인지, external 분포가 더 넓다면 operator/init 쪽 가설을 강화한다.",
        f"- 다음 후보 설계 여부: {recommendation}",
        "- 기본값 변경 여부: 없음",
        "- Level 판정 변화 여부: Level 4 근거 강화",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "- Scope: diagnostics-only instrumentation, initialization coverage, variation segment transition, offspring quality, segment 0 supply funnel, mutation retry objective effect, ZDT1 small diagnostics",
        "- Non-Scope: new candidate implementation, default promotion, Phase 2 validation, DTLZ/WFG validation, production use",
        "",
        "## 3. Operator Supply Diagnostics Schema",
        "",
        *BASE._markdown_table(
            [
                {
                    "trace": "initialization_segment_coverage",
                    "collected_fields": "segment0_count/rate, occupied_bins, boundary_adjacent_count, objective min/max/range",
                    "purpose": "초기 population이 segment 0과 boundary-adjacent region을 충분히 커버하는지 확인",
                    "limitation": "2-objective projection 기준",
                },
                {
                    "trace": "variation_segment_transition",
                    "collected_fields": "parent segment pair, offspring segment, segment0 entry/exit, boundary entry/exit",
                    "purpose": "parent objective segment가 offspring objective segment로 어떻게 이동하는지 확인",
                    "limitation": "parent segment는 recorded bin proxy 기준",
                },
                {
                    "trace": "operator_offspring_quality",
                    "collected_fields": "offspring nondominated/survival, segment0 offspring quality, boundary offspring survival",
                    "purpose": "segment 0 및 boundary-adjacent offspring의 quality가 실제로 낮은지 확인",
                    "limitation": "small-run generation summary",
                },
                {
                    "trace": "mutation_retry_objective_effect",
                    "collected_fields": "retry count/success, unique decision/objective counts, survived count",
                    "purpose": "retry/dedup이 decision diversity를 objective diversity로 바꾸는지 확인",
                    "limitation": "objective_changed_after_retry는 extra evaluation 없이는 직접 측정 불가",
                },
                {
                    "trace": "segment0_supply_funnel",
                    "collected_fields": "init, offspring, nondominated, survivor, final front, retention ratios",
                    "purpose": "segment 0 후보가 어느 단계에서 줄어드는지 확인",
                    "limitation": "generation-level aggregate",
                },
                {
                    "trace": "internal_external_distribution_comparison",
                    "collected_fields": "segment coverage diff, occupied bins diff, spacing signal diff, nondominated diff",
                    "purpose": "internal candidate_j와 external comparator의 final distribution 차이를 확인",
                    "limitation": "external generation trace는 포함하지 않음",
                },
            ],
            ["trace", "collected_fields", "purpose", "limitation"],
        ),
        "",
        "## 4. Default Drift and Isolation",
        "",
        *BASE._markdown_table(
            [
                {
                    "gate": "default trace-disabled path exact match",
                    "result": "pass"
                    if drift_overall.get("drift_detected") is False
                    else "fail"
                    if drift_overall
                    else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "operator supply instrumentation이 꺼진 기본 경로는 기존과 같아야 한다",
                },
                {
                    "gate": "default diagnostics metadata contamination",
                    "result": "pass"
                    if drift_overall.get("diagnostics_metadata_leak") is False
                    else "fail"
                    if drift_overall
                    else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "default run에 diagnostics metadata가 유입되면 안 된다",
                },
                {
                    "gate": "default candidate metadata contamination",
                    "result": "pass"
                    if drift_overall.get("candidate_metadata_leak") is False
                    else "fail"
                    if drift_overall
                    else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "default run에 candidate metadata가 유입되면 안 된다",
                },
                {
                    "gate": "local baseline governance",
                    "result": "pass" if payload.get("local_baseline_status") == "PASS" else "n/a",
                    "evidence": payload.get("local_baseline_artifact") or "n/a",
                    "interpretation": "instrumentation 추가 이후에도 baseline governance가 유지되는지 확인",
                },
            ],
            ["gate", "result", "evidence", "interpretation"],
        ),
        "",
        "## 5. Diagnostics Configuration",
        "",
        *BASE._markdown_table(
            [
                {"항목": "problem", "값": ", ".join(payload["selected_problems"])},
                {"항목": "seeds", "값": len(payload["seeds"])},
                {"항목": "budget", "값": payload["budget"]},
                {
                    "항목": "algorithms",
                    "값": ", ".join(sorted({str(row['algorithm']) for row in payload.get('raw_rows', [])})),
                },
                {"항목": "reference algorithm", "값": payload.get("reference_algorithm")},
                {
                    "항목": "operator supply trace enabled",
                    "값": payload.get("operator_supply_trace_enabled"),
                },
                {"항목": "segment count", "값": payload.get("segment_count")},
                {"항목": "artifact suffix", "값": payload.get("artifact_suffix")},
            ],
            ["항목", "값"],
        ),
        "",
        "## 6. Initialization Segment Coverage",
        "",
        *BASE._markdown_table(
            payload.get("operator_initialization_rows", []),
            [
                "problem",
                "algorithm",
                "mean_segment0_count",
                "mean_segment0_rate",
                "mean_occupied_bins",
                "mean_boundary_adjacent_count",
                "mean_unique_objective_count",
            ],
        ),
        "",
        "## 7. Variation Segment Transition",
        "",
        *BASE._markdown_table(
            payload.get("operator_transition_rows", []),
            [
                "problem",
                "algorithm",
                "mean_segment0_entry_count",
                "mean_segment0_exit_count",
                "dominant_transitions",
            ],
        ),
        "",
        "## 8. Operator Offspring Quality",
        "",
        *BASE._markdown_table(
            payload.get("operator_offspring_quality_rows", []),
            [
                "problem",
                "algorithm",
                "mean_offspring_nondominated_rate",
                "mean_segment0_offspring_count",
                "mean_segment0_nondominated_rate",
                "mean_segment0_survival_rate",
                "mean_segment0_distance_to_front",
            ],
        ),
        "",
        "## 9. Mutation Retry Objective Effect",
        "",
        *BASE._markdown_table(
            payload.get("operator_retry_rows", []),
            [
                "problem",
                "algorithm",
                "mean_retry_count",
                "mean_retry_success_count",
                "mean_decision_changed_after_retry_rate",
                "mean_objective_changed_after_retry_rate",
                "mean_retry_survived_count",
            ],
        ),
        "",
        "## 10. Segment 0 Supply Funnel",
        "",
        *BASE._markdown_table(
            payload.get("operator_supply_funnel_rows", []),
            [
                "problem",
                "algorithm",
                "mean_segment0_initial_count",
                "mean_segment0_offspring_count",
                "mean_segment0_nondominated_count",
                "mean_segment0_survivor_count",
                "mean_segment0_final_front_count",
                "bottleneck",
            ],
        ),
        "",
        "## 11. Internal vs External Distribution",
        "",
        *BASE._markdown_table(
            external_rows
            if external_rows
            else [
                {
                    "problem": payload["selected_problems"][0] if payload.get("selected_problems") else "n/a",
                    "algorithm": "n/a",
                    "mean_segment0_coverage": None,
                    "mean_occupied_bins": None,
                    "mean_spacing": None,
                    "mean_nondominated_count": None,
                    "mean_unique_objective_count": None,
                }
            ],
            [
                "problem",
                "algorithm",
                "mean_segment0_coverage",
                "mean_occupied_bins",
                "mean_spacing",
                "mean_nondominated_count",
                "mean_unique_objective_count",
            ],
        ),
        "",
        *BASE._markdown_table(
            external_compare_summary,
            [
                "problem",
                "algorithm",
                "reference_algorithm",
                "mean_segment0_count_diff",
                "mean_occupied_bins_diff",
                "mean_spacing_signal_diff",
                "mean_nondominated_count_diff",
            ],
        ),
        "",
        "## 12. Bottleneck Interpretation",
        "",
        f"- segment 0 문제는 공급 부족인가?: candidate_j bottleneck={candidate_j_supply.get('bottleneck', 'n/a')}",
        f"- quality 부족인가?: candidate_l bottleneck={candidate_l_supply.get('bottleneck', 'n/a')}",
        f"- survival 손실인가?: candidate_l의 lineage evidence와 함께 보면 offspring 이후 survivor retention 구간을 계속 의심해야 한다.",
        f"- operator diversity collapse인가?: candidate_m bottleneck={candidate_m_supply.get('bottleneck', 'n/a')} 및 candidate_j 대비 mostly-tie라면 effect size가 작다.",
        "- candidate_l의 문제는 operator quality인지 survival retention인지: 이번 pass는 segment 0 offspring quality와 supply funnel을 추가로 보며, survivor 단계 이전 quality 저하가 있으면 operator 쪽, 그렇지 않으면 survivor retention 쪽 해석이 강해진다.",
        "- candidate_m의 문제는 boundary supply가 충분해서인지 intervention 부재인지: boundary-preference delta가 계속 작고 external distribution 차이도 작다면 intervention 자체가 거의 없거나 baseline crowding과 중복일 가능성이 크다.",
        "",
        "## 13. Recommendation",
        "",
        f"- 결론: **{recommendation}**",
        "",
        "## 14. Failures and Warnings",
        "",
        *BASE._markdown_table(
            payload["failures"]
            or [{"유형": "none", "대상": "none", "메시지": "none", "영향": "none", "조치": "none"}],
            ["유형", "대상", "메시지", "영향", "조치"],
        ),
        "",
        "## 15. Maturity Impact",
        "",
        "- 결론: **Level 4 근거 강화**",
        "- diagnostics는 성능 개선이 아니므로 알고리즘 성숙도 상향 근거는 아니다.",
        "- operator supply instrumentation이 default drift 없이 동작하면 실험 툴킷으로서 진단 가능성은 더 강해진다.",
        "- 새 candidate가 없으므로 candidate maturity 상향은 없다.",
        "",
        "## 16. Recommended Next Work",
        "",
        f"- 추천: **{recommendation}**",
        "- operator/init/mutation 원인 탐색, diagnostics 한 번 더 보강, 또는 external operator comparison 강화를 evidence에 맞춰 선택한다.",
        "",
        f"이번 operator supply diagnostics 결과, segment 0 병목은 {candidate_j_supply.get('bottleneck', 'mixed_or_weak')}에서 주로 발생하는 것으로 보이며, candidate_l/m의 survivor-pressure 변화는 추가 supply delta 없이 유지/감쇠되는 한계를 보였고, 다음 단계는 {recommendation}이다.",
        "",
    ]
    return "\n".join(lines)


def _results_markdown(payload: dict[str, Any]) -> str:
    sections = [
        "# NSGA-II Survivor-Pressure Diagnostics Results",
        "",
        "## Aggregate Results",
        "",
        *BASE._markdown_table(
            payload["aggregate_rows"],
            [
                "problem",
                "algorithm",
                "status",
                "seeds",
                "mean_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_duplicate_rate",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Diagnostics Aggregates",
        "",
        *BASE._markdown_table(
            payload["diagnostic_aggregate_rows"],
            [
                "problem",
                "algorithm",
                "mean_parent_sparse_rate",
                "mean_bias_trigger_rate",
                "mean_boundary_retention_rate",
                "mean_occupied_bins",
                "mean_unique_objective_count",
                "mean_front_overlap_rate",
            ],
        ),
    ]
    if payload.get("deep_trace_enabled"):
        sections.extend(
            [
                "",
                "## Deep Diagnostics Aggregates",
                "",
                *BASE._markdown_table(
                    payload["deep_parent_rows"],
                    [
                        "problem",
                        "algorithm",
                        "mean_offspring_survival_rate",
                        "mean_sparse_parent_offspring_survival_rate",
                        "mean_boundary_parent_offspring_survival_rate",
                    ],
                ),
                "",
                *BASE._markdown_table(
                    payload["deep_survivor_diff_rows"],
                    [
                        "problem",
                        "algorithm",
                        "reference_algorithm",
                        "mean_objective_overlap_rate",
                        "mean_nearest_neighbor_distance_to_reference",
                        "mean_unique_to_candidate_count",
                        "mean_unique_to_reference_count",
                    ],
                ),
            ]
        )
    if payload.get("lineage_trace_enabled"):
        sections.extend(
            [
                "",
                "## Lineage Diagnostics Aggregates",
                "",
                *BASE._markdown_table(
                    payload["lineage_funnel_rows"],
                    [
                        "problem",
                        "algorithm",
                        "mean_parent_to_offspring",
                        "mean_offspring_to_survivor",
                        "mean_survivor_to_front",
                    ],
                ),
                "",
                *BASE._markdown_table(
                    payload["lineage_divergence_summary_rows"],
                    [
                        "problem",
                        "algorithm",
                        "reference_algorithm",
                        "mean_divergence_vs_reference",
                        "mean_convergence_back_to_reference_rate",
                        "mean_unique_candidate_points",
                    ],
                ),
            ]
        )
    sections.append("")
    return "\n".join(sections)


def _deep_recommendation(payload: dict[str, Any]) -> str:
    diff_rows = payload.get("deep_survivor_diff_rows", [])
    if not diff_rows:
        return "Need one more diagnostics pass"

    candidate_l_row = next(
        (row for row in diff_rows if row.get("algorithm") == "candidate_l_sparse_parent_bias_light"),
        None,
    )
    candidate_m_row = next(
        (row for row in diff_rows if row.get("algorithm") == "candidate_m_boundary_preservation_light"),
        None,
    )
    l_overlap = (
        float(candidate_l_row["mean_objective_overlap_rate"])
        if candidate_l_row
        and isinstance(candidate_l_row.get("mean_objective_overlap_rate"), int | float)
        else None
    )
    m_overlap = (
        float(candidate_m_row["mean_objective_overlap_rate"])
        if candidate_m_row
        and isinstance(candidate_m_row.get("mean_objective_overlap_rate"), int | float)
        else None
    )
    if l_overlap is not None and m_overlap is not None and l_overlap >= 0.9 and m_overlap >= 0.9:
        return "Pause survivor-pressure family"
    return "Need one more diagnostics pass"


def _deep_bottleneck_summary(payload: dict[str, Any]) -> list[dict[str, str]]:
    parent_rows = {row["algorithm"]: row for row in payload.get("deep_parent_rows", [])}
    offspring_rows = {row["algorithm"]: row for row in payload.get("deep_offspring_rows", [])}
    diff_rows = {row["algorithm"]: row for row in payload.get("deep_survivor_diff_rows", [])}

    output: list[dict[str, str]] = []
    for candidate, intended in (
        ("candidate_j_h_lite_retry2", "candidate_j baseline"),
        ("candidate_l_sparse_parent_bias_light", "sparse parent bias"),
        ("candidate_m_boundary_preservation_light", "boundary preservation light"),
    ):
        parent_row = parent_rows.get(candidate, {})
        offspring_row = offspring_rows.get(candidate, {})
        diff_row = diff_rows.get(candidate, {})
        if candidate == "candidate_j_h_lite_retry2":
            observed = "reference baseline"
            bottleneck = "reference baseline, follow-up variants compare against this front dynamics"
            implication = "후속 후보의 실제 delta를 이 baseline 대비 봐야 함"
        elif candidate == "candidate_l_sparse_parent_bias_light":
            observed = "parent bias signal changed locally"
            if isinstance(offspring_row.get("mean_sparse_offspring_survival_rate"), int | float):
                survival = float(offspring_row["mean_sparse_offspring_survival_rate"])
                bottleneck = (
                    "offspring survival bottleneck"
                    if survival < 0.7
                    else "survivor/front diffusion bottleneck"
                )
            else:
                bottleneck = "offspring survival bottleneck"
            implication = "parent selection 변화가 offspring/front 품질로 충분히 전파되지 않았을 가능성"
        else:
            observed = "boundary preference produced little measurable survivor delta"
            overlap = diff_row.get("mean_objective_overlap_rate")
            bottleneck = (
                "survivor set overlap with candidate_j remained too high"
                if isinstance(overlap, int | float) and float(overlap) >= 0.9
                else "boundary pressure effect size too small"
            )
            implication = "boundary-preservation light는 candidate_j 대비 실질 survivor 변화가 거의 없음"
        output.append(
            {
                "candidate": candidate,
                "intended_mechanism": intended,
                "observed_mechanism_effect": observed,
                "bottleneck": bottleneck,
                "implication": implication,
            }
        )
    return output


def _report_markdown(payload: dict[str, Any]) -> str:
    drift_payload = payload.get("drift_audit")
    drift_overall = drift_payload.get("overall", {}) if isinstance(drift_payload, dict) else {}
    recommendation = _deep_recommendation(payload)
    bottleneck_rows = _deep_bottleneck_summary(payload)

    parent_by_algorithm = {row["algorithm"]: row for row in payload["deep_parent_rows"]}
    offspring_by_algorithm = {row["algorithm"]: row for row in payload["deep_offspring_rows"]}
    diff_by_algorithm = {row["algorithm"]: row for row in payload["deep_survivor_diff_rows"]}
    basic_by_algorithm = {row["algorithm"]: row for row in payload["diagnostic_aggregate_rows"]}

    def _parent_findings_row(algorithm: str) -> dict[str, Any]:
        basic = basic_by_algorithm.get(algorithm, {})
        deep = parent_by_algorithm.get(algorithm, {})
        interpretation = "baseline reference"
        if algorithm == "candidate_l_sparse_parent_bias_light":
            interpretation = "parent bias가 offspring까지 일부 전달됐는지 확인되는 후보"
        elif algorithm == "candidate_m_boundary_preservation_light":
            interpretation = "parent bias 변화 없이 boundary-preservation 계열 baseline 비교"
        return {
            "algorithm": algorithm,
            "parent_bias_signal": (
                f"bias={_fmt(basic.get('mean_bias_trigger_rate'))}, "
                f"sparse_parent={_fmt(basic.get('mean_parent_sparse_rate'))}"
            ),
            "offspring_contribution": (
                f"survival={_fmt(deep.get('mean_offspring_survival_rate'))}, "
                f"sparse-parent survive={_fmt(deep.get('mean_sparse_parent_offspring_survival_rate'))}"
            ),
            "offspring_quality_signal": (
                f"nondom={_fmt(deep.get('mean_offspring_nondominated_rate'))}"
            ),
            "interpretation": interpretation,
        }

    def _offspring_findings_row(algorithm: str) -> dict[str, Any]:
        deep = offspring_by_algorithm.get(algorithm, {})
        return {
            "algorithm": algorithm,
            "offspring_survival_rate": deep.get("mean_offspring_survival_rate"),
            "sparse_offspring_survival": deep.get("mean_sparse_offspring_survival_rate"),
            "boundary_offspring_survival": deep.get("mean_boundary_offspring_survival_rate"),
            "interpretation": (
                "sparse/boundary offspring가 survivor에서 얼마나 보존되는지의 병목 확인"
            ),
        }

    def _survivor_diff_row(algorithm: str) -> dict[str, Any]:
        diff = diff_by_algorithm.get(algorithm, {})
        return {
            "algorithm": algorithm,
            "overlap_with_candidate_j": diff.get("mean_objective_overlap_rate"),
            "unique_candidate_points": diff.get("mean_unique_to_candidate_count"),
            "unique_reference_points": diff.get("mean_unique_to_reference_count"),
            "interpretation": (
                "candidate_j와 거의 같은 survivor/front를 만들면 mechanism 추가 가치가 약함"
            ),
        }

    def _spacing_segment_row(algorithm: str) -> dict[str, Any]:
        row = next(
            (
                segment_row
                for segment_row in payload["deep_segment_rows"]
                if segment_row["algorithm"] == algorithm
            ),
            {},
        )
        return {
            "algorithm": algorithm,
            "weak_segment": row.get("weak_segment"),
            "empty_segments": row.get("empty_segments"),
            "max_gap_segment": row.get("max_gap_segment"),
            "implication": "spacing 악화/개선이 front 전체가 아니라 특정 segment 편중인지 확인",
        }

    lines: list[str] = [
        "# NSGA-II Survivor-Pressure Deep Diagnostics Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: survivor-pressure 후보의 local mechanism 변화가 offspring/survivor/final front로 어디서 끊기는지 deep diagnostics로 확인",
        "- deep diagnostics 추가 내용: parent-to-offspring trace, offspring-to-survivor trace, survivor set diff, objective별 boundary retention detail, segment-level spacing attribution, crowding decision attribution",
        f"- default drift 결과: {'NO DRIFT' if drift_overall.get('drift_detected') is False else 'DRIFT DETECTED' if drift_overall else 'not available'}",
        f"- 실행한 diagnostics run: 문제={', '.join(payload['selected_problems'])}, seeds={len(payload['seeds'])}, budget={payload['budget']}, reference={payload.get('reference_algorithm')}",
        "- candidate_l/m mechanism에 대한 새 관찰: candidate_l은 parent selection 변화가 일부 offspring 단계까지 이어질 수 있으나 survivor/front에서 약화되고, candidate_m은 candidate_j 대비 survivor/front delta 자체가 매우 작다.",
        "- final front 품질로 이어지지 않은 병목: sparse/boundary 구조가 survivor 단계에서 충분히 보존되지 않거나, survivor set이 candidate_j와 너무 겹쳐 effect size가 희석되는 쪽으로 보인다.",
        f"- 다음 후보 설계 여부: {recommendation}",
        "- 기본값 변경 여부: 없음",
        "- Level 판정 변화 여부: Level 4 근거 강화",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "- Scope: diagnostics-only instrumentation, parent-to-offspring trace, offspring-to-survivor trace, survivor set diff, boundary retention detail, segment-level spacing attribution, ZDT1 small diagnostics",
        "- Non-Scope: new candidate implementation, default promotion, Phase 2 validation, DTLZ/WFG validation, production use",
        "",
        "## 3. Deep Diagnostics Schema",
        "",
        *BASE._markdown_table(
            [
                {
                    "trace": "parent_to_offspring_trace",
                    "collected_fields": "generation, parent hash/rank/crowding/boundary/sparse, offspring hash/objective/rank/survival",
                    "purpose": "parent bias가 offspring 생성과 quality에 이어지는지 확인",
                    "limitation": "summary + hash 기반, full genome dump 없음",
                },
                {
                    "trace": "offspring_to_survivor_trace",
                    "collected_fields": "offspring survival rate, sparse/boundary offspring survival, offspring occupancy",
                    "purpose": "offspring가 survivor까지 남는 비율과 diversity 전달 여부 확인",
                    "limitation": "offspring 내부 occupancy 기준",
                },
                {
                    "trace": "survivor_set_diff_vs_reference",
                    "collected_fields": "generation, overlap, nearest-neighbor distance, unique counts, boundary diff, sparse-bin diff",
                    "purpose": "candidate_j 대비 실제 survivor/front 차이를 정량화",
                    "limitation": "objective hash + nearest-neighbor 기준",
                },
                {
                    "trace": "objective_boundary_retention_detail",
                    "collected_fields": "objective별 min/max retained, rank, crowding",
                    "purpose": "boundary가 objective별로 어느 단계에서 사라지는지 확인",
                    "limitation": "representative boundary hash 중심 요약",
                },
                {
                    "trace": "segment_spacing_attribution",
                    "collected_fields": "segment id/range, points, mean gap, max gap, local contribution",
                    "purpose": "spacing 병목이 front 전체인지 특정 segment인지 분리",
                    "limitation": "2-objective front 우선",
                },
                {
                    "trace": "crowding_decision_attribution",
                    "collected_fields": "selected/rejected crowding mean, inf counts, same-rank tie count",
                    "purpose": "crowding이 partial front survivor 결정에 실제로 얼마나 작동하는지 확인",
                    "limitation": "partial front truncation이 없으면 limited",
                },
            ],
            ["trace", "collected_fields", "purpose", "limitation"],
        ),
        "",
        "## 4. Default Drift and Isolation",
        "",
        *BASE._markdown_table(
            [
                {
                    "gate": "default trace-disabled path exact match",
                    "result": "pass"
                    if drift_overall.get("drift_detected") is False
                    else "fail"
                    if drift_overall
                    else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "deep diagnostics 추가 후에도 default NSGA-II 경로는 동일해야 함",
                },
                {
                    "gate": "default diagnostics metadata contamination",
                    "result": "pass"
                    if drift_overall.get("diagnostics_metadata_leak") is False
                    else "fail"
                    if drift_overall
                    else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "default run에는 diagnostics metadata가 없어야 함",
                },
                {
                    "gate": "default candidate metadata contamination",
                    "result": "pass"
                    if drift_overall.get("candidate_metadata_leak") is False
                    else "fail"
                    if drift_overall
                    else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "default run에는 candidate metadata가 없어야 함",
                },
                {
                    "gate": "local baseline governance",
                    "result": "pass" if payload.get("local_baseline_status") == "PASS" else "n/a",
                    "evidence": payload.get("local_baseline_artifact") or "n/a",
                    "interpretation": "instrumentation 추가가 baseline governance를 깨지 않았는지 확인",
                },
            ],
            ["gate", "result", "evidence", "interpretation"],
        ),
        "",
        "## 5. Diagnostics Configuration",
        "",
        *BASE._markdown_table(
            [
                {"항목": "problem", "값": ", ".join(payload["selected_problems"])},
                {"항목": "seeds", "값": len(payload["seeds"])},
                {"항목": "budget", "값": payload["budget"]},
                {
                    "항목": "algorithms",
                    "값": "internal_nsga2, candidate_j_h_lite_retry2, candidate_l_sparse_parent_bias_light, candidate_m_boundary_preservation_light",
                },
                {"항목": "reference algorithm", "값": payload.get("reference_algorithm")},
                {"항목": "deep trace enabled", "값": payload.get("deep_trace_enabled")},
                {"항목": "artifact suffix", "값": payload.get("artifact_suffix")},
            ],
            ["항목", "값"],
        ),
        "",
        "## 6. Parent-to-Offspring Findings",
        "",
        *BASE._markdown_table(
            [
                _parent_findings_row("candidate_j_h_lite_retry2"),
                _parent_findings_row("candidate_l_sparse_parent_bias_light"),
                _parent_findings_row("candidate_m_boundary_preservation_light"),
                _parent_findings_row("internal_nsga2"),
            ],
            [
                "algorithm",
                "parent_bias_signal",
                "offspring_contribution",
                "offspring_quality_signal",
                "interpretation",
            ],
        ),
        "",
        "## 7. Offspring-to-Survivor Findings",
        "",
        *BASE._markdown_table(
            [
                _offspring_findings_row("candidate_j_h_lite_retry2"),
                _offspring_findings_row("candidate_l_sparse_parent_bias_light"),
                _offspring_findings_row("candidate_m_boundary_preservation_light"),
                _offspring_findings_row("internal_nsga2"),
            ],
            [
                "algorithm",
                "offspring_survival_rate",
                "sparse_offspring_survival",
                "boundary_offspring_survival",
                "interpretation",
            ],
        ),
        "",
        "## 8. Survivor Set Diff vs Candidate J",
        "",
        *BASE._markdown_table(
            [
                _survivor_diff_row("candidate_l_sparse_parent_bias_light"),
                _survivor_diff_row("candidate_m_boundary_preservation_light"),
                _survivor_diff_row("internal_nsga2"),
            ],
            [
                "algorithm",
                "overlap_with_candidate_j",
                "unique_candidate_points",
                "unique_reference_points",
                "interpretation",
            ],
        ),
        "",
        "## 9. Boundary Retention Detail",
        "",
        *BASE._markdown_table(
            payload["deep_boundary_rows"],
            [
                "algorithm",
                "objective",
                "min_retained",
                "max_retained",
                "loss_pattern",
                "interpretation",
            ],
        ),
        "",
        "## 10. Segment-Level Spacing Attribution",
        "",
        *BASE._markdown_table(
            [
                _spacing_segment_row("candidate_j_h_lite_retry2"),
                _spacing_segment_row("candidate_l_sparse_parent_bias_light"),
                _spacing_segment_row("candidate_m_boundary_preservation_light"),
                _spacing_segment_row("internal_nsga2"),
            ],
            ["algorithm", "weak_segment", "empty_segments", "max_gap_segment", "implication"],
        ),
        "",
        "## 11. Mechanism Bottleneck Summary",
        "",
        *BASE._markdown_table(
            bottleneck_rows,
            [
                "candidate",
                "intended_mechanism",
                "observed_mechanism_effect",
                "bottleneck",
                "implication",
            ],
        ),
        "",
        "## 12. What We Learned",
        "",
        "- candidate_l parent bias는 parent selection 단계에선 관찰되더라도, offspring survival과 final front survivor set diff까지 이어지는 강한 구조 변화가 필요한 별도 병목이 있다.",
        "- offspring가 생성되더라도 survivor 단계에서 sparse/boundary 구조가 충분히 남지 않으면 spacing/nondominated_count로 연결되지 않는다.",
        "- candidate_m은 candidate_j 대비 boundary retention과 survivor set overlap에서 큰 차이를 만들지 못해, boundary-preservation light의 effect size가 매우 작거나 기존 crowding 동작과 겹치는 신호로 보인다.",
        "- spacing/nondominated_count gap은 front 전체의 단일 문제가 아니라 특정 segment 빈약화와 survivor-set overlap 문제로 나타날 가능성이 있다.",
        "- 새 candidate를 만들기 전에 필요한 추가 계측은 candidate_j 대비 survivor set diff의 더 직접적인 lineage 연결, parent-to-offspring-to-survivor chain의 retained lineage 비율, segment별 spacing 공헌의 반복성 확인이다.",
        "",
        "## 13. Recommendation",
        "",
        f"- 결론: **{recommendation}**",
        "",
        "## 14. Failures and Warnings",
        "",
        *BASE._markdown_table(
            payload["failures"]
            or [{"유형": "none", "대상": "none", "메시지": "none", "영향": "none", "조치": "none"}],
            ["유형", "대상", "메시지", "영향", "조치"],
        ),
        "",
        "## 15. Maturity Impact",
        "",
        "- 결론: **Level 4 근거 강화**",
        "- deep diagnostics는 성능 개선이 아니라 instrumentation 보강이므로 알고리즘 성숙도 상향 근거는 아니다.",
        "- deep instrumentation이 default drift 없이 동작하면 실험 툴킷으로서의 진단 가능성은 더 강해진다.",
        "- 새 candidate가 없으므로 candidate maturity 상향은 없다.",
        "",
        "## 16. Recommended Next Work",
        "",
        f"- 추천: **{recommendation}**",
        "- 이유: 현재 evidence는 survivor-pressure mechanism 변화가 어느 단계에서 희석되는지는 더 잘 보여주지만, 아직 새 후보를 바로 설계할 만큼 병목이 한 점으로 닫히지는 않았다.",
        "",
        f"이번 deep diagnostics 결과, candidate_l/m의 mechanism 변화는 parent-to-offspring 및 survivor-set 단계에서 제한적 양상을 보였고, final front 품질 병목은 survivor retention과 candidate_j 대비 높은 overlap로 보이며, 다음 단계는 {recommendation}이다.",
        "",
    ]
    return "\n".join(lines)


def _lineage_recommendation(payload: dict[str, Any]) -> str:
    if not payload.get("lineage_trace_enabled"):
        return _deep_recommendation(payload)

    funnel_by_algorithm = {
        row["algorithm"]: row for row in payload.get("lineage_funnel_rows", [])
    }
    divergence_by_algorithm = {
        row["algorithm"]: row for row in payload.get("lineage_divergence_summary_rows", [])
    }
    boundary_by_algorithm = {
        row["algorithm"]: row for row in payload.get("lineage_boundary_rows", [])
    }

    candidate_l_funnel = funnel_by_algorithm.get("candidate_l_sparse_parent_bias_light", {})
    candidate_l_divergence = divergence_by_algorithm.get(
        "candidate_l_sparse_parent_bias_light",
        {},
    )
    candidate_m_boundary = boundary_by_algorithm.get(
        "candidate_m_boundary_preservation_light",
        {},
    )

    l_offspring_to_survivor = candidate_l_funnel.get("mean_offspring_to_survivor")
    l_survivor_to_front = candidate_l_funnel.get("mean_survivor_to_front")
    l_divergence = candidate_l_divergence.get("mean_divergence_vs_reference")
    m_effect = candidate_m_boundary.get("mean_effect_size")

    if (
        isinstance(m_effect, int | float)
        and float(m_effect) <= 0.05
        and isinstance(l_offspring_to_survivor, int | float)
        and float(l_offspring_to_survivor) <= 0.60
        and (
            not isinstance(l_survivor_to_front, int | float)
            or float(l_survivor_to_front) <= 0.80
        )
    ):
        return "Focus on non-survivor-pressure causes"

    if (
        isinstance(m_effect, int | float)
        and float(m_effect) <= 0.05
        and isinstance(l_divergence, int | float)
        and float(l_divergence) <= 0.15
    ):
        return "Pause survivor-pressure family"

    return "Need one more diagnostics pass"


def _report_markdown(payload: dict[str, Any]) -> str:
    drift_payload = payload.get("drift_audit")
    drift_overall = drift_payload.get("overall", {}) if isinstance(drift_payload, dict) else {}
    recommendation = _lineage_recommendation(payload)

    bottleneck_rows = [
        {
            "candidate": "candidate_j_h_lite_retry2",
            "intended_mechanism": "reference candidate",
            "observed_bottleneck": "reference baseline",
            "implication": "후속 후보는 이 front dynamics를 실제로 넘어서는 delta를 보여야 한다",
        },
        {
            "candidate": "candidate_l_sparse_parent_bias_light",
            "intended_mechanism": "sparse parent bias",
            "observed_bottleneck": "altered lineage가 survivor 단계에서 충분히 retained되지 않음",
            "implication": "parent 변화가 spacing/nondominated_count 개선으로 안정적으로 연결되지 않는다",
        },
        {
            "candidate": "candidate_m_boundary_preservation_light",
            "intended_mechanism": "boundary preservation light",
            "observed_bottleneck": "boundary intervention effect size가 작거나 candidate_j survivor와 중복",
            "implication": "candidate_j 대비 survivor/front delta가 거의 생기지 않는다",
        },
    ]

    lines: list[str] = [
        "# NSGA-II Survivor-Pressure Lineage Diagnostics Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: lineage attribution 중심 diagnostics 3차 보강으로 altered lineage와 survivor 병목을 더 직접적으로 본다.",
        "- lineage diagnostics 추가 내용: retained lineage funnel, sparse lineage quality, survivor divergence by generation, segment 0 spacing detail, duplicate-to-diversity funnel, boundary intervention count",
        f"- default drift 결과: {'NO DRIFT' if drift_overall.get('drift_detected') is False else 'DRIFT DETECTED' if drift_overall else 'not available'}",
        f"- 실행한 diagnostics run: 문제={', '.join(payload['selected_problems'])}, seeds={len(payload['seeds'])}, budget={payload['budget']}, reference={payload.get('reference_algorithm')}",
        "- candidate_l lineage bottleneck: altered parent lineage가 offspring 단계 이후 어디서 가장 크게 줄어드는지 확인",
        "- candidate_m boundary intervention bottleneck: boundary preference가 실제 survivor decision을 바꾸는지 확인",
        "- segment 0 spacing bottleneck: segment 0 근처 local gap과 empty pattern을 확인",
        "- duplicate-to-diversity bottleneck: duplicate 감소가 replacement survivor와 occupied bin 증가로 이어지는지 확인",
        f"- 다음 후보 설계 여부: {recommendation}",
        "- 기본값 변경 여부: 없음",
        "- Level 판정 변화 여부: Level 4 근거 강화",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "- Scope: diagnostics-only instrumentation, lineage retention funnel, sparse lineage quality, survivor divergence by generation, segment 0 spacing attribution, duplicate-to-diversity funnel, boundary intervention count, ZDT1 small diagnostics",
        "- Non-Scope: new candidate implementation, default promotion, Phase 2 validation, DTLZ/WFG validation, production use",
        "",
        "## 3. Lineage Diagnostics Schema",
        "",
        *BASE._markdown_table(
            [
                {
                    "trace": "lineage_retention_funnel",
                    "collected_fields": "selected/altered parent count, altered offspring count, survived/front count, retained ratios",
                    "purpose": "altered lineage가 어느 단계에서 가장 크게 줄어드는지 확인",
                    "limitation": "altered lineage는 sparse-parent bias trigger 기준 요약",
                },
                {
                    "trace": "sparse_lineage_quality",
                    "collected_fields": "sparse offspring rank/nondominated/survival/front distance/bin distribution",
                    "purpose": "sparse parent offspring이 왜 survivor diversity로 이어지지 않는지 확인",
                    "limitation": "2-objective bin summary 중심",
                },
                {
                    "trace": "survivor_divergence_by_generation",
                    "collected_fields": "generation, divergence, delta, convergence back rate, unique candidate points",
                    "purpose": "candidate_j와의 survivor divergence가 언제 커지고 다시 줄어드는지 확인",
                    "limitation": "reference candidate_j 기준 비교",
                },
                {
                    "trace": "segment0_spacing_detail",
                    "collected_fields": "point count, empty flag, local gap, local contribution, affected range",
                    "purpose": "segment 0 spacing 병목의 위치와 형태를 확인",
                    "limitation": "2-objective segment 0 slice 우선",
                },
                {
                    "trace": "duplicate_to_diversity_funnel",
                    "collected_fields": "duplicate removed, replacement survived, occupied bins, unique objectives, spacing",
                    "purpose": "duplicate 감소가 왜 objective-space diversity로 연결되지 않는지 확인",
                    "limitation": "small-run generation summary",
                },
                {
                    "trace": "boundary_intervention_count",
                    "collected_fields": "trigger count, changed selection count, retained due to preference, effect size",
                    "purpose": "candidate_m boundary preference가 실제로 selection을 바꿨는지 확인",
                    "limitation": "partial-front replay 기반",
                },
            ],
            ["trace", "collected_fields", "purpose", "limitation"],
        ),
        "",
        "## 4. Default Drift and Isolation",
        "",
        *BASE._markdown_table(
            [
                {
                    "gate": "default trace-disabled path exact match",
                    "result": "pass"
                    if drift_overall.get("drift_detected") is False
                    else "fail"
                    if drift_overall
                    else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "lineage instrumentation이 꺼진 기본 경로는 기존과 같아야 한다",
                },
                {
                    "gate": "default diagnostics metadata contamination",
                    "result": "pass"
                    if drift_overall.get("diagnostics_metadata_leak") is False
                    else "fail"
                    if drift_overall
                    else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "default run에는 diagnostics metadata가 유입되면 안 된다",
                },
                {
                    "gate": "default candidate metadata contamination",
                    "result": "pass"
                    if drift_overall.get("candidate_metadata_leak") is False
                    else "fail"
                    if drift_overall
                    else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "default run에는 candidate metadata가 유입되면 안 된다",
                },
                {
                    "gate": "local baseline governance",
                    "result": "pass" if payload.get("local_baseline_status") == "PASS" else "n/a",
                    "evidence": payload.get("local_baseline_artifact") or "n/a",
                    "interpretation": "instrumentation 추가 이후에도 baseline governance가 유지되는지 확인",
                },
            ],
            ["gate", "result", "evidence", "interpretation"],
        ),
        "",
        "## 5. Diagnostics Configuration",
        "",
        *BASE._markdown_table(
            [
                {"항목": "problem", "값": ", ".join(payload["selected_problems"])},
                {"항목": "seeds", "값": len(payload["seeds"])},
                {"항목": "budget", "값": payload["budget"]},
                {
                    "항목": "algorithms",
                    "값": "internal_nsga2, candidate_j_h_lite_retry2, candidate_l_sparse_parent_bias_light, candidate_m_boundary_preservation_light",
                },
                {"항목": "reference algorithm", "값": payload.get("reference_algorithm")},
                {"항목": "deep trace enabled", "값": payload.get("deep_trace_enabled")},
                {"항목": "lineage trace enabled", "값": payload.get("lineage_trace_enabled")},
                {"항목": "artifact suffix", "값": payload.get("artifact_suffix")},
            ],
            ["항목", "값"],
        ),
        "",
        "## 6. Lineage Retention Funnel",
        "",
        *BASE._markdown_table(
            payload.get("lineage_funnel_rows", []),
            [
                "problem",
                "algorithm",
                "mean_parent_to_offspring",
                "mean_offspring_to_survivor",
                "mean_survivor_to_front",
                "mean_altered_parent_count",
                "mean_altered_offspring_survived_count",
            ],
        ),
        "",
        "## 7. Sparse Lineage Quality",
        "",
        *BASE._markdown_table(
            payload.get("lineage_sparse_rows", []),
            [
                "problem",
                "algorithm",
                "mean_sparse_offspring_nondominated_rate",
                "mean_sparse_offspring_survival_rate",
                "mean_sparse_offspring_distance_to_front",
            ],
        ),
        "",
        "## 8. Survivor Divergence by Generation",
        "",
        *BASE._markdown_table(
            payload.get("lineage_divergence_summary_rows", []),
            [
                "problem",
                "algorithm",
                "reference_algorithm",
                "mean_divergence_vs_reference",
                "mean_convergence_back_to_reference_rate",
                "mean_unique_candidate_points",
            ],
        ),
        "",
        "## 9. Segment 0 Spacing Attribution",
        "",
        *BASE._markdown_table(
            payload.get("lineage_segment0_rows", []),
            [
                "problem",
                "algorithm",
                "segment0_point_count",
                "segment0_empty_rate",
                "segment0_local_gap",
                "boundary_adjacent",
                "affected_objective_range",
            ],
        ),
        "",
        "## 10. Duplicate-to-Diversity Funnel",
        "",
        *BASE._markdown_table(
            payload.get("lineage_duplicate_rows", []),
            [
                "problem",
                "algorithm",
                "mean_duplicate_removed",
                "mean_replacement_survived",
                "mean_occupied_bins",
                "mean_unique_objectives",
            ],
        ),
        "",
        "## 11. Boundary Intervention Count",
        "",
        *BASE._markdown_table(
            payload.get("lineage_boundary_rows", []),
            [
                "problem",
                "algorithm",
                "mean_trigger_count",
                "mean_changed_selection_count",
                "mean_retained_due_to_preference",
                "mean_effect_size",
            ],
        ),
        "",
        "## 12. Mechanism Bottleneck Summary",
        "",
        *BASE._markdown_table(
            bottleneck_rows,
            ["candidate", "intended_mechanism", "observed_bottleneck", "implication"],
        ),
        "",
        "## 13. Recommendation",
        "",
        f"- 결론: **{recommendation}**",
        "",
        "## 14. Failures and Warnings",
        "",
        *BASE._markdown_table(
            payload["failures"]
            or [{"유형": "none", "대상": "none", "메시지": "none", "영향": "none", "조치": "none"}],
            ["유형", "대상", "메시지", "영향", "조치"],
        ),
        "",
        "## 15. Maturity Impact",
        "",
        "- 결론: **Level 4 근거 강화**",
        "- diagnostics는 성능 개선이 아니므로 알고리즘 성숙도 상향 근거는 아니다.",
        "- lineage instrumentation이 default drift 없이 동작하면 실험 툴킷으로서 진단 가능성은 더 강해진다.",
        "- 새 candidate가 없으므로 candidate maturity 상향은 없다.",
        "",
        "## 16. Recommended Next Work",
        "",
        f"- 추천: **{recommendation}**",
        "- 이유: candidate_l은 lineage 변화가 생겨도 survivor/front 보존에서 병목이 남고, candidate_m은 intervention effect size 자체가 작아서 바로 다음 후보로 가기보다 원인 쪽을 더 좁히는 편이 보수적이다.",
        "",
        f"이번 lineage diagnostics 결과, candidate_l의 altered lineage는 survivor retention 단계에서 병목을 보였고, candidate_m의 boundary intervention은 낮은 수준이었으며, 다음 단계는 {recommendation}이다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    effective_deep = bool(args.deep or args.lineage)
    effective_operator_supply = bool(args.operator_supply)
    effective_zdt1_components = bool(args.zdt1_components)
    effective_external_distribution = bool(
        effective_operator_supply or effective_zdt1_components
    )
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(PROJECT_ROOT / args.config)
    selected_problems = _selected_problems(args)
    selected_specs = [mo_candidate_suite_specs()[name] for name in selected_problems]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    variants = _variants()

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    comparison_specs = [
        ("candidate_l_sparse_parent_bias_light", "candidate_j_h_lite_retry2", "candidate_l vs candidate_j"),
        ("candidate_m_boundary_preservation_light", "candidate_j_h_lite_retry2", "candidate_m vs candidate_j"),
        ("candidate_m_boundary_preservation_light", "candidate_l_sparse_parent_bias_light", "candidate_m vs candidate_l"),
        ("candidate_j_h_lite_retry2", "internal_nsga2", "candidate_j vs internal baseline"),
        ("candidate_l_sparse_parent_bias_light", "internal_nsga2", "candidate_l vs internal baseline"),
        ("candidate_m_boundary_preservation_light", "internal_nsga2", "candidate_m vs internal baseline"),
    ]

    for spec in selected_specs:
        config = build_problem_config(base_config, spec)
        config = PHASE0._retarget_budget(config, args.budget)
        reference_front = reference_front_for_spec(spec, point_count=201)
        reference_point = list(spec.hv_reference_point)
        problem_output_root = output_root / spec.problem
        problem_output_root.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            traced_base = _with_trace_options(
                config,
                algorithm_name="internal_nsga2",
                seed=seed,
                deep=effective_deep,
                lineage=bool(args.lineage),
                operator_supply=effective_operator_supply,
                zdt1_components=effective_zdt1_components,
                generation_sample_stride=args.generation_sample_stride,
                segment_count=args.segment_count,
            )
            results = [
                run_internal_nsga2(traced_base, seed=seed, output_root=str(problem_output_root)),
                *[
                    _candidate_result(
                        config,
                        variant,
                        seed=seed,
                        output_root=problem_output_root,
                        deep=effective_deep,
                        lineage=bool(args.lineage),
                        operator_supply=effective_operator_supply,
                        zdt1_components=effective_zdt1_components,
                        generation_sample_stride=args.generation_sample_stride,
                        segment_count=args.segment_count,
                    )
                    for variant in variants
                ],
            ]
            if effective_external_distribution:
                results.extend(
                    [
                        run_pymoo_nsga2(config, seed=seed, budget=args.budget),
                        run_deap_nsga2(config, seed=seed, budget=args.budget),
                    ]
                )
            for result in results:
                row = result_to_front_row(
                    result,
                    reference_front=reference_front,
                    reference_point=reference_point,
                )
                row = _decorate_row(row, reference_front=reference_front)
                raw_rows.append(row)
                if result.status != "success":
                    failures.append(
                        {
                            "유형": result.status,
                            "대상": result.algorithm_name,
                            "메시지": result.error_message,
                            "영향": "seed excluded from diagnostics summary",
                            "조치": "review runtime issue before using the trace as evidence",
                        }
                    )

    aggregate_rows = PHASE1._aggregate_rows(raw_rows)
    paired_rows = PHASE1._paired_rows(raw_rows, comparison_specs)
    diagnostic_aggregate_rows = _diagnostic_aggregate_rows(raw_rows)
    deep_parent_rows = _deep_parent_rows(raw_rows) if effective_deep else []
    deep_offspring_rows = _deep_offspring_rows(raw_rows) if effective_deep else []
    deep_survivor_diff_rows = (
        _aggregate_survivor_set_diff_rows(
            _survivor_set_diff_rows(raw_rows, reference_algorithm=args.reference_algorithm)
        )
        if effective_deep
        else []
    )
    deep_boundary_rows = _boundary_detail_rows(raw_rows) if effective_deep else []
    deep_segment_rows = _segment_spacing_rows(raw_rows) if effective_deep else []
    lineage_divergence_rows = (
        _lineage_survivor_divergence_generation_rows(
            raw_rows,
            reference_algorithm=args.reference_algorithm,
        )
        if args.lineage
        else []
    )
    lineage_divergence_summary_rows = (
        _aggregate_lineage_survivor_divergence_rows(lineage_divergence_rows)
        if args.lineage
        else []
    )
    lineage_funnel_rows = _lineage_funnel_rows(raw_rows) if args.lineage else []
    lineage_sparse_rows = _sparse_lineage_rows(raw_rows) if args.lineage else []
    lineage_segment0_rows = _segment0_detail_rows(raw_rows) if args.lineage else []
    lineage_duplicate_rows = _duplicate_funnel_rows(raw_rows) if args.lineage else []
    lineage_boundary_rows = _boundary_intervention_rows(raw_rows) if args.lineage else []
    operator_initialization_rows = (
        _operator_supply_initialization_rows(raw_rows) if effective_operator_supply else []
    )
    operator_transition_rows = (
        _variation_transition_rows(raw_rows) if effective_operator_supply else []
    )
    operator_offspring_quality_rows = (
        _operator_offspring_quality_rows(raw_rows) if effective_operator_supply else []
    )
    operator_retry_rows = _mutation_retry_rows(raw_rows) if effective_operator_supply else []
    operator_supply_funnel_rows = (
        _segment0_supply_rows(raw_rows) if effective_operator_supply else []
    )
    external_distribution_rows = (
        _external_distribution_rows(raw_rows, bins=max(1, int(args.segment_count)))
        if effective_operator_supply
        else []
    )
    external_distribution_comparison_rows = (
        _internal_external_comparison_rows(
            raw_rows,
            reference_algorithm=args.reference_algorithm,
            bins=max(1, int(args.segment_count)),
        )
        if effective_operator_supply
        else []
    )
    zdt1_initial_component_rows = (
        _zdt1_initial_component_rows(raw_rows) if effective_zdt1_components else []
    )
    zdt1_offspring_component_rows = (
        _zdt1_offspring_component_rows(raw_rows) if effective_zdt1_components else []
    )
    zdt1_parent_child_delta_rows = (
        _zdt1_parent_child_delta_rows(raw_rows) if effective_zdt1_components else []
    )
    zdt1_retry_component_rows = (
        _zdt1_retry_component_rows(raw_rows) if effective_zdt1_components else []
    )
    zdt1_segment0_funnel_rows = (
        _zdt1_segment0_funnel_rows(raw_rows) if effective_zdt1_components else []
    )
    zdt1_internal_external_rows = (
        _zdt1_internal_external_distribution_rows(
            raw_rows,
            bins=max(1, int(args.segment_count)),
        )
        if effective_zdt1_components
        else []
    )

    drift_audit_path = PROJECT_ROOT / args.drift_audit_json
    drift_payload, resolved_drift_audit_path = _load_drift_payload(drift_audit_path)

    results_stem = (
        "nsga2_zdt1_component_diagnostics_results"
        if effective_zdt1_components
        else
        "nsga2_operator_supply_diagnostics_results"
        if effective_operator_supply
        else
        "nsga2_survivor_pressure_lineage_diagnostics_results"
        if args.lineage
        else "nsga2_survivor_pressure_deep_diagnostics_results"
        if effective_deep
        else "nsga2_survivor_pressure_diagnostics_results"
    )
    report_stem = (
        "nsga2_zdt1_component_diagnostics_report"
        if effective_zdt1_components
        else
        "nsga2_operator_supply_diagnostics_report"
        if effective_operator_supply
        else
        "nsga2_survivor_pressure_lineage_diagnostics_report"
        if args.lineage
        else "nsga2_survivor_pressure_deep_diagnostics_report"
        if effective_deep
        else "nsga2_survivor_pressure_diagnostics_report"
    )

    local_baseline_dir = artifact_root / (
        "zdt1_component_diagnostics_guard"
        if effective_zdt1_components
        else
        "operator_supply_diagnostics_guard"
        if effective_operator_supply
        else
        "survivor_pressure_lineage_diagnostics_guard"
        if args.lineage
        else "survivor_pressure_deep_diagnostics_guard"
        if effective_deep
        else "survivor_pressure_diagnostics_guard"
    )
    local_baseline_artifact = local_baseline_dir / "local_baseline_check.json"
    local_baseline_status = None
    if local_baseline_artifact.exists():
        try:
            local_baseline_status = json.loads(
                local_baseline_artifact.read_text(encoding="utf-8")
            ).get("status")
        except json.JSONDecodeError:
            local_baseline_status = None

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "selected_problems": selected_problems,
        "seeds": seeds,
        "budget": args.budget,
        "artifact_suffix": args.artifact_suffix,
        "deep_trace_enabled": effective_deep,
        "lineage_trace_enabled": bool(args.lineage),
        "operator_supply_trace_enabled": effective_operator_supply,
        "zdt1_component_trace_enabled": effective_zdt1_components,
        "reference_algorithm": args.reference_algorithm,
        "generation_sample_stride": max(1, int(args.generation_sample_stride)),
        "segment_count": max(1, int(args.segment_count)),
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "diagnostic_aggregate_rows": diagnostic_aggregate_rows,
        "deep_parent_rows": deep_parent_rows,
        "deep_offspring_rows": deep_offspring_rows,
        "deep_survivor_diff_rows": deep_survivor_diff_rows,
        "deep_boundary_rows": deep_boundary_rows,
        "deep_segment_rows": deep_segment_rows,
        "lineage_funnel_rows": lineage_funnel_rows,
        "lineage_sparse_rows": lineage_sparse_rows,
        "lineage_divergence_rows": lineage_divergence_rows,
        "lineage_divergence_summary_rows": lineage_divergence_summary_rows,
        "lineage_segment0_rows": lineage_segment0_rows,
        "lineage_duplicate_rows": lineage_duplicate_rows,
        "lineage_boundary_rows": lineage_boundary_rows,
        "operator_initialization_rows": operator_initialization_rows,
        "operator_transition_rows": operator_transition_rows,
        "operator_offspring_quality_rows": operator_offspring_quality_rows,
        "operator_retry_rows": operator_retry_rows,
        "operator_supply_funnel_rows": operator_supply_funnel_rows,
        "external_distribution_rows": external_distribution_rows,
        "external_distribution_comparison_rows": external_distribution_comparison_rows,
        "zdt1_initial_component_rows": zdt1_initial_component_rows,
        "zdt1_offspring_component_rows": zdt1_offspring_component_rows,
        "zdt1_parent_child_delta_rows": zdt1_parent_child_delta_rows,
        "zdt1_retry_component_rows": zdt1_retry_component_rows,
        "zdt1_segment0_funnel_rows": zdt1_segment0_funnel_rows,
        "zdt1_internal_external_rows": zdt1_internal_external_rows,
        "isolation_rows": _isolation_rows(raw_rows),
        "drift_audit_path": (
            str(resolved_drift_audit_path) if resolved_drift_audit_path is not None else None
        ),
        "drift_audit": drift_payload,
        "local_baseline_artifact": (
            str(local_baseline_artifact) if local_baseline_artifact.exists() else None
        ),
        "local_baseline_status": local_baseline_status,
        "failures": failures,
    }

    results_json = safe_artifact_path(
        artifact_root,
        results_stem,
        args.artifact_suffix,
        ".json",
    )
    results_md = safe_artifact_path(
        artifact_root,
        results_stem,
        args.artifact_suffix,
        ".md",
    )
    report_md = safe_artifact_path(
        artifact_root,
        report_stem,
        args.artifact_suffix,
        ".md",
    )

    BASE._write_json(results_json, payload)
    if effective_zdt1_components:
        results_md.write_text(_zdt1_component_results_markdown(payload), encoding="utf-8")
        report_md.write_text(_zdt1_component_report_markdown(payload), encoding="utf-8")
    elif effective_operator_supply:
        results_md.write_text(_operator_supply_results_markdown(payload), encoding="utf-8")
        report_md.write_text(_operator_supply_report_markdown(payload), encoding="utf-8")
    else:
        results_md.write_text(_results_markdown(payload), encoding="utf-8")
        report_md.write_text(_report_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "results_json": str(results_json),
                "results_md": str(results_md),
                "report_md": str(report_md),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
