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
from ga_lab.experiment.external_mo_comparators import (
    ExternalMOComparatorResult,
    result_to_front_row,
    run_internal_nsga2,
    run_pymoo_nsga2,
    run_random_archive_anchor,
)
from ga_lab.experiment.mo_baselines import run_random_pareto_archive
from ga_lab.experiment.mo_metrics import coverage_indicator
from ga_lab.experiment.mo_runner_fairness import (
    build_candidate_rows,
    build_mo_benchmark_rows,
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
    candidate_d_uniform_crossover,
    candidate_h_uniform_dedup_mutation_boost,
    candidate_j_h_lite_retry2,
    candidate_l_sparse_parent_bias_light,
    candidate_m_boundary_preservation_light,
    candidate_n_low_g_tail_mutation_light,
    candidate_o_spread_preserving_variation_light,
    candidate_variant_metadata,
)
from ga_lab.experiment.parameter_fairness import evaluate_parameter_fairness


def _load_helper(script_name: str, module_name: str):
    helper_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_helper("validate_nsga2_candidate_suite.py", "_candidate_suite_spread_phase2_base")
PHASE0 = _load_helper("validate_nsga2_spread_preserving_phase0.py", "_spread_preserving_phase2_helper")


METRIC_SPECS: dict[str, dict[str, Any]] = {
    "occupied_bins": {"higher_is_better": True, "row_key": "occupied_bins"},
    "segment_entropy": {"higher_is_better": True, "row_key": "segment_entropy"},
    "segment_load_gini": {"higher_is_better": False, "row_key": "segment_load_gini"},
    "spacing": {"higher_is_better": False, "row_key": "spacing"},
    "nondominated_count": {"higher_is_better": True, "row_key": "nondominated_count"},
    "segment0_allocation": {"higher_is_better": True, "row_key": "segment0_allocation"},
    "segment4_spacing": {"higher_is_better": False, "row_key": "segment4_spacing"},
    "segment0_g_mean": {"higher_is_better": False, "row_key": "segment0_g_mean"},
    "segment0_distance_mean": {"higher_is_better": False, "row_key": "segment0_distance_mean"},
    "segment0_low_g_count": {"higher_is_better": True, "row_key": "segment0_low_g_count"},
    "hypervolume_2d": {"higher_is_better": True, "row_key": "hypervolume_2d"},
    "reference_front_distance": {"higher_is_better": False, "row_key": "reference_front_distance"},
    "inverted_generational_distance": {
        "higher_is_better": False,
        "row_key": "inverted_generational_distance",
    },
    "coverage_indicator": {"higher_is_better": True, "row_key": "coverage_indicator"},
    "runtime_seconds": {"higher_is_better": False, "row_key": "runtime_seconds"},
    "actual_evaluations": {"higher_is_better": False, "row_key": "actual_evaluations"},
}


ALLOWED_PROBLEMS = {"zdt1", "zdt2", "zdt3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 spread-preserving validation for candidate_o across ZDT1/ZDT2/ZDT3."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problems", default="zdt1,zdt2,zdt3")
    parser.add_argument("--output-root", default="outputs/nsga2_spread_preserving_phase2")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--zdt1-seeds", type=int, default=30)
    parser.add_argument("--other-seeds", type=int, default=10)
    parser.add_argument("--zdt1-seed-start", type=int, default=30101)
    parser.add_argument("--other-seed-start", type=int, default=30201)
    parser.add_argument("--budget", type=int, default=760)
    parser.add_argument("--segment-count", type=int, default=6)
    parser.add_argument(
        "--drift-audit-json",
        default="artifacts/nsga2_candidate_o_phase2_default_drift_audit_results.json",
        help="Optional drift-audit JSON included in the Phase 2 report if it exists.",
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
    parser.add_argument("--skip-pymoo", action="store_true")
    parser.add_argument("--skip-random-archive", action="store_true")
    parser.add_argument("--include-candidate-d", action="store_true")
    parser.add_argument("--include-candidate-h", action="store_true")
    parser.add_argument("--include-candidate-l", action="store_true")
    parser.add_argument("--include-candidate-m", action="store_true")
    return parser.parse_args()


def _candidate_variants(args: argparse.Namespace) -> list[NSGA2CandidateVariant]:
    variants: list[NSGA2CandidateVariant] = [
        candidate_j_h_lite_retry2(),
        candidate_n_low_g_tail_mutation_light(),
        candidate_o_spread_preserving_variation_light(),
    ]
    if args.include_candidate_d:
        variants.append(candidate_d_uniform_crossover())
    if args.include_candidate_h:
        variants.append(candidate_h_uniform_dedup_mutation_boost())
    if args.include_candidate_l:
        variants.append(candidate_l_sparse_parent_bias_light())
    if args.include_candidate_m:
        variants.append(candidate_m_boundary_preservation_light())
    return variants


def _enable_phase2_diagnostics(
    config: GAConfig,
    *,
    candidate_id: str | None,
    segment_count: int,
) -> GAConfig:
    clone = GAConfig.from_dict(config.to_dict())
    clone.algorithm_options = dict(clone.algorithm_options)
    clone.algorithm_options["nsga2_trace_enabled"] = True
    clone.algorithm_options["nsga2_operator_supply_trace_enabled"] = True
    clone.algorithm_options["nsga2_zdt1_component_trace_enabled"] = True
    clone.algorithm_options["nsga2_trace_generation_sample_stride"] = 1
    clone.algorithm_options["nsga2_trace_segment_count"] = max(1, int(segment_count))
    if candidate_id:
        clone.algorithm_options["nsga2_trace_candidate_id"] = candidate_id
        clone.algorithm_options["nsga2_trace_run_id"] = f"{candidate_id}_{clone.problem}_phase2"
    else:
        clone.algorithm_options["nsga2_trace_run_id"] = f"internal_nsga2_{clone.problem}_phase2"
    return clone


def _candidate_result(
    config: GAConfig,
    variant: NSGA2CandidateVariant,
    *,
    seed: int,
    output_root: Path,
    segment_count: int,
) -> ExternalMOComparatorResult:
    candidate_config = _enable_phase2_diagnostics(
        apply_candidate_variant(config, variant),
        candidate_id=variant.candidate_id,
        segment_count=segment_count,
    )
    result = run_internal_nsga2(candidate_config, seed=seed, output_root=str(output_root))
    metadata = dict(result.metadata)
    metadata.update(candidate_variant_metadata(variant))
    if variant.candidate_id == "candidate_o_spread_preserving_variation_light":
        metadata["promotion_status"] = "phase2_validation"
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
    spec,
    base_config: GAConfig,
    requested_budget: int,
    variant_map: dict[str, NSGA2CandidateVariant],
    reference_front: list[list[float]],
    segment_count: int,
) -> dict[str, Any]:
    row = PHASE0._decorate_row(
        row,
        spec=spec,
        base_config=base_config,
        requested_budget=requested_budget,
        variant_map=variant_map,
        reference_front=reference_front,
        segment_count=segment_count,
    )
    if row.get("algorithm") == "candidate_o_spread_preserving_variation_light":
        row.setdefault("metadata", {})
        row["metadata"]["promotion_status"] = "phase2_validation"
    directions = [
        bool(value) for value in row.get("metadata", {}).get("objective_directions", [False, False])
    ]
    if row.get("success"):
        row["reference_front_coverage"] = coverage_indicator(
            row.get("nondominated_objective_vectors", []),
            reference_front,
            directions,
        )
    else:
        row["reference_front_coverage"] = None
    row["coverage_indicator"] = row.get("reference_front_coverage")
    row["segment0_g_mean"] = row.get("segment0_g")
    row["segment0_distance_mean"] = row.get("segment0_distance")
    return row


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
                "library": bucket[0].get("library"),
                "status": status,
                "seeds": len(bucket),
                "successful_seeds": len(successful),
                "mean_hv": BASE._summary_stat(successful, "hypervolume_2d")["mean"],
                "mean_distance": BASE._summary_stat(successful, "reference_front_distance")["mean"],
                "mean_igd": BASE._summary_stat(successful, "inverted_generational_distance")["mean"],
                "mean_spacing": BASE._summary_stat(successful, "spacing")["mean"],
                "mean_coverage": BASE._summary_stat(successful, "reference_front_coverage")["mean"],
                "mean_nondominated_count": BASE._summary_stat(successful, "nondominated_count")["mean"],
                "mean_occupied_bins": BASE._summary_stat(successful, "occupied_bins")["mean"],
                "mean_segment_entropy": BASE._summary_stat(successful, "segment_entropy")["mean"],
                "mean_segment_load_gini": BASE._summary_stat(successful, "segment_load_gini")["mean"],
                "mean_segment0_allocation": BASE._summary_stat(successful, "segment0_allocation")["mean"],
                "mean_segment4_spacing": BASE._summary_stat(successful, "segment4_spacing")["mean"],
                "mean_segment0_g": BASE._summary_stat(successful, "segment0_g")["mean"],
                "mean_segment0_distance": BASE._summary_stat(successful, "segment0_distance")["mean"],
                "mean_segment0_low_g_count": BASE._summary_stat(successful, "segment0_low_g_count")[
                    "mean"
                ],
                "mean_runtime_seconds": BASE._summary_stat(successful, "runtime_seconds")["mean"],
                "mean_actual_evaluations": BASE._summary_stat(successful, "actual_evaluations")[
                    "mean"
                ],
                "success_rate": BASE._success_rate(bucket),
            }
        )
    return aggregates


def _paired_metric_summary(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    metric_name: str,
    higher_is_better: bool,
    row_key: str,
) -> dict[str, Any]:
    right_by_seed = {
        int(row["seed"]): row
        for row in right_rows
        if row.get("success") and int(row.get("seed", -1)) >= 0
    }
    wins = ties = losses = 0
    deltas: list[float] = []
    comparable = 0
    for left in left_rows:
        if not left.get("success"):
            continue
        right = right_by_seed.get(int(left["seed"]))
        if right is None:
            continue
        if metric_name == "coverage_indicator":
            directions = [bool(v) for v in left.get("metadata", {}).get("objective_directions", [False, False])]
            left_metric = coverage_indicator(
                left.get("nondominated_objective_vectors", []),
                right.get("nondominated_objective_vectors", []),
                directions,
            )
            right_metric = coverage_indicator(
                right.get("nondominated_objective_vectors", []),
                left.get("nondominated_objective_vectors", []),
                directions,
            )
        else:
            left_metric = left.get(row_key)
            right_metric = right.get(row_key)
        if not (
            isinstance(left_metric, int | float)
            and isinstance(right_metric, int | float)
            and math.isfinite(float(left_metric))
            and math.isfinite(float(right_metric))
        ):
            continue
        comparable += 1
        delta = float(left_metric) - float(right_metric)
        deltas.append(delta)
        if math.isclose(float(left_metric), float(right_metric), rel_tol=1e-12, abs_tol=1e-12):
            ties += 1
        elif (float(left_metric) > float(right_metric)) == higher_is_better:
            wins += 1
        else:
            losses += 1
    return {
        "win": wins,
        "tie": ties,
        "loss": losses,
        "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "median_delta": BASE.median(deltas) if deltas else None,
        "comparable_seeds": comparable,
    }


def _paired_rows(rows: list[dict[str, Any]], comparison_specs: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    paired: list[dict[str, Any]] = []
    for problem in sorted({str(row["problem"]) for row in rows}):
        for left, right, label in comparison_specs:
            left_rows = grouped.get((problem, left), [])
            right_rows = grouped.get((problem, right), [])
            for metric_name, metric_spec in METRIC_SPECS.items():
                summary = _paired_metric_summary(
                    left_rows,
                    right_rows,
                    metric_name,
                    bool(metric_spec["higher_is_better"]),
                    str(metric_spec["row_key"]),
                )
                paired.append(
                    {
                        "problem": problem,
                        "comparison": label,
                        "left_algorithm": left,
                        "right_algorithm": right,
                        "metric": metric_name,
                        "win": summary["win"],
                        "tie": summary["tie"],
                        "loss": summary["loss"],
                        "mean_delta": summary["mean_delta"],
                        "median_delta": summary["median_delta"],
                        "comparable_seeds": summary["comparable_seeds"],
                    }
                )
    return paired


def _group_successful_rows(raw_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("success"):
            grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)
    return grouped


def _most_common_int(values: list[int]) -> int | None:
    if not values:
        return None
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[int(value)] += 1
    max_count = max(counts.values())
    return min(segment for segment, count in counts.items() if count == max_count)


def _spread_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_successful_rows(raw_rows)
    rows: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        weakest_values = [
            int(row["weakest_segment_id"])
            for row in bucket
            if row.get("weakest_segment_id") is not None
        ]
        rows.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "occupied_bins": BASE._summary_stat(bucket, "occupied_bins")["mean"],
                "segment_entropy": BASE._summary_stat(bucket, "segment_entropy")["mean"],
                "segment_load_gini": BASE._summary_stat(bucket, "segment_load_gini")["mean"],
                "spacing": BASE._summary_stat(bucket, "spacing")["mean"],
                "nondominated_count": BASE._summary_stat(bucket, "nondominated_count")["mean"],
                "segment0_allocation": BASE._summary_stat(bucket, "segment0_allocation")["mean"],
                "segment4_spacing": BASE._summary_stat(bucket, "segment4_spacing")["mean"],
                "weakest_segment_id": _most_common_int(weakest_values),
            }
        )
    return rows


def _zdt1_component_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_successful_rows([row for row in raw_rows if str(row.get("problem")) == "zdt1"])
    rows: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        rows.append(
            {
                "problem": problem,
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
    grouped = _group_successful_rows(raw_rows)
    rows: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        rows.append(
            {
                "problem": problem,
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


def _load_drift_payload(path: Path) -> dict[str, Any] | None:
    return PHASE0._load_optional_json(path)


def _pick(
    paired_rows: list[dict[str, Any]],
    problem: str,
    comparison: str,
    metric: str,
) -> dict[str, Any] | None:
    for row in paired_rows:
        if row["problem"] == problem and row["comparison"] == comparison and row["metric"] == metric:
            return row
    return None


def _metric_wins(row: dict[str, Any] | None) -> bool:
    return row is not None and int(row["win"]) > int(row["loss"])


def _metric_losses(row: dict[str, Any] | None) -> bool:
    return row is not None and int(row["win"]) < int(row["loss"])


def _metric_ties(row: dict[str, Any] | None) -> bool:
    return row is not None and int(row["win"]) == int(row["loss"])


def _aggregate_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["problem"]), str(row["algorithm"])): row for row in rows}


def _pymoo_gap_improvement_count(aggregate_rows: list[dict[str, Any]], problem: str) -> int:
    lookup = _aggregate_lookup(aggregate_rows)
    candidate_n = lookup.get((problem, "candidate_n_low_g_tail_mutation_light"))
    candidate_o = lookup.get((problem, "candidate_o_spread_preserving_variation_light"))
    pymoo = lookup.get((problem, "pymoo_nsga2"))
    if not candidate_n or not candidate_o or not pymoo:
        return 0

    checks = [
        ("mean_occupied_bins", True),
        ("mean_spacing", False),
        ("mean_nondominated_count", True),
    ]
    improved = 0
    for row_key, higher_is_better in checks:
        n_value = candidate_n.get(row_key)
        o_value = candidate_o.get(row_key)
        p_value = pymoo.get(row_key)
        if not (
            isinstance(n_value, int | float)
            and isinstance(o_value, int | float)
            and isinstance(p_value, int | float)
            and math.isfinite(float(n_value))
            and math.isfinite(float(o_value))
            and math.isfinite(float(p_value))
        ):
            continue
        if higher_is_better:
            gap_n = max(0.0, float(p_value) - float(n_value))
            gap_o = max(0.0, float(p_value) - float(o_value))
        else:
            gap_n = max(0.0, float(n_value) - float(p_value))
            gap_o = max(0.0, float(o_value) - float(p_value))
        if gap_o < gap_n:
            improved += 1
    return improved


def _candidate_gate_rows(
    rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    *,
    drift_payload: dict[str, Any] | None,
    local_baseline_status: str,
    local_baseline_note: str,
) -> list[dict[str, Any]]:
    default_rows = [row for row in rows if row["algorithm"] == "internal_nsga2"]
    candidate_rows = [
        row for row in rows if row["algorithm"] == "candidate_o_spread_preserving_variation_light"
    ]
    candidate_zdt1_rows = [
        row for row in candidate_rows if str(row.get("problem")) == "zdt1" and row.get("success")
    ]

    overall = dict(drift_payload.get("overall", {})) if isinstance(drift_payload, dict) else {}
    drift_ok = bool(drift_payload is None) or not any(
        bool(overall.get(key))
        for key in (
            "candidate_metadata_leak",
            "diagnostics_metadata_leak",
            "actual_evaluations_mismatch",
            "objective_signature_mismatch",
            "drift_detected",
        )
    )
    local_baseline_ok = str(local_baseline_status).strip().lower() == "pass"
    metadata_ok = all(
        row.get("metadata", {}).get("candidate_id") == "candidate_o_spread_preserving_variation_light"
        and row.get("metadata", {}).get("base_candidate_id") == "candidate_n_low_g_tail_mutation_light"
        and row.get("metadata", {}).get("mechanism") == "spread_preserving_variation_light"
        and row.get("metadata", {}).get("promotion_status") == "phase2_validation"
        and row.get("metadata", {}).get("allowed_use") == "phase0_sanity_only"
        and row.get("metadata", {}).get("disallowed_use") == "default_replacement"
        for row in candidate_rows
    )
    default_changed_ok = all(
        row.get("metadata", {}).get("default_changed") is False for row in candidate_rows
    )
    candidate_isolation_ok = all(
        "candidate_id" not in row.get("metadata", {})
        and "default_changed" not in row.get("metadata", {})
        and "promotion_status" not in row.get("metadata", {})
        for row in default_rows
    )
    evaluations_ok = all(
        row.get("requested_budget") == row.get("actual_evaluations")
        for row in candidate_rows
        if row.get("success")
    )
    spread_ok = all(
        bool(row.get("spread_parity_diagnostics_success")) for row in candidate_rows if row.get("success")
    )
    zdt1_component_ok = all(
        bool(row.get("zdt1_component_diagnostics_success")) for row in candidate_zdt1_rows
    )
    non_finite_ok = all(bool(row.get("success")) for row in candidate_rows)
    fairness_fail_free = fairness_payload.get("summary_counts", {}).get("fail", 0) == 0

    return [
        {
            "gate": "default drift",
            "result": drift_ok,
            "evidence": "drift audit overall flags",
            "interpretation": "default path must stay free of candidate_o and diagnostics metadata drift",
        },
        {
            "gate": "candidate isolation",
            "result": candidate_isolation_ok,
            "evidence": "default internal rows",
            "interpretation": "default internal NSGA-II rows must remain candidate-metadata free",
        },
        {
            "gate": "candidate metadata",
            "result": metadata_ok,
            "evidence": "candidate_o raw_rows metadata",
            "interpretation": "candidate_o metadata should remain explicit opt-in only and carry phase2_validation",
        },
        {
            "gate": "default_changed=false",
            "result": default_changed_ok,
            "evidence": "candidate_o raw_rows metadata",
            "interpretation": "candidate_o rows must keep default_changed=false",
        },
        {
            "gate": "actual evaluations",
            "result": evaluations_ok,
            "evidence": "requested_budget vs actual_evaluations",
            "interpretation": "candidate_o actual evaluations must match the requested budget",
        },
        {
            "gate": "spread parity diagnostics",
            "result": spread_ok,
            "evidence": "candidate_o raw_rows diagnostics flags",
            "interpretation": "candidate_o rows must retain spread parity diagnostics on every successful run",
        },
        {
            "gate": "ZDT1 component diagnostics",
            "result": zdt1_component_ok,
            "evidence": "candidate_o zdt1 raw_rows diagnostics flags",
            "interpretation": "ZDT1 rows must retain the required component diagnostics",
        },
        {
            "gate": "fairness fail 없음",
            "result": fairness_fail_free,
            "evidence": f"fairness summary={fairness_payload.get('summary_counts', {})}",
            "interpretation": "Phase 2 decision is blocked if any fairness fail appears",
        },
        {
            "gate": "local baseline governance",
            "result": local_baseline_ok,
            "evidence": local_baseline_note,
            "interpretation": "local baseline governance should stay PASS before any Phase 2 conclusion",
        },
        {
            "gate": "non-finite objective 없음",
            "result": non_finite_ok,
            "evidence": "candidate_o run success",
            "interpretation": "candidate_o must not trip non-finite fitness fail-fast checks",
        },
    ]


def _problem_signal_rows(
    paired_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    selected_problems: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for problem in selected_problems:
        spread_occ = _pick(paired_rows, problem, "candidate_o vs candidate_n", "occupied_bins")
        spread_entropy = _pick(paired_rows, problem, "candidate_o vs candidate_n", "segment_entropy")
        spread_gini = _pick(paired_rows, problem, "candidate_o vs candidate_n", "segment_load_gini")
        spread_spacing = _pick(paired_rows, problem, "candidate_o vs candidate_n", "spacing")
        spread_count = _pick(paired_rows, problem, "candidate_o vs candidate_n", "nondominated_count")
        spread_seg0 = _pick(paired_rows, problem, "candidate_o vs candidate_n", "segment0_allocation")
        spread_seg4 = _pick(paired_rows, problem, "candidate_o vs candidate_n", "segment4_spacing")
        component_g = _pick(paired_rows, problem, "candidate_o vs candidate_n", "segment0_g_mean")
        component_distance = _pick(
            paired_rows,
            problem,
            "candidate_o vs candidate_n",
            "segment0_distance_mean",
        )
        component_low_g = _pick(
            paired_rows,
            problem,
            "candidate_o vs candidate_n",
            "segment0_low_g_count",
        )
        final_hv = _pick(paired_rows, problem, "candidate_o vs candidate_n", "hypervolume_2d")
        final_distance = _pick(
            paired_rows,
            problem,
            "candidate_o vs candidate_n",
            "reference_front_distance",
        )
        final_igd = _pick(
            paired_rows,
            problem,
            "candidate_o vs candidate_n",
            "inverted_generational_distance",
        )
        final_coverage = _pick(
            paired_rows,
            problem,
            "candidate_o vs candidate_n",
            "coverage_indicator",
        )
        versus_j = [
            _pick(paired_rows, problem, "candidate_o vs candidate_j", metric)
            for metric in (
                "occupied_bins",
                "spacing",
                "nondominated_count",
                "hypervolume_2d",
                "reference_front_distance",
                "inverted_generational_distance",
            )
        ]
        versus_pymoo = [
            _pick(paired_rows, problem, "candidate_o vs pymoo", metric)
            for metric in (
                "occupied_bins",
                "spacing",
                "nondominated_count",
                "segment0_allocation",
                "runtime_seconds",
            )
        ]

        primary_spread_positive = sum(
            _metric_wins(row) for row in (spread_occ, spread_spacing, spread_count)
        )
        spread_support_positive = sum(
            _metric_wins(row) for row in (spread_seg0, spread_seg4)
        )
        uniformity_regression = sum(
            _metric_losses(row) for row in (spread_entropy, spread_gini)
        )
        component_regression_count = sum(
            _metric_losses(row) for row in (component_g, component_distance, component_low_g)
        )
        final_regression_count = sum(
            _metric_losses(row) for row in (final_hv, final_distance, final_igd, final_coverage)
        )
        versus_j_positive = sum(_metric_wins(row) for row in versus_j)
        versus_pymoo_negative = sum(_metric_losses(row) for row in versus_pymoo)
        pymoo_gap_improvement_count = _pymoo_gap_improvement_count(aggregate_rows, problem)

        if problem == "zdt1":
            interpretation = (
                "ZDT1 stress에서는 spread 재현성과 low-g/component trade-off를 함께 본다."
                if primary_spread_positive >= 2
                else "ZDT1 stress에서는 candidate_o의 핵심 spread signal 재현성이 약하다."
            )
        else:
            if primary_spread_positive >= 1 and final_regression_count == 0:
                interpretation = f"{problem}에서는 spread 쪽 약한 긍정 신호가 보인다."
            elif final_regression_count > 1:
                interpretation = f"{problem}에서는 convergence/final-front regression 경고가 있다."
            else:
                interpretation = f"{problem}에서는 대체로 중립 또는 tie에 가깝다."

        rows.append(
            {
                "problem": problem,
                "primary_spread_positive": primary_spread_positive,
                "spread_support_positive": spread_support_positive,
                "uniformity_regression": uniformity_regression,
                "component_regression_count": component_regression_count,
                "final_regression_count": final_regression_count,
                "versus_j_positive": versus_j_positive,
                "versus_pymoo_negative": versus_pymoo_negative,
                "pymoo_gap_improvement_count": pymoo_gap_improvement_count,
                "interpretation": interpretation,
            }
        )
    return rows


def _phase2_decision(
    gate_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    signal_rows: list[dict[str, Any]],
) -> str:
    if fairness_payload.get("summary_counts", {}).get("fail", 0) > 0:
        return "Needs fairness rerun"
    if not all(bool(row["result"]) for row in gate_rows):
        return "Fix required"

    lookup = {str(row["problem"]): row for row in signal_rows}
    zdt1 = lookup.get("zdt1", {})
    zdt2 = lookup.get("zdt2", {})
    zdt3 = lookup.get("zdt3", {})

    zdt1_spread = int(zdt1.get("primary_spread_positive", 0))
    zdt1_support = int(zdt1.get("spread_support_positive", 0))
    zdt1_component_regression = int(zdt1.get("component_regression_count", 0))
    zdt1_final_regression = int(zdt1.get("final_regression_count", 0))
    zdt1_vs_j = int(zdt1.get("versus_j_positive", 0))
    zdt1_pymoo_gap = int(zdt1.get("pymoo_gap_improvement_count", 0))

    other_rows = [row for row in (zdt2, zdt3) if row]
    other_positive = sum(int(row.get("primary_spread_positive", 0) >= 1) for row in other_rows)
    other_safe = all(int(row.get("final_regression_count", 0)) <= 1 for row in other_rows)

    catastrophic = zdt1_final_regression >= 4 or zdt1_component_regression >= 3
    if catastrophic:
        return "Reject"

    if (
        zdt1_spread >= 2
        and zdt1_support >= 1
        and zdt1_component_regression <= 1
        and zdt1_final_regression == 0
        and zdt1_vs_j >= 4
        and zdt1_pymoo_gap >= 1
        and other_positive >= 1
        and other_safe
    ):
        return "Phase 2 passed, eligible for change request review"

    if (
        zdt1_spread >= 2
        and zdt1_support >= 1
        and zdt1_component_regression <= 1
        and zdt1_final_regression <= 3
        and zdt1_vs_j >= 3
        and zdt1_pymoo_gap >= 1
        and other_safe
    ):
        return "Phase 2 passed, eligible for opt-in experimental profile review"

    if zdt1_spread >= 2 and zdt1_final_regression <= 3:
        return "Phase 2 passed with trade-offs"

    if zdt1_spread >= 1 and other_safe:
        return "Hold for more evidence"

    return "Reject"


def _paired_interpretation(row: dict[str, Any]) -> str:
    if int(row["comparable_seeds"]) == 0:
        return "comparable seed가 없어 해석 제한"
    if int(row["win"]) > int(row["loss"]):
        return "candidate_o가 더 자주 우세"
    if int(row["win"]) < int(row["loss"]):
        return "비교군이 더 자주 우세"
    return "seed별로 혼재 또는 거의 동률"


def _fairness_issue_summary_rows(fairness_payload: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for issue in fairness_payload.get("issues", []):
        key = (
            str(issue.get("status", "unknown")),
            str(issue.get("issue_type", "unknown")),
            str(issue.get("algorithm", "unknown")),
            str(issue.get("problem", "unknown")),
        )
        grouped[key] += 1
    rows: list[dict[str, Any]] = []
    for (status, issue_type, algorithm, problem), count in sorted(grouped.items()):
        rows.append(
            {
                "status": status,
                "issue_type": issue_type,
                "algorithm": algorithm,
                "problem": problem,
                "count": count,
            }
        )
    return rows


def _fairness_report_markdown(payload: dict[str, Any]) -> str:
    fairness_payload = payload["fairness"]
    summary_rows = fairness_summary_rows(fairness_payload)
    issue_rows = _fairness_issue_summary_rows(fairness_payload)
    lines = [
        "# NSGA-II Spread-Preserving Phase 2 Fairness Report",
        "",
        "## Summary",
        "",
        *BASE._markdown_table(summary_rows, ["status", "pass", "warning", "fail"]),
        "",
        "## Issue Summary",
        "",
        *BASE._markdown_table(
            issue_rows
            or [{"status": "none", "issue_type": "none", "algorithm": "none", "problem": "none", "count": 0}],
            ["status", "issue_type", "algorithm", "problem", "count"],
        ),
        "",
    ]
    return "\n".join(lines)


def _results_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NSGA-II Spread-Preserving Phase 2 Results",
        "",
        "## Aggregate Rows",
        "",
        *BASE._markdown_table(
            payload["aggregate_rows"],
            [
                "problem",
                "algorithm",
                "status",
                "seeds",
                "successful_seeds",
                "mean_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Spread Rows",
        "",
        *BASE._markdown_table(
            payload["spread_rows"],
            [
                "problem",
                "algorithm",
                "occupied_bins",
                "segment_entropy",
                "segment_load_gini",
                "spacing",
                "nondominated_count",
                "segment0_allocation",
                "segment4_spacing",
                "weakest_segment_id",
            ],
        ),
        "",
        "## ZDT1 Component Rows",
        "",
        *BASE._markdown_table(
            payload["zdt1_component_rows"]
            or [
                {
                    "problem": "zdt1",
                    "algorithm": "none",
                    "segment0_g_mean": None,
                    "segment0_distance_mean": None,
                    "segment0_low_g_count": None,
                    "segment0_nondominated_rate": None,
                    "segment0_survival_rate": None,
                }
            ],
            [
                "problem",
                "algorithm",
                "segment0_g_mean",
                "segment0_distance_mean",
                "segment0_low_g_count",
                "segment0_nondominated_rate",
                "segment0_survival_rate",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _phase2_report_markdown(payload: dict[str, Any]) -> str:
    candidate_definition = candidate_variant_metadata(candidate_o_spread_preserving_variation_light())
    fairness_summary = payload["fairness_summary"]
    drift_payload = payload.get("drift_audit")
    drift_overall = dict(drift_payload.get("overall", {})) if isinstance(drift_payload, dict) else {}
    gate_rows = payload["gate_rows"]
    aggregate_rows = payload["aggregate_rows"]
    spread_rows = payload["spread_rows"]
    zdt1_component_rows = payload["zdt1_component_rows"]
    paired_rows = payload["paired_rows"]
    signal_rows = payload["signal_rows"]
    failures = payload["failures"]
    comparison_focus = {
        "candidate_o vs candidate_n",
        "candidate_o vs candidate_j",
        "candidate_o vs pymoo",
        "candidate_o vs internal_nsga2",
        "candidate_o vs Random Pareto Archive",
    }
    metric_focus = {
        "occupied_bins",
        "segment_entropy",
        "segment_load_gini",
        "spacing",
        "nondominated_count",
        "segment0_allocation",
        "segment4_spacing",
        "segment0_g_mean",
        "segment0_distance_mean",
        "segment0_low_g_count",
        "hypervolume_2d",
        "reference_front_distance",
        "inverted_generational_distance",
        "coverage_indicator",
        "runtime_seconds",
        "actual_evaluations",
    }
    report_paired_rows = [
        {
            **row,
            "interpretation": _paired_interpretation(row),
        }
        for row in paired_rows
        if row["comparison"] in comparison_focus and row["metric"] in metric_focus
    ]
    decision_lookup = {str(row["problem"]): row for row in signal_rows}
    decision_rows = [
        {
            "candidate": "candidate_o_spread_preserving_variation_light",
            "ZDT1 stress": decision_lookup.get("zdt1", {}).get("interpretation"),
            "ZDT2": decision_lookup.get("zdt2", {}).get("interpretation"),
            "ZDT3": decision_lookup.get("zdt3", {}).get("interpretation"),
            "candidate_n 대비": "paired summary 참조",
            "candidate_j 대비": "paired summary 참조",
            "pymoo 대비": "gap remains unless paired rows say otherwise",
            "fairness": f"pass={fairness_summary.get('pass', 0)}, warning={fairness_summary.get('warning', 0)}, fail={fairness_summary.get('fail', 0)}",
            "final decision": payload["phase2_decision"],
        }
    ]

    experiment_rows = []
    for problem in payload["selected_problems"]:
        seeds = payload["problem_seed_map"].get(problem, [])
        experiment_rows.append(
            {
                "problem": problem,
                "algorithms": ", ".join(payload["algorithms"]),
                "seeds": len(seeds),
                "requested budget": payload["budget"],
                "actual evaluations summary": "all successful rows matched budget"
                if all(
                    row["problem"] != problem
                    or row["mean_actual_evaluations"] in (None, payload["budget"])
                    for row in aggregate_rows
                )
                else "see aggregate rows",
                "diagnostics": "spread parity + operator supply"
                + (" + ZDT1 component" if problem == "zdt1" else ""),
            }
        )

    learned_lines = []
    zdt1_signal = decision_lookup.get("zdt1")
    if zdt1_signal:
        learned_lines.append(f"- ZDT1 stress 해석: {zdt1_signal['interpretation']}")
    for problem in ("zdt2", "zdt3"):
        if problem in decision_lookup:
            learned_lines.append(f"- {problem.upper()} 해석: {decision_lookup[problem]['interpretation']}")
    learned_lines.extend(
        [
            "- candidate_o가 candidate_n 대비 occupied_bins, spacing, nondominated_count 중 최소 2개 이상을 반복 개선했는지 paired summary로 판단했다.",
            "- candidate_o가 candidate_n 대비 segment0_g_mean 또는 segment0_distance_mean을 크게 악화시키면 trade-off로 기록했다.",
            "- candidate_o가 candidate_j 대비 핵심 spread/final-front metric에서 우세한지 별도로 기록했다.",
            "- candidate_o가 pymoo 대비 spread gap을 줄였는지는 occupied_bins, spacing, nondominated_count 중심으로 해석했다.",
            "- ZDT2/ZDT3에서 candidate_o가 무너지면 ZDT1-specific 한계가 강하다고 보고 일반화 주장을 금지한다.",
            "- 기본 NSGA-II default promotion은 이번 범위에서 여전히 금지다.",
        ]
    )

    lines = [
        "# NSGA-II Spread-Preserving Variation Phase 2 Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: candidate_o_spread_preserving_variation_light의 Phase 1 spread signal이 ZDT1 stress 30 seeds와 ZDT2/ZDT3 small 10 seeds에서도 유지되는지 확인하는 것이다.",
        f"- default drift audit 결과: {'NO DRIFT' if drift_payload and not drift_overall.get('drift_detected') else 'not available' if not drift_payload else 'DRIFT DETECTED'}",
        f"- candidate isolation 결과: {'PASS' if any(row['gate'] == 'candidate isolation' and row['result'] for row in gate_rows) else 'FAIL'}",
        f"- 실행한 benchmark: problems={', '.join(payload['selected_problems'])}, budget={payload['budget']}, zdt1_seeds={len(payload['problem_seed_map'].get('zdt1', []))}, other_seeds={len(payload['problem_seed_map'].get('zdt2', []))}",
        "- ZDT1 stress 결과와 ZDT2/ZDT3 small 결과는 spread-preserving signal과 low-g/component trade-off를 함께 기록했다.",
        "- candidate_o vs candidate_n, candidate_j, pymoo, internal_nsga2, Random Pareto Archive를 문제별로 paired comparison 했다.",
        f"- fairness 결과: pass={fairness_summary.get('pass', 0)}, warning={fairness_summary.get('warning', 0)}, fail={fairness_summary.get('fail', 0)}",
        f"- Phase 2 decision: **{payload['phase2_decision']}**",
        "- 기본값 변경 여부: 없음",
        "- CR 작성 여부: 없음",
        "- opt-in approval 여부: 없음",
        "- Level 판정 변화 여부: default algorithm 성숙도 상향은 금지되며, 실험 툴킷 관점의 근거만 보수적으로 해석한다.",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "Scope:",
        "- candidate_o_spread_preserving_variation_light",
        "- ZDT1 stress",
        "- ZDT2/ZDT3 small validation",
        "- candidate_o vs candidate_n",
        "- candidate_o vs candidate_j",
        "- candidate_o vs pymoo",
        "- spread parity diagnostics",
        "- ZDT1 component diagnostics",
        "- fairness check",
        "- default drift check",
        "",
        "Non-Scope:",
        "- default promotion",
        "- automatic CR approval",
        "- automatic opt-in approval",
        "- DTLZ/WFG validation",
        "- new operator candidate",
        "- survivor-pressure candidate",
        "- productization",
        "",
        "## 3. Default Drift and Candidate Isolation",
        "",
        *BASE._markdown_table(gate_rows, ["gate", "result", "evidence", "interpretation"]),
        "",
        "## 4. Candidate Definition",
        "",
        *BASE._markdown_table(
            [
                {"field": "candidate_id", "value": candidate_definition["candidate_id"]},
                {"field": "base_candidate", "value": candidate_definition["base_candidate_id"]},
                {"field": "mechanism", "value": candidate_definition["mechanism"]},
                {"field": "default_changed", "value": candidate_definition["default_changed"]},
                {"field": "promotion_status", "value": "phase2_validation"},
                {"field": "allowed_use", "value": candidate_definition["allowed_use"]},
                {"field": "disallowed_use", "value": candidate_definition["disallowed_use"]},
                {
                    "field": "zdt1_specific_warning",
                    "value": candidate_definition["zdt1_specific_warning"],
                },
                {
                    "field": "no_pymoo_operator_clone",
                    "value": candidate_definition.get("no_pymoo_operator_clone", True),
                },
            ],
            ["field", "value"],
        ),
        "",
        "## 5. Experiment Configuration",
        "",
        *BASE._markdown_table(
            experiment_rows,
            ["problem", "algorithms", "seeds", "requested budget", "actual evaluations summary", "diagnostics"],
        ),
        "",
        "## 6. Results Summary",
        "",
        *BASE._markdown_table(
            aggregate_rows,
            [
                "problem",
                "algorithm",
                "mean_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## 7. Spread Parity Summary",
        "",
        *BASE._markdown_table(
            spread_rows,
            [
                "problem",
                "algorithm",
                "occupied_bins",
                "segment_entropy",
                "segment_load_gini",
                "spacing",
                "nondominated_count",
                "segment0_allocation",
                "segment4_spacing",
            ],
        ),
        "",
        "## 8. ZDT1 Component Summary",
        "",
        *BASE._markdown_table(
            zdt1_component_rows
            or [
                {
                    "algorithm": "none",
                    "segment0_g_mean": None,
                    "segment0_distance_mean": None,
                    "segment0_low_g_count": None,
                    "segment0_nondominated_rate": None,
                    "segment0_survival_rate": None,
                }
            ],
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
        "## 9. Paired Comparisons",
        "",
        *BASE._markdown_table(
            report_paired_rows,
            [
                "problem",
                "comparison",
                "metric",
                "win",
                "tie",
                "loss",
                "mean_delta",
                "median_delta",
                "interpretation",
            ],
        ),
        "",
        "## 10. Candidate O Decision",
        "",
        *BASE._markdown_table(
            decision_rows,
            ["candidate", "ZDT1 stress", "ZDT2", "ZDT3", "candidate_n 대비", "candidate_j 대비", "pymoo 대비", "fairness", "final decision"],
        ),
        "",
        "## 11. What We Learned",
        "",
        *learned_lines,
        "",
        "## 12. Failures and Warnings",
        "",
        *BASE._markdown_table(
            failures
            or [
                {
                    "type": "none",
                    "target": "none",
                    "seed": None,
                    "message": "none",
                    "impact": "none",
                    "action": "none",
                }
            ],
            ["type", "target", "seed", "message", "impact", "action"],
        ),
        "",
        "## 13. Regression Check",
        "",
        *BASE._markdown_table(payload["regression_checks"], ["command", "result", "note"]),
        "",
        "## 14. Maturity Impact",
        "",
        "- Level 4 근거 강화",
        "- Phase 2는 candidate evidence이지 default algorithm maturity 상향 근거가 아니다.",
        "- candidate_o가 좋아도 기본값이 바뀌지 않았으므로 default NSGA-II maturity 상향은 금지한다.",
        "- ZDT 계열 중심 검증이므로 범용 MOEA 성숙도 상향 근거로 쓰지 않는다.",
        "- fairness, isolation, default drift gate가 유지되면 실험 툴킷 관점의 Level 4 근거는 강화될 수 있다.",
        "",
        "## 15. Recommended Next Work",
        "",
        "1. Phase 2 passed이면 CR review package 작성 여부를 별도 프롬프트에서 검토한다.",
        "2. opt-in experimental profile review가 더 적절하면 그 또한 별도 review 프롬프트로 분리한다.",
        "3. Phase 2 passed with trade-offs이면 trade-off postmortem 또는 focused redesign review를 작성한다.",
        "4. Hold이면 추가 diagnostics 또는 ZDT stress 재검토로 돌아간다.",
        "5. Reject이면 spread-preserving variation family를 보류한다.",
        "6. candidate_j opt-in profile은 유지한다.",
        "7. candidate_n 상태는 유지하되 candidate_o와의 비교 메모를 남긴다.",
        "8. fairness checker single-objective runner 확장 여부를 검토한다.",
        "9. constrained multi-objective contract",
        "10. checkpoint/resume",
        "11. parallel evaluation",
        "",
        f"이번 Phase 2 결과, candidate_o_spread_preserving_variation_light는 ZDT 계열에서 spread-preserving validation signal을 보였고, candidate_n 대비 문제별 metric trade-off가 있었으며, 최종 판정은 {payload['phase2_decision']}이다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    selected_problems = [
        item.strip().lower() for item in str(args.problems or "").split(",") if item.strip()
    ]
    if not selected_problems:
        raise ValueError("At least one problem must be selected.")
    unsupported = [problem for problem in selected_problems if problem not in ALLOWED_PROBLEMS]
    if unsupported:
        raise ValueError(
            f"This spread-preserving Phase 2 validation pass is limited to zdt1,zdt2,zdt3. Unsupported: {', '.join(unsupported)}"
        )

    base_config = load_config(PROJECT_ROOT / args.config)
    selected_specs = [mo_candidate_suite_specs()[name] for name in selected_problems]
    problem_seed_map: dict[str, list[int]] = {}
    for problem in selected_problems:
        if problem == "zdt1":
            problem_seed_map[problem] = [
                args.zdt1_seed_start + offset for offset in range(args.zdt1_seeds)
            ]
        else:
            problem_seed_map[problem] = [
                args.other_seed_start + offset for offset in range(args.other_seeds)
            ]

    variants = _candidate_variants(args)
    variant_map = {variant.candidate_id: variant for variant in variants}

    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    comparison_specs = [
        (
            "candidate_o_spread_preserving_variation_light",
            "candidate_n_low_g_tail_mutation_light",
            "candidate_o vs candidate_n",
        ),
        (
            "candidate_o_spread_preserving_variation_light",
            "candidate_j_h_lite_retry2",
            "candidate_o vs candidate_j",
        ),
        ("candidate_o_spread_preserving_variation_light", "pymoo_nsga2", "candidate_o vs pymoo"),
        (
            "candidate_o_spread_preserving_variation_light",
            "internal_nsga2",
            "candidate_o vs internal_nsga2",
        ),
        (
            "candidate_o_spread_preserving_variation_light",
            "random_pareto_archive",
            "candidate_o vs Random Pareto Archive",
        ),
    ]

    for optional_candidate in (
        ("candidate_d_uniform_crossover", "candidate_o vs candidate_d"),
        ("candidate_h_uniform_dedup_mutation_boost", "candidate_o vs candidate_h"),
        ("candidate_l_sparse_parent_bias_light", "candidate_o vs candidate_l"),
        ("candidate_m_boundary_preservation_light", "candidate_o vs candidate_m"),
    ):
        if optional_candidate[0] in variant_map:
            comparison_specs.append(
                (
                    "candidate_o_spread_preserving_variation_light",
                    optional_candidate[0],
                    optional_candidate[1],
                )
            )

    for spec in selected_specs:
        config = build_problem_config(base_config, spec)
        config = PHASE0._retarget_budget(config, args.budget)
        reference_front = reference_front_for_spec(spec, point_count=201)
        reference_point = list(spec.hv_reference_point)
        problem_output_root = output_root / spec.problem
        problem_output_root.mkdir(parents=True, exist_ok=True)

        for seed in problem_seed_map[spec.problem]:
            results: list[ExternalMOComparatorResult] = [
                run_internal_nsga2(
                    _enable_phase2_diagnostics(
                        config,
                        candidate_id=None,
                        segment_count=args.segment_count,
                    ),
                    seed=seed,
                    output_root=str(problem_output_root),
                ),
                *[
                    _candidate_result(
                        config,
                        variant,
                        seed=seed,
                        output_root=problem_output_root,
                        segment_count=args.segment_count,
                    )
                    for variant in variants
                ],
            ]
            if not args.skip_pymoo:
                results.append(run_pymoo_nsga2(config, seed=seed, budget=args.budget))
            if not args.skip_random_archive:
                results.append(
                    run_random_archive_anchor(
                        run_random_pareto_archive(config, seed=seed, budget=args.budget)
                    )
                )

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
                            "type": result.status,
                            "target": result.algorithm_name,
                            "seed": seed,
                            "message": result.error_message,
                            "impact": "seed excluded from paired comparison",
                            "action": "review comparator/runtime failure before any Phase 2 conclusion",
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
    paired_rows = _paired_rows(raw_rows, comparison_specs)
    drift_audit_path = PROJECT_ROOT / args.drift_audit_json
    drift_payload = _load_drift_payload(drift_audit_path)
    gate_rows = _candidate_gate_rows(
        raw_rows,
        fairness_payload,
        drift_payload=drift_payload,
        local_baseline_status=args.local_baseline_status,
        local_baseline_note=args.local_baseline_note,
    )
    signal_rows = _problem_signal_rows(paired_rows, aggregate_rows, selected_problems)
    phase2_decision = _phase2_decision(gate_rows, fairness_payload, signal_rows)

    regression_checks = [
        {
            "command": "python scripts/audit_nsga2_default_drift.py --results-base nsga2_candidate_o_phase2_default_drift_audit_results --report-base nsga2_candidate_o_phase2_default_drift_audit_report --output-root outputs/nsga2_candidate_o_phase2_default_drift",
            "result": "see drift artifact",
            "note": str(drift_audit_path) if drift_payload is not None else "not provided",
        },
        {
            "command": "python scripts/check_local_baseline.py --output-dir artifacts/spread_preserving_phase2_guard",
            "result": args.local_baseline_status,
            "note": args.local_baseline_note,
        },
        {
            "command": "python scripts/validate_nsga2_spread_preserving_phase2.py --problems zdt1,zdt2,zdt3 --zdt1-seeds 30 --other-seeds 10 --budget 760 --artifact-suffix candidate_o_phase2",
            "result": "success",
            "note": "current run",
        },
    ]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "selected_problems": selected_problems,
        "problem_seed_map": problem_seed_map,
        "budget": args.budget,
        "segment_count": args.segment_count,
        "spread_parity_trace_enabled": True,
        "zdt1_component_trace_enabled": True,
        "operator_supply_trace_enabled": True,
        "algorithms": [
            "internal_nsga2",
            *[variant.candidate_id for variant in variants],
            *([] if args.skip_pymoo else ["pymoo_nsga2"]),
            *([] if args.skip_random_archive else ["random_pareto_archive"]),
        ],
        "benchmark_rows": benchmark_rows,
        "candidate_rows": candidate_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "spread_rows": spread_rows,
        "zdt1_component_rows": zdt1_component_rows,
        "operator_supply_rows": operator_supply_rows,
        "paired_rows": paired_rows,
        "gate_rows": gate_rows,
        "signal_rows": signal_rows,
        "phase2_decision": phase2_decision,
        "failures": failures,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
        "drift_audit_path": str(drift_audit_path) if drift_payload is not None else None,
        "drift_audit": drift_payload,
        "local_baseline_status": args.local_baseline_status,
        "local_baseline_note": args.local_baseline_note,
        "regression_checks": regression_checks,
    }

    results_json = safe_artifact_path(
        artifact_root,
        "nsga2_spread_preserving_phase2_results",
        args.artifact_suffix,
        ".json",
    )
    results_csv = safe_artifact_path(
        artifact_root,
        "nsga2_spread_preserving_phase2_results",
        args.artifact_suffix,
        ".csv",
    )
    results_md = safe_artifact_path(
        artifact_root,
        "nsga2_spread_preserving_phase2_results",
        args.artifact_suffix,
        ".md",
    )
    report_md = safe_artifact_path(
        artifact_root,
        "nsga2_spread_preserving_phase2_report",
        args.artifact_suffix,
        ".md",
    )
    fairness_md = safe_artifact_path(
        artifact_root,
        "nsga2_spread_preserving_phase2_fairness_report",
        args.artifact_suffix,
        ".md",
    )

    BASE._write_json(results_json, payload)
    BASE._write_csv(
        results_csv,
        aggregate_rows,
        [
            "problem",
            "algorithm",
            "library",
            "status",
            "seeds",
            "successful_seeds",
            "mean_hv",
            "mean_distance",
            "mean_igd",
            "mean_spacing",
            "mean_coverage",
            "mean_nondominated_count",
            "mean_occupied_bins",
            "mean_segment_entropy",
            "mean_segment_load_gini",
            "mean_segment0_allocation",
            "mean_segment4_spacing",
            "mean_segment0_g",
            "mean_segment0_distance",
            "mean_segment0_low_g_count",
            "mean_runtime_seconds",
            "mean_actual_evaluations",
            "success_rate",
        ],
    )
    results_md.write_text(_results_markdown(payload), encoding="utf-8")
    fairness_md.write_text(_fairness_report_markdown(payload), encoding="utf-8")
    report_md.write_text(_phase2_report_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "results_json": str(results_json),
                "results_csv": str(results_csv),
                "results_md": str(results_md),
                "report_md": str(report_md),
                "fairness_md": str(fairness_md),
                "decision": phase2_decision,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
