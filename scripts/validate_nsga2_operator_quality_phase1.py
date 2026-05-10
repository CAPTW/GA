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
    candidate_d_uniform_crossover,
    candidate_h_uniform_dedup_mutation_boost,
    candidate_j_h_lite_retry2,
    candidate_l_sparse_parent_bias_light,
    candidate_m_boundary_preservation_light,
    candidate_n_low_g_tail_mutation_light,
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


BASE = _load_helper("validate_nsga2_candidate_suite.py", "_candidate_suite_operator_quality_phase1")
PHASE0 = _load_helper("validate_nsga2_operator_quality_phase0.py", "_operator_quality_phase0_helper")


METRIC_SPECS: dict[str, dict[str, Any]] = {
    "segment0_g_mean": {"higher_is_better": False, "row_key": "segment0_g_mean"},
    "segment0_distance_mean": {"higher_is_better": False, "row_key": "segment0_distance_mean"},
    "segment0_low_g_count": {"higher_is_better": True, "row_key": "segment0_low_g_count"},
    "segment0_nondominated_rate": {"higher_is_better": True, "row_key": "segment0_nondominated_rate"},
    "segment0_survival_rate": {"higher_is_better": True, "row_key": "segment0_survival_rate"},
    "hypervolume_2d": {"higher_is_better": True, "row_key": "hypervolume_2d"},
    "reference_front_distance": {"higher_is_better": False, "row_key": "reference_front_distance"},
    "inverted_generational_distance": {"higher_is_better": False, "row_key": "inverted_generational_distance"},
    "spacing": {"higher_is_better": False, "row_key": "spacing"},
    "nondominated_count": {"higher_is_better": True, "row_key": "nondominated_count"},
    "coverage_indicator": {"higher_is_better": True, "row_key": "coverage_indicator"},
    "runtime_seconds": {"higher_is_better": False, "row_key": "runtime_seconds"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 ZDT1 repeated validation for low-g operator-quality candidate_n."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problem", dest="problems", default="zdt1")
    parser.add_argument("--problems", dest="problems")
    parser.add_argument("--output-root", default="outputs/nsga2_operator_quality_phase1")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=20101)
    parser.add_argument("--budget", type=int, default=760)
    parser.add_argument(
        "--drift-audit-json",
        default="artifacts/nsga2_candidate_n_phase1_default_drift_audit_results.json",
        help="Optional drift-audit JSON included in the Phase 1 report if it exists.",
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
    return parser.parse_args()


def _candidate_variants() -> list[NSGA2CandidateVariant]:
    return [
        candidate_d_uniform_crossover(),
        candidate_h_uniform_dedup_mutation_boost(),
        candidate_j_h_lite_retry2(),
        candidate_l_sparse_parent_bias_light(),
        candidate_m_boundary_preservation_light(),
        candidate_n_low_g_tail_mutation_light(),
    ]


def _enable_phase1_diagnostics(config: GAConfig, *, candidate_id: str | None) -> GAConfig:
    clone = GAConfig.from_dict(config.to_dict())
    clone.algorithm_options = dict(clone.algorithm_options)
    clone.algorithm_options["nsga2_trace_enabled"] = True
    clone.algorithm_options["nsga2_operator_supply_trace_enabled"] = True
    clone.algorithm_options["nsga2_zdt1_component_trace_enabled"] = True
    clone.algorithm_options["nsga2_trace_generation_sample_stride"] = 1
    clone.algorithm_options["nsga2_trace_segment_count"] = 6
    if candidate_id:
        clone.algorithm_options["nsga2_trace_candidate_id"] = candidate_id
        clone.algorithm_options["nsga2_trace_run_id"] = f"{candidate_id}_{clone.problem}_phase1"
    else:
        clone.algorithm_options["nsga2_trace_run_id"] = f"internal_nsga2_{clone.problem}_phase1"
    return clone


def _candidate_result(
    config: GAConfig,
    variant: NSGA2CandidateVariant,
    *,
    seed: int,
    output_root: Path,
) -> ExternalMOComparatorResult:
    candidate_config = _enable_phase1_diagnostics(
        apply_candidate_variant(config, variant),
        candidate_id=variant.candidate_id,
    )
    result = run_internal_nsga2(candidate_config, seed=seed, output_root=str(output_root))
    metadata = dict(result.metadata)
    metadata.update(candidate_variant_metadata(variant))
    if variant.candidate_id == "candidate_n_low_g_tail_mutation_light":
        metadata["promotion_status"] = "phase1_validation"
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
    reference_front: list[list[float]],
    requested_budget: int,
    variant_map: dict[str, NSGA2CandidateVariant],
) -> dict[str, Any]:
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
    row = PHASE0._decorate_row(row, reference_front=reference_front)
    row = decorate_fairness_row(
        row,
        spec=spec,
        base_config=base_config,
        requested_budget=requested_budget,
        variant_map=variant_map,
    )
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
                "mean_runtime_seconds": BASE._summary_stat(successful, "runtime_seconds")["mean"],
                "mean_actual_evaluations": BASE._summary_stat(successful, "actual_evaluations")["mean"],
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


def _pick(
    paired_rows: list[dict[str, Any]],
    comparison: str,
    metric: str,
) -> dict[str, Any] | None:
    for row in paired_rows:
        if row["comparison"] == comparison and row["metric"] == metric:
            return row
    return None


def _metric_wins(row: dict[str, Any] | None) -> bool:
    return row is not None and int(row["win"]) > int(row["loss"])


def _metric_losses(row: dict[str, Any] | None) -> bool:
    return row is not None and int(row["win"]) < int(row["loss"])


def _zdt1_component_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return PHASE0._zdt1_component_rows(raw_rows)


def _operator_supply_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return PHASE0._operator_supply_rows(raw_rows)


def _candidate_gate_rows(
    rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    default_rows = [row for row in rows if row["algorithm"] == "internal_nsga2"]
    candidate_rows = [row for row in rows if row["algorithm"] == "candidate_n_low_g_tail_mutation_light"]

    metadata_ok = all(
        row.get("metadata", {}).get("candidate_id") == "candidate_n_low_g_tail_mutation_light"
        and row.get("metadata", {}).get("base_candidate_id") == "candidate_j_h_lite_retry2"
        and row.get("metadata", {}).get("mechanism") == "low_g_tail_mutation_light"
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
        for row in default_rows
    )
    evaluations_ok = all(
        row.get("requested_budget") == row.get("actual_evaluations")
        for row in candidate_rows
        if row.get("success")
    )
    zdt1_component_ok = all(
        bool(row.get("zdt1_component_diagnostics_success")) for row in candidate_rows if row.get("success")
    )
    non_finite_ok = all(bool(row.get("success")) for row in candidate_rows)
    fairness_fail_free = fairness_payload.get("summary_counts", {}).get("fail", 0) == 0
    catastrophic_regression = any(
        isinstance(row.get("hypervolume_2d"), int | float) and float(row["hypervolume_2d"]) <= 0.0
        for row in candidate_rows
        if row.get("success")
    )

    return [
        {
            "gate": "candidate metadata",
            "result": metadata_ok,
            "evidence": "candidate_n raw_rows metadata",
            "interpretation": "candidate_n metadata should preserve candidate_id/base_candidate/mechanism and remain opt-in only",
        },
        {
            "gate": "default_changed=false",
            "result": default_changed_ok,
            "evidence": "candidate_n raw_rows metadata",
            "interpretation": "candidate_n rows must keep default_changed=false",
        },
        {
            "gate": "candidate isolation",
            "result": candidate_isolation_ok,
            "evidence": "default internal rows + candidate rows",
            "interpretation": "default internal NSGA-II rows must remain candidate-metadata free",
        },
        {
            "gate": "fairness fail 없음",
            "result": fairness_fail_free,
            "evidence": f"fairness summary={fairness_payload.get('summary_counts', {})}",
            "interpretation": "Phase 1 promotion logic is blocked if any fairness fail appears",
        },
        {
            "gate": "actual evaluations",
            "result": evaluations_ok,
            "evidence": "requested_budget vs actual_evaluations",
            "interpretation": "candidate_n actual evaluations must match the requested budget",
        },
        {
            "gate": "ZDT1 component diagnostics",
            "result": zdt1_component_ok,
            "evidence": "candidate_n raw_rows diagnostics flags",
            "interpretation": "candidate_n rows must retain the required ZDT1 component summaries",
        },
        {
            "gate": "non-finite objective 없음",
            "result": non_finite_ok,
            "evidence": "candidate_n run success",
            "interpretation": "candidate_n must not trip non-finite fitness fail-fast checks",
        },
        {
            "gate": "catastrophic regression",
            "result": not catastrophic_regression,
            "evidence": "candidate_n hypervolume_2d",
            "interpretation": "no zero-or-negative hypervolume collapse should appear",
        },
    ]


def _phase1_signal_rows(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    component_g = _pick(paired_rows, "candidate_n vs candidate_j", "segment0_g_mean")
    component_distance = _pick(paired_rows, "candidate_n vs candidate_j", "segment0_distance_mean")
    component_low_g = _pick(paired_rows, "candidate_n vs candidate_j", "segment0_low_g_count")
    component_nd = _pick(paired_rows, "candidate_n vs candidate_j", "segment0_nondominated_rate")
    component_survival = _pick(paired_rows, "candidate_n vs candidate_j", "segment0_survival_rate")
    final_hv = _pick(paired_rows, "candidate_n vs candidate_j", "hypervolume_2d")
    final_distance = _pick(paired_rows, "candidate_n vs candidate_j", "reference_front_distance")
    final_igd = _pick(paired_rows, "candidate_n vs candidate_j", "inverted_generational_distance")
    final_spacing = _pick(paired_rows, "candidate_n vs candidate_j", "spacing")
    final_count = _pick(paired_rows, "candidate_n vs candidate_j", "nondominated_count")
    final_coverage = _pick(paired_rows, "candidate_n vs candidate_j", "coverage_indicator")

    component_positive = _metric_wins(component_g) or _metric_wins(component_distance)
    weak_component_positive = _metric_wins(component_nd) or _metric_wins(component_survival)
    low_g_conversion_positive = _metric_wins(component_low_g)
    final_regression_count = sum(
        _metric_losses(row)
        for row in (final_hv, final_distance, final_igd, final_spacing, final_count, final_coverage)
    )
    final_improvement_count = sum(
        _metric_wins(row)
        for row in (final_hv, final_distance, final_igd, final_spacing, final_count, final_coverage)
    )
    severe_final_regression = (
        final_regression_count >= 4
        or (
            _metric_losses(final_spacing)
            and _metric_losses(final_count)
            and (
                _metric_losses(final_hv)
                or _metric_losses(final_distance)
                or _metric_losses(final_igd)
            )
        )
    )

    if component_positive and not severe_final_regression and final_regression_count <= 1:
        interpretation = "segment0_g 또는 segment0_distance 개선이 반복 seed에서 유지되고 final front 회귀도 제한적이다"
    elif component_positive and not severe_final_regression:
        interpretation = "component signal은 남아 있지만 final front 품질 trade-off가 함께 나타난다"
    elif weak_component_positive and not severe_final_regression:
        interpretation = "g/distance 핵심 signal은 약하지만 segment0 nondominated/survival 지표에서만 약한 개선이 보인다"
    elif severe_final_regression:
        interpretation = "component signal도 약하고 final front regression이 커서 reject 쪽에 가깝다"
    else:
        interpretation = "Phase 0 약한 signal이 반복 seed에서 충분히 재현되지 않았고 candidate_j와 거의 유사하거나 혼합 신호다"

    return [
        {
            "problem": "zdt1",
            "component_positive": component_positive,
            "weak_component_positive": weak_component_positive,
            "low_g_conversion_positive": low_g_conversion_positive,
            "final_regression_count": final_regression_count,
            "final_improvement_count": final_improvement_count,
            "severe_final_regression": severe_final_regression,
            "interpretation": interpretation,
        }
    ]


def _phase1_decision(
    gate_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    signal_rows: list[dict[str, Any]],
) -> str:
    if fairness_payload.get("summary_counts", {}).get("fail", 0) > 0:
        return "Needs fairness rerun"
    if not all(bool(row["result"]) for row in gate_rows):
        return "Fix required"

    row = signal_rows[0]
    component_positive = bool(row["component_positive"])
    weak_component_positive = bool(row["weak_component_positive"])
    final_regression_count = int(row["final_regression_count"])
    severe_final_regression = bool(row["severe_final_regression"])

    if component_positive and not severe_final_regression and final_regression_count <= 1:
        return "Phase 1 passed, eligible for Phase 2 planning"
    if component_positive and not severe_final_regression:
        return "Phase 1 passed with trade-offs"
    if severe_final_regression and not component_positive:
        return "Reject"
    if weak_component_positive or final_regression_count <= 2:
        return "Hold for more evidence"
    return "Reject"


def _load_drift_payload(path: Path, selected_problems: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary_rows = [
        row
        for row in payload.get("summary_rows", [])
        if str(row.get("problem")) in set(selected_problems)
        and str(row.get("metric"))
        in {
            "hypervolume_2d",
            "reference_front_distance",
            "generational_distance",
            "inverted_generational_distance",
            "spacing",
            "nondominated_count",
            "actual_evaluations",
        }
    ]
    return {
        "overall": payload.get("overall", {}),
        "summary_rows": summary_rows,
    }


def _results_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# NSGA-II Operator Quality Phase 1 Results",
        "",
        "## Aggregate Summary",
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
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## ZDT1 Component Summary",
        "",
        *BASE._markdown_table(
            payload["zdt1_component_rows"],
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
        "## Paired Summary",
        "",
        *BASE._markdown_table(
            payload["paired_rows"],
            [
                "problem",
                "comparison",
                "metric",
                "win",
                "tie",
                "loss",
                "mean_delta",
                "median_delta",
                "comparable_seeds",
            ],
        ),
        "",
        "## Fairness Summary",
        "",
        *BASE._markdown_table(
            fairness_summary_rows(payload["fairness"]),
            ["status", "pass", "warning", "fail"],
        ),
        "",
        "## Decision",
        "",
        f"- `{payload['phase1_decision']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def _fairness_report_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# NSGA-II Operator Quality Phase 1 Fairness Report",
        "",
        "## Fairness Summary",
        "",
        *BASE._markdown_table(
            fairness_summary_rows(payload["fairness"]),
            ["status", "pass", "warning", "fail"],
        ),
        "",
        "## Fairness Issues",
        "",
        *BASE._markdown_table(
            payload["fairness"]["issues"],
            ["status", "issue_type", "algorithm", "problem", "message", "severity", "recommended_action"],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _phase1_report_markdown(payload: dict[str, Any]) -> str:
    drift_payload = payload.get("drift_audit")
    drift_rows = drift_payload.get("summary_rows", []) if isinstance(drift_payload, dict) else []
    drift_detected = (
        bool(drift_payload.get("overall", {}).get("drift_detected"))
        if isinstance(drift_payload, dict)
        else None
    )
    local_baseline_status = str(payload.get("local_baseline_status", "not_run"))
    candidate_definition = {
        "candidate_id": "candidate_n_low_g_tail_mutation_light",
        "base_candidate": "candidate_j_h_lite_retry2",
        "mechanism": "low_g_tail_mutation_light",
        "default_changed": False,
        "promotion_status": "phase1_validation",
        "allowed_use": "phase0_sanity_only",
        "disallowed_use": "default_replacement",
        "zdt1_specific_warning": "ZDT1-only evidence; broad MOEA superiority claim forbidden",
    }

    drift_table_rows = [
        {
            "gate": "default metadata contamination",
            "result": "pass" if drift_detected is False else "fail" if drift_detected else "n/a",
            "evidence": payload.get("drift_audit_path") or "n/a",
            "interpretation": "default internal path should remain free of candidate_n metadata",
        },
        {
            "gate": "diagnostics metadata contamination",
            "result": "pass"
            if isinstance(drift_payload, dict)
            and drift_payload.get("overall", {}).get("diagnostics_metadata_leak") is False
            else "fail"
            if isinstance(drift_payload, dict)
            and drift_payload.get("overall", {}).get("diagnostics_metadata_leak") is True
            else "n/a",
            "evidence": payload.get("drift_audit_path") or "n/a",
            "interpretation": "default internal path should remain free of diagnostics metadata unless explicitly enabled",
        },
        {
            "gate": "default_changed=false",
            "result": "pass"
            if any(row["gate"] == "default_changed=false" and row["result"] for row in payload["gate_rows"])
            else "fail",
            "evidence": "candidate_n metadata",
            "interpretation": "candidate_n must keep default_changed=false",
        },
        {
            "gate": "actual evaluations",
            "result": "pass"
            if any(row["gate"] == "actual evaluations" and row["result"] for row in payload["gate_rows"])
            else "fail",
            "evidence": "requested_budget vs actual_evaluations",
            "interpretation": "candidate_n actual evaluations should match the requested budget",
        },
        {
            "gate": "candidate isolation",
            "result": "pass"
            if any(row["gate"] == "candidate isolation" and row["result"] for row in payload["gate_rows"])
            else "fail",
            "evidence": "default internal rows + candidate rows",
            "interpretation": "candidate_n metadata should appear only on explicit opt-in runs",
        },
        {
            "gate": "local baseline governance",
            "result": local_baseline_status,
            "evidence": "python scripts/check_local_baseline.py ...",
            "interpretation": str(payload.get("local_baseline_note", "see regression check section")),
        },
    ]

    paired_focus = {
        "candidate_n vs candidate_j",
        "candidate_n vs pymoo",
        "candidate_n vs internal_nsga2",
        "candidate_n vs Random Pareto Archive",
        "candidate_n vs candidate_d",
        "candidate_n vs candidate_h",
        "candidate_n vs candidate_l",
        "candidate_n vs candidate_m",
    }

    lines: list[str] = [
        "# NSGA-II Operator Quality Phase 1 ZDT1 Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: `candidate_n_low_g_tail_mutation_light`의 Phase 0 약한 signal이 ZDT1 10-seed 반복 검증에서도 유지되는지 확인하는 것이다.",
        f"- default drift audit 결과: `{('DRIFT DETECTED' if drift_detected else 'NO DRIFT') if drift_detected is not None else 'not available'}`",
        f"- candidate isolation 결과: `{'pass' if any(row['gate'] == 'candidate isolation' and row['result'] for row in payload['gate_rows']) else 'fail'}`",
        f"- 실행한 benchmark: {', '.join(payload['selected_problems'])} / seeds={len(payload['seeds'])} / budget={payload['budget']}",
        f"- candidate_n vs candidate_j 핵심 결과: `{payload['phase1_decision']}` 이전에 component signal과 final front trade-off를 함께 확인했다.",
        "- candidate_n vs pymoo 핵심 결과: external gap이 남는지 paired comparison과 aggregate summary에 그대로 기록했다.",
        "- ZDT1 component signal: segment0_g_mean, segment0_distance_mean, segment0_low_g_count, segment0_nondominated_rate, segment0_survival_rate를 재검증했다.",
        f"- fairness 결과: pass={payload['fairness_summary'].get('pass', 0)}, warning={payload['fairness_summary'].get('warning', 0)}, fail={payload['fairness_summary'].get('fail', 0)}",
        f"- Phase 1 decision: **{payload['phase1_decision']}**",
        "- 기본값 변경 여부: 없음",
        "- Level 판정 변화 여부: Level 판정 유지 또는 실험 툴킷 근거 강화 범위에서만 해석한다.",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "- Scope: candidate_n_low_g_tail_mutation_light, ZDT1 10-seed repeated validation, candidate_n vs candidate_j, candidate_n vs pymoo, ZDT1 component diagnostics, operator supply diagnostics, fairness check, default drift check",
        "- Non-Scope: default promotion, candidate_n change request, approved opt-in profile, DTLZ/WFG validation, new operator candidate, survivor-pressure candidate, productization",
        "",
        "## 3. Default Drift and Candidate Isolation",
        "",
        *BASE._markdown_table(
            drift_table_rows,
            ["gate", "result", "evidence", "interpretation"],
        ),
        "",
        "## 4. Candidate Definition",
        "",
        *BASE._markdown_table(
            [{"field": key, "value": value} for key, value in candidate_definition.items()],
            ["field", "value"],
        ),
        "",
        "## 5. Experiment Configuration",
        "",
        *BASE._markdown_table(
            [
                {
                    "문제": row["problem"],
                    "알고리즘": row["algorithm"],
                    "seeds": row["seeds"],
                    "requested_budget": payload["budget"],
                    "actual_evaluations_summary": row["mean_actual_evaluations"],
                    "주요_설정": row["status"],
                }
                for row in payload["aggregate_rows"]
            ],
            ["문제", "알고리즘", "seeds", "requested_budget", "actual_evaluations_summary", "주요_설정"],
        ),
        "",
        "## 6. Results Summary",
        "",
        *BASE._markdown_table(
            payload["aggregate_rows"],
            [
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
        "## 7. ZDT1 Component Summary",
        "",
        *BASE._markdown_table(
            payload["zdt1_component_rows"],
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
        "## 8. Paired Comparisons",
        "",
        *BASE._markdown_table(
            [row for row in payload["paired_rows"] if row["comparison"] in paired_focus],
            [
                "comparison",
                "metric",
                "win",
                "tie",
                "loss",
                "mean_delta",
                "median_delta",
            ],
        ),
        "",
        "## 9. Candidate N Decision",
        "",
        f"- **{payload['phase1_decision']}**",
        "",
        "## 10. What We Learned",
        "",
        *[f"- {row['problem']}: {row['interpretation']}" for row in payload["signal_rows"]],
        "- low-g tail mutation이 segment0_g 또는 segment0_distance를 반복 seed 기준으로 개선했는지는 candidate_n vs candidate_j paired rows로 판단했다.",
        "- segment0_low_g_count가 계속 0이면 low-g conversion 자체는 아직 확증되지 않은 것으로 본다.",
        "- component signal이 final front 품질로 이어지지 않으면 Hold 또는 redesign 쪽으로 해석해야 한다.",
        "- final nondominated_count나 spacing regression이 반복되면 ZDT1-specific perturbation risk를 더 크게 봐야 한다.",
        "- candidate_n이 pymoo와의 gap을 줄였는지는 aggregate rows와 paired rows를 함께 봐야 하며, gap이 남으면 그대로 기록해야 한다.",
        "- Phase 2로 갈 근거는 fairness, isolation, drift gate뿐 아니라 component signal과 final front signal의 정렬까지 필요하다.",
        "",
        "## 11. Failures and Warnings",
        "",
        *BASE._markdown_table(
            payload["failures"]
            or [{"type": "none", "target": "none", "seed": None, "message": "none", "impact": "none", "action": "none"}],
            ["type", "target", "seed", "message", "impact", "action"],
        ),
        "",
        "## 12. Regression Check",
        "",
        *BASE._markdown_table(
            payload["regression_checks"],
            ["command", "result", "note"],
        ),
        "",
        "## 13. Maturity Impact",
        "",
        "- Level 판정 유지.",
        "- Phase 1은 candidate evidence이지 default algorithm maturity 상향 근거가 아니다.",
        "- candidate_n이 좋아 보여도 기본값이 바뀌지 않았으므로 default NSGA-II maturity 상향은 금지한다.",
        "- ZDT1-specific candidate이므로 범용 MOEA 성숙도 상향 근거로 사용하면 안 된다.",
        "- fairness / isolation / default drift gate가 유지되면 실험 툴킷으로서의 Level 4 근거는 강화될 수 있다.",
        "",
        "## 14. Recommended Next Work",
        "",
        "1. Phase 1 passed이면 Phase 2 planning을 작성하되 ZDT1-specific risk를 명시한다.",
        "2. Phase 1 passed with trade-offs이면 candidate_n 설계 조정 여부를 먼저 검토한다.",
        "3. Hold이면 external operator parity diagnostics로 회귀한다.",
        "4. Reject이면 low_g_tail_mutation_light family를 폐기하거나 redesign backlog로 회수한다.",
        "5. candidate_j opt-in documentation은 유지한다.",
        "6. survivor-pressure family는 계속 pause 상태로 둔다.",
        "7. fairness checker single-objective runner 확장을 검토한다.",
        "8. constrained multi-objective contract",
        "9. checkpoint/resume",
        "10. parallel evaluation",
        "",
        f"이번 Phase 1 결과, candidate_n_low_g_tail_mutation_light는 ZDT1에서 {payload['summary_sentence_component']} 신호를 보였고, candidate_j 대비 {payload['summary_sentence_final']}였으며, 최종 판정은 {payload['phase1_decision']}이다.",
        "",
    ]

    if drift_rows:
        lines.extend(
            [
                "## Drift Excerpt",
                "",
                *BASE._markdown_table(
                    drift_rows,
                    [
                        "problem",
                        "metric",
                        "previous_mean",
                        "current_mean",
                        "delta",
                        "drift_detected",
                        "interpretation",
                    ],
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _summary_sentence_from_payload(payload: dict[str, Any]) -> tuple[str, str]:
    g_row = _pick(payload["paired_rows"], "candidate_n vs candidate_j", "segment0_g_mean")
    d_row = _pick(payload["paired_rows"], "candidate_n vs candidate_j", "segment0_distance_mean")
    spacing_row = _pick(payload["paired_rows"], "candidate_n vs candidate_j", "spacing")
    count_row = _pick(payload["paired_rows"], "candidate_n vs candidate_j", "nondominated_count")
    hv_row = _pick(payload["paired_rows"], "candidate_n vs candidate_j", "hypervolume_2d")
    igd_row = _pick(payload["paired_rows"], "candidate_n vs candidate_j", "inverted_generational_distance")

    if _metric_wins(g_row) or _metric_wins(d_row):
        component_sentence = "약한 component 개선"
    elif _metric_losses(g_row) and _metric_losses(d_row):
        component_sentence = "component 악화"
    else:
        component_sentence = "혼합 또는 미약한 component"

    regressions = sum(_metric_losses(row) for row in (spacing_row, count_row, hv_row, igd_row))
    improvements = sum(_metric_wins(row) for row in (spacing_row, count_row, hv_row, igd_row))
    if regressions == 0 and improvements > 0:
        final_sentence = "final front에서도 제한적 개선"
    elif regressions > improvements:
        final_sentence = "final front에서는 trade-off 또는 회귀"
    else:
        final_sentence = "final front에서는 대체로 유사하거나 혼합"
    return component_sentence, final_sentence


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    selected_problems = [
        item.strip().lower() for item in str(args.problems or "zdt1").split(",") if item.strip()
    ]
    if any(problem != "zdt1" for problem in selected_problems):
        raise ValueError("This Phase 1 operator-quality validation pass is limited to zdt1 only.")

    base_config = load_config(PROJECT_ROOT / args.config)
    selected_specs = [mo_candidate_suite_specs()[name] for name in selected_problems]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    variants = _candidate_variants()
    variant_map = {variant.candidate_id: variant for variant in variants}

    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    comparison_specs = [
        ("candidate_n_low_g_tail_mutation_light", "candidate_j_h_lite_retry2", "candidate_n vs candidate_j"),
        ("candidate_n_low_g_tail_mutation_light", "pymoo_nsga2", "candidate_n vs pymoo"),
        ("candidate_n_low_g_tail_mutation_light", "internal_nsga2", "candidate_n vs internal_nsga2"),
        ("candidate_n_low_g_tail_mutation_light", "random_pareto_archive", "candidate_n vs Random Pareto Archive"),
        ("candidate_n_low_g_tail_mutation_light", "candidate_d_uniform_crossover", "candidate_n vs candidate_d"),
        ("candidate_n_low_g_tail_mutation_light", "candidate_h_uniform_dedup_mutation_boost", "candidate_n vs candidate_h"),
        ("candidate_n_low_g_tail_mutation_light", "candidate_l_sparse_parent_bias_light", "candidate_n vs candidate_l"),
        ("candidate_n_low_g_tail_mutation_light", "candidate_m_boundary_preservation_light", "candidate_n vs candidate_m"),
    ]

    for spec in selected_specs:
        config = build_problem_config(base_config, spec)
        config = PHASE0._retarget_budget(config, args.budget)
        reference_front = reference_front_for_spec(spec, point_count=201)
        reference_point = list(spec.hv_reference_point)
        problem_output_root = output_root / spec.problem
        problem_output_root.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            results: list[ExternalMOComparatorResult] = [
                run_internal_nsga2(
                    _enable_phase1_diagnostics(config, candidate_id=None),
                    seed=seed,
                    output_root=str(problem_output_root),
                ),
                *[
                    _candidate_result(config, variant, seed=seed, output_root=problem_output_root)
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
                    reference_front=reference_front,
                    requested_budget=args.budget,
                    variant_map=variant_map,
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
                            "action": "review comparator/runtime failure before any Phase 1 conclusion",
                        }
                    )

    fairness_payload = evaluate_parameter_fairness(
        raw_rows,
        benchmark_rows=benchmark_rows,
        candidate_rows=candidate_rows,
    )
    aggregate_rows = _aggregate_rows(raw_rows)
    zdt1_component_rows = _zdt1_component_rows(raw_rows)
    operator_supply_rows = _operator_supply_rows(raw_rows)
    paired_rows = _paired_rows(raw_rows, comparison_specs)
    gate_rows = _candidate_gate_rows(raw_rows, fairness_payload)
    signal_rows = _phase1_signal_rows(paired_rows)
    phase1_decision = _phase1_decision(gate_rows, fairness_payload, signal_rows)
    drift_audit_path = PROJECT_ROOT / args.drift_audit_json
    drift_payload = _load_drift_payload(drift_audit_path, selected_problems)
    component_sentence, final_sentence = _summary_sentence_from_payload(
        {
            "paired_rows": paired_rows,
        }
    )

    regression_checks = [
        {
            "command": "python scripts/audit_nsga2_default_drift.py --results-base nsga2_candidate_n_phase1_default_drift_audit_results --report-base nsga2_candidate_n_phase1_default_drift_audit_report --output-root outputs/nsga2_candidate_n_phase1_default_drift",
            "result": "see drift artifact",
            "note": str(drift_audit_path) if drift_payload is not None else "not provided",
        },
        {
            "command": "python scripts/check_local_baseline.py --output-dir artifacts/operator_quality_phase1_guard",
            "result": args.local_baseline_status,
            "note": args.local_baseline_note,
        },
        {
            "command": "python scripts/validate_nsga2_operator_quality_phase1.py --problem zdt1 --seeds 10 --budget 760 --artifact-suffix candidate_n_phase1_zdt1",
            "result": "success",
            "note": "current run",
        },
    ]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "selected_problems": selected_problems,
        "seeds": seeds,
        "budget": args.budget,
        "benchmark_rows": benchmark_rows,
        "candidate_rows": candidate_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "zdt1_component_rows": zdt1_component_rows,
        "operator_supply_rows": operator_supply_rows,
        "paired_rows": paired_rows,
        "gate_rows": gate_rows,
        "signal_rows": signal_rows,
        "phase1_decision": phase1_decision,
        "failures": failures,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
        "drift_audit_path": str(drift_audit_path) if drift_payload is not None else None,
        "drift_audit": drift_payload,
        "local_baseline_status": args.local_baseline_status,
        "local_baseline_note": args.local_baseline_note,
        "regression_checks": regression_checks,
        "summary_sentence_component": component_sentence,
        "summary_sentence_final": final_sentence,
    }

    results_json = safe_artifact_path(
        artifact_root,
        "nsga2_operator_quality_phase1_results",
        args.artifact_suffix,
        ".json",
    )
    results_csv = safe_artifact_path(
        artifact_root,
        "nsga2_operator_quality_phase1_results",
        args.artifact_suffix,
        ".csv",
    )
    results_md = safe_artifact_path(
        artifact_root,
        "nsga2_operator_quality_phase1_results",
        args.artifact_suffix,
        ".md",
    )
    report_md = safe_artifact_path(
        artifact_root,
        "nsga2_operator_quality_phase1_report",
        args.artifact_suffix,
        ".md",
    )
    fairness_md = safe_artifact_path(
        artifact_root,
        "nsga2_operator_quality_phase1_fairness_report",
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
            "mean_runtime_seconds",
            "mean_actual_evaluations",
            "success_rate",
        ],
    )
    results_md.write_text(_results_markdown(payload), encoding="utf-8")
    fairness_md.write_text(_fairness_report_markdown(payload), encoding="utf-8")
    report_md.write_text(_phase1_report_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "results_json": str(results_json),
                "results_csv": str(results_csv),
                "results_md": str(results_md),
                "report_md": str(report_md),
                "fairness_md": str(fairness_md),
                "decision": phase1_decision,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
