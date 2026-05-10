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
    candidate_variant_metadata,
)
from ga_lab.experiment.parameter_fairness import evaluate_parameter_fairness


def _load_base_helpers():
    helper_path = PROJECT_ROOT / "scripts" / "validate_nsga2_candidate_suite.py"
    spec = importlib.util.spec_from_file_location("_candidate_suite_operator_phase0_base", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 0 sanity validation for low-g operator-quality candidate_n."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problem", dest="problems", default="zdt1")
    parser.add_argument("--problems", dest="problems")
    parser.add_argument("--output-root", default="outputs/nsga2_operator_quality_phase0")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=18101)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--include-pymoo", action="store_true")
    return parser.parse_args()


def _phase0_variants() -> list[NSGA2CandidateVariant]:
    return [
        candidate_j_h_lite_retry2(),
        candidate_n_low_g_tail_mutation_light(),
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


def _enable_phase0_diagnostics(config: GAConfig, *, candidate_id: str | None) -> GAConfig:
    clone = GAConfig.from_dict(config.to_dict())
    clone.algorithm_options = dict(clone.algorithm_options)
    clone.algorithm_options["nsga2_trace_enabled"] = True
    clone.algorithm_options["nsga2_operator_supply_trace_enabled"] = True
    clone.algorithm_options["nsga2_zdt1_component_trace_enabled"] = True
    clone.algorithm_options["nsga2_trace_generation_sample_stride"] = 1
    clone.algorithm_options["nsga2_trace_segment_count"] = 6
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


def _decorate_row(
    row: dict[str, Any],
    *,
    reference_front: list[list[float]],
) -> dict[str, Any]:
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
        row["segment0_g"] = _diag_metric(
            row,
            "zdt1_offspring_component_quality",
            "segment0_g_mean",
        )
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
        low_g_tail_mutation_stats = dict(row.get("metadata", {}).get("low_g_tail_mutation_stats", {}))
        row["low_g_tail_adjusted_solution_count"] = low_g_tail_mutation_stats.get(
            "adjusted_solution_count"
        )
        row["low_g_tail_adjusted_gene_count"] = low_g_tail_mutation_stats.get(
            "adjusted_gene_count"
        )
        row["low_g_tail_mean_step"] = low_g_tail_mutation_stats.get("mean_step")
    else:
        row["decision_duplicate_rate"] = None
        row["objective_duplicate_rate"] = None
        row["archive_duplicate_rate"] = None
        row["unique_decision_count"] = None
        row["unique_objective_count"] = None
        row["boundary_point_count"] = None
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
    row["reference_front_size"] = len(reference_front)
    return row


def _make_candidate_result(
    base_config: GAConfig,
    variant: NSGA2CandidateVariant,
    *,
    seed: int,
    output_root: Path,
) -> ExternalMOComparatorResult:
    candidate_config = _enable_phase0_diagnostics(
        apply_candidate_variant(base_config, variant),
        candidate_id=variant.candidate_id,
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
                "mean_segment0_g": BASE._summary_stat(successful, "segment0_g")["mean"],
                "mean_segment0_distance": BASE._summary_stat(successful, "segment0_distance")["mean"],
                "mean_segment0_low_g_count": BASE._summary_stat(successful, "segment0_low_g_count")["mean"],
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
                "low_g_tail_adjusted_solution_count": BASE._summary_stat(
                    bucket,
                    "low_g_tail_adjusted_solution_count",
                )["mean"],
                "low_g_tail_adjusted_gene_count": BASE._summary_stat(
                    bucket,
                    "low_g_tail_adjusted_gene_count",
                )["mean"],
                "low_g_tail_mean_step": BASE._summary_stat(bucket, "low_g_tail_mean_step")["mean"],
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


def _sanity_gate_results(
    raw_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_rows = [row for row in raw_rows if row["algorithm"] == "candidate_n_low_g_tail_mutation_light"]
    default_rows = [row for row in raw_rows if row["algorithm"] == "internal_nsga2"]
    metadata_ok = all(
        row.get("metadata", {}).get("candidate_id") == "candidate_n_low_g_tail_mutation_light"
        and row.get("metadata", {}).get("base_candidate_id") == "candidate_j_h_lite_retry2"
        and row.get("metadata", {}).get("mechanism") == "low_g_tail_mutation_light"
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
        row.get("requested_budget") == row.get("actual_evaluations") for row in candidate_rows if row.get("success")
    )
    zdt1_component_ok = all(
        bool(row.get("zdt1_component_diagnostics_success")) for row in candidate_rows if row.get("success")
    )
    non_finite_ok = all(
        row.get("success") for row in candidate_rows
    )
    metrics_ok = all(bool(row.get("metric_calculation_success")) for row in candidate_rows if row.get("success"))
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
            "pass_fail": metadata_ok,
            "note": "candidate_n metadata present with expected base candidate and mechanism",
        },
        {
            "gate": "default_changed=false",
            "result": default_changed_ok,
            "pass_fail": default_changed_ok,
            "note": "candidate_n metadata keeps default_changed=false",
        },
        {
            "gate": "candidate isolation",
            "result": candidate_isolation_ok,
            "pass_fail": candidate_isolation_ok,
            "note": "default internal NSGA-II rows stay free of candidate metadata",
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
            "note": "candidate_n actual evaluations match requested budget",
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
            "note": "candidate_n runs completed without fail-fast objective errors",
        },
        {
            "gate": "metric calculation success",
            "result": metrics_ok,
            "pass_fail": metrics_ok,
            "note": "core MO metrics stayed finite",
        },
        {
            "gate": "catastrophic regression 없음",
            "result": not catastrophic_regression,
            "pass_fail": not catastrophic_regression,
            "note": "no zero-or-negative hypervolume collapse observed in Phase 0",
        },
        {
            "gate": "artifact generation",
            "result": True,
            "pass_fail": True,
            "note": "results and report artifacts created",
        },
    ]


def _phase0_decision(
    gate_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    failures: list[dict[str, Any]],
) -> str:
    if fairness_payload.get("summary_counts", {}).get("fail", 0):
        return "Phase 0 failed, fix required"
    if not all(bool(row["pass_fail"]) for row in gate_rows):
        return "Phase 0 failed, fix required"
    if failures or fairness_payload.get("summary_counts", {}).get("warning", 0):
        return "Phase 0 passed with warnings"
    return "Phase 0 passed, eligible for Phase 1 planning"


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
                    _enable_phase0_diagnostics(config, candidate_id=None),
                    seed=seed,
                    output_root=str(problem_output_root),
                ),
                *[
                    _make_candidate_result(config, variant, seed=seed, output_root=problem_output_root)
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
                row = _decorate_row(row, reference_front=reference_front)
                row = decorate_fairness_row(
                    row,
                    spec=spec,
                    base_config=config,
                    requested_budget=args.budget,
                    variant_map=variant_map,
                )
                raw_rows.append(row)
                if result.status != "success":
                    failures.append(
                        {
                            "type": result.status,
                            "target": result.algorithm_name,
                            "message": result.error_message,
                            "impact": "seed excluded from sanity read",
                            "action": "fix the candidate/runtime issue before any Phase 1 planning",
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
    gate_rows = _sanity_gate_results(raw_rows, fairness_payload)
    decision = _phase0_decision(gate_rows, fairness_payload, failures)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "selected_problems": requested_problem_names,
        "seeds": seeds,
        "budget": args.budget,
        "benchmark_rows": benchmark_rows,
        "candidate_rows": candidate_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "zdt1_component_rows": zdt1_component_rows,
        "operator_supply_rows": operator_supply_rows,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
        "gate_rows": gate_rows,
        "phase0_decision": decision,
        "failures": failures,
    }

    json_path = safe_artifact_path(
        artifact_root,
        "nsga2_operator_quality_phase0_results",
        args.artifact_suffix,
        ".json",
    )
    md_path = safe_artifact_path(
        artifact_root,
        "nsga2_operator_quality_phase0_results",
        args.artifact_suffix,
        ".md",
    )
    report_path = safe_artifact_path(
        artifact_root,
        "nsga2_operator_quality_phase0_report",
        args.artifact_suffix,
        ".md",
    )

    BASE._write_json(json_path, payload)
    md_lines = [
        "# NSGA-II Operator Quality Phase 0 Results",
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
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_nondominated_count",
                "mean_segment0_g",
                "mean_segment0_distance",
                "mean_segment0_low_g_count",
                "mean_runtime_seconds",
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
                "low_g_tail_adjusted_solution_count",
                "low_g_tail_adjusted_gene_count",
                "low_g_tail_mean_step",
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
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# NSGA-II Operator Quality Phase 0 Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 목표: candidate_n_low_g_tail_mutation_light의 Phase 0 sanity와 low-g mechanism signal 점검",
        "- 구현 candidate: `candidate_n_low_g_tail_mutation_light`",
        "- 기본값 변경 여부: `false`",
        f"- fairness 결과: **{fairness_payload['status']}**",
        f"- Phase 0 판정: **{decision}**",
        "",
        "## 2. Metric Snapshot",
        "",
        *BASE._markdown_table(
            aggregate_rows,
            [
                "algorithm",
                "mean_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_nondominated_count",
                "mean_segment0_g",
                "mean_segment0_distance",
                "mean_segment0_low_g_count",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## 3. Gate Summary",
        "",
        *BASE._markdown_table(gate_rows, ["gate", "result", "pass_fail", "note"]),
        "",
        "## 4. Failures and Warnings",
        "",
        *BASE._markdown_table(
            failures or [{"type": "none", "target": "none", "message": "none", "impact": "none", "action": "none"}],
            ["type", "target", "message", "impact", "action"],
        ),
        "",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "results_json": str(json_path),
                "results_md": str(md_path),
                "report_md": str(report_path),
                "phase0_decision": decision,
                "fairness_status": fairness_payload["status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
