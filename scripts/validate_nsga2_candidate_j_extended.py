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

from ga_lab.config import load_config
from ga_lab.experiment.diversity_diagnostics import evaluate_diversity_diagnostics
from ga_lab.experiment.external_mo_comparators import (
    ExternalMOComparatorResult,
    result_to_front_row,
    run_deap_nsga2,
    run_internal_nsga2,
    run_pymoo_nsga2,
    run_random_archive_anchor,
)
from ga_lab.experiment.mo_baselines import run_random_pareto_archive
from ga_lab.experiment.mo_metrics import coverage_indicator
from ga_lab.experiment.nsga2_candidate_suite import (
    MOBenchmarkSpec,
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
    candidate_variant_metadata,
)
from ga_lab.experiment.parameter_fairness import (
    METRIC_POSTPROCESSING_ID,
    evaluate_parameter_fairness,
)


def _load_base_helpers():
    helper_path = PROJECT_ROOT / "scripts" / "validate_nsga2_candidate_suite.py"
    spec = importlib.util.spec_from_file_location("_candidate_suite_base_j_ext", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_helpers()

METRIC_SPECS: dict[str, dict[str, bool]] = {
    "hypervolume_2d": {"higher_is_better": True},
    "reference_front_distance": {"higher_is_better": False},
    "generational_distance": {"higher_is_better": False},
    "inverted_generational_distance": {"higher_is_better": False},
    "spacing": {"higher_is_better": False},
    "nondominated_count": {"higher_is_better": True},
    "coverage_indicator": {"higher_is_better": True},
    "archive_duplicate_rate": {"higher_is_better": False},
    "objective_duplicate_rate": {"higher_is_better": False},
    "decision_duplicate_rate": {"higher_is_better": False},
    "runtime_seconds": {"higher_is_better": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extended validation and fairness automation for candidate_j_h_lite_retry2."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problems", default="zdt1,zdt2,zdt3,dtlz2,dtlz3,dtlz4")
    parser.add_argument("--output-root", default="outputs/nsga2_candidate_j_extended_validation")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=9401)
    parser.add_argument("--budget", type=int, default=760)
    return parser.parse_args()


def _candidate_variants() -> list[NSGA2CandidateVariant]:
    return [
        candidate_d_uniform_crossover(),
        candidate_h_uniform_dedup_mutation_boost(),
        candidate_j_h_lite_retry2(),
    ]


def _artifact_base_names(problem_names: list[str]) -> dict[str, str]:
    wfg_only = bool(problem_names) and all(name.startswith("wfg") for name in problem_names)
    if wfg_only:
        return {
            "results": "nsga2_candidate_j_wfg_results",
            "report": "nsga2_candidate_j_wfg_report",
            "fairness": "nsga2_candidate_j_wfg_fairness_report",
        }
    return {
        "results": "nsga2_candidate_j_extended_results",
        "report": "nsga2_candidate_j_extended_report",
        "fairness": "nsga2_candidate_j_fairness_report",
    }


def _algorithm_contract(
    *,
    base_config,
    spec: MOBenchmarkSpec,
    algorithm: str,
    variant: NSGA2CandidateVariant | None = None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "objective_count": spec.objectives,
        "variable_count": spec.variables,
        "bounds": list(spec.bounds),
        "reference_front_source": spec.reference_front_name,
        "hypervolume_reference_point": list(spec.hv_reference_point),
        "metric_postprocessing": METRIC_POSTPROCESSING_ID,
        "population_size": base_config.population_size,
        "configured_generations": base_config.generations,
        "repair_bounds": "representation_bounds",
    }
    if algorithm == "internal_nsga2":
        return contract | {
            "algorithm_family": "internal_nsga2",
            "operator_family": "internal_nsga2_arithmetic_gaussian",
            "crossover_type": base_config.crossover,
            "mutation_type": base_config.mutation,
            "mutation_probability": base_config.mutation_rate,
            "duplicate_handling": "none",
        }
    if variant is not None:
        duplicate_handling = "none"
        if variant.candidate_id == "candidate_h_uniform_dedup_mutation_boost":
            duplicate_handling = "decision_dedup + retry2 + novelty_survival"
        elif variant.candidate_id == "candidate_j_h_lite_retry2":
            duplicate_handling = "decision_dedup + retry2_lite"
        return contract | candidate_variant_metadata(variant) | {
            "algorithm_family": "internal_nsga2_candidate",
            "operator_family": variant.candidate_id,
            "crossover_type": "uniform",
            "mutation_type": base_config.mutation,
            "mutation_probability": base_config.mutation_rate,
            "duplicate_handling": duplicate_handling,
        }
    if algorithm == "pymoo_nsga2":
        mutation_probability = (
            base_config.mutation_rate
            if base_config.mutation_rate > 0
            else 1.0 / max(1, spec.variables)
        )
        return contract | {
            "algorithm_family": "external_nsga2",
            "operator_family": "pymoo_standard_sbx_pm",
            "crossover_type": "sbx",
            "mutation_type": "polynomial",
            "mutation_probability": mutation_probability,
            "duplicate_handling": "none",
        }
    if algorithm == "deap_nsga2":
        return contract | {
            "algorithm_family": "external_nsga2",
            "operator_family": "deap_selNSGA2_sbx_poly",
            "crossover_type": "sbx",
            "mutation_type": "polynomial",
            "mutation_probability": base_config.mutation_rate,
            "duplicate_handling": "none",
        }
    if algorithm == "random_pareto_archive":
        return contract | {
            "algorithm_family": "random_archive_baseline",
            "operator_family": "random_archive_baseline",
            "population_size": None,
            "configured_generations": None,
            "crossover_type": None,
            "mutation_type": None,
            "mutation_probability": None,
            "duplicate_handling": "archive_nondominated_only",
        }
    return contract | {
        "algorithm_family": "unknown",
        "operator_family": "unknown",
        "crossover_type": None,
        "mutation_type": None,
        "mutation_probability": None,
        "duplicate_handling": "unknown",
    }


def _decorate_row(
    row: dict[str, Any],
    *,
    spec: MOBenchmarkSpec,
    base_config,
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
        decision_vectors = row.get("front_decision_vectors") or row.get("decision_vectors")
        row.update(
            evaluate_diversity_diagnostics(
                row.get("objective_vectors", []),
                directions=directions,
                decision_vectors=decision_vectors if isinstance(decision_vectors, list) else None,
            )
        )
    else:
        row["reference_front_coverage"] = None
        row.update(
            {
                "decision_duplicate_rate": None,
                "objective_duplicate_rate": None,
                "archive_duplicate_rate": None,
                "unique_decision_count": None,
                "unique_objective_count": None,
                "boundary_point_count": None,
            }
        )
    algorithm = str(row["algorithm"])
    variant = variant_map.get(algorithm)
    metadata = dict(row.get("metadata", {}))
    metadata.update(
        _algorithm_contract(
            base_config=base_config,
            spec=spec,
            algorithm=algorithm,
            variant=variant,
        )
    )
    row["metadata"] = metadata
    row["metric_postprocessing"] = METRIC_POSTPROCESSING_ID
    row["reference_front_source"] = spec.reference_front_name
    row["hv_reference_point"] = list(spec.hv_reference_point)
    row["problem_objectives"] = spec.objectives
    row["problem_variables"] = spec.variables
    row["problem_bounds"] = list(spec.bounds)
    row["requested_budget"] = requested_budget
    row["benchmark_notes"] = spec.notes
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
                "mean_gd": BASE._summary_stat(successful, "generational_distance")["mean"],
                "mean_igd": BASE._summary_stat(successful, "inverted_generational_distance")["mean"],
                "mean_spacing": BASE._summary_stat(successful, "spacing")["mean"],
                "mean_coverage": BASE._summary_stat(successful, "reference_front_coverage")["mean"],
                "mean_nondominated_count": BASE._summary_stat(successful, "nondominated_count")["mean"],
                "mean_duplicate_rate": BASE._summary_stat(successful, "archive_duplicate_rate")["mean"],
                "mean_archive_duplicate_rate": BASE._summary_stat(successful, "archive_duplicate_rate")["mean"],
                "mean_objective_duplicate_rate": BASE._summary_stat(successful, "objective_duplicate_rate")["mean"],
                "mean_decision_duplicate_rate": BASE._summary_stat(successful, "decision_duplicate_rate")["mean"],
                "mean_unique_decision_count": BASE._summary_stat(successful, "unique_decision_count")["mean"],
                "mean_unique_objective_count": BASE._summary_stat(successful, "unique_objective_count")["mean"],
                "mean_boundary_point_count": BASE._summary_stat(successful, "boundary_point_count")["mean"],
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
            left_metric = left.get(metric_name)
            right_metric = right.get(metric_name)
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
                summary = _paired_metric_summary(left_rows, right_rows, metric_name, metric_spec["higher_is_better"])
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


def _candidate_result(config, variant, *, seed: int, output_root: Path) -> ExternalMOComparatorResult:
    candidate_config = apply_candidate_variant(config, variant)
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


def _pick(
    paired_rows: list[dict[str, Any]],
    problem: str,
    left: str,
    right: str,
    metric: str,
) -> dict[str, Any] | None:
    for row in paired_rows:
        if (
            row["problem"] == problem
            and row["left_algorithm"] == left
            and row["right_algorithm"] == right
            and row["metric"] == metric
        ):
            return row
    return None


def _problem_read(
    paired_rows: list[dict[str, Any]],
    problem: str,
) -> tuple[str, str, str]:
    core_metrics = [
        "hypervolume_2d",
        "reference_front_distance",
        "generational_distance",
        "inverted_generational_distance",
        "coverage_indicator",
    ]
    diversity_metrics = [
        "spacing",
        "nondominated_count",
        "archive_duplicate_rate",
        "decision_duplicate_rate",
    ]

    def _score(left: str, right: str, metrics: list[str]) -> tuple[int, int]:
        wins = 0
        losses = 0
        for metric in metrics:
            row = _pick(paired_rows, problem, left, right, metric)
            if row is None:
                continue
            if int(row["win"]) > int(row["loss"]):
                wins += 1
            elif int(row["win"]) < int(row["loss"]):
                losses += 1
        return wins, losses

    d_core_wins, d_core_losses = _score("candidate_j_h_lite_retry2", "candidate_d_uniform_crossover", core_metrics)
    d_div_wins, _ = _score("candidate_j_h_lite_retry2", "candidate_d_uniform_crossover", diversity_metrics)
    h_core_wins, h_core_losses = _score("candidate_j_h_lite_retry2", "candidate_h_uniform_dedup_mutation_boost", core_metrics)
    p_core_wins, p_core_losses = _score("candidate_j_h_lite_retry2", "pymoo_nsga2", core_metrics)
    deap_core_wins, deap_core_losses = _score("candidate_j_h_lite_retry2", "deap_nsga2", core_metrics)

    if d_div_wins >= 2 and d_core_wins >= d_core_losses:
        versus_d = "candidate_d 대비 diversity 개선과 convergence 유지"
    elif d_core_wins > d_core_losses:
        versus_d = "candidate_d 대비 convergence 쪽 우세, diversity 개선은 제한적"
    else:
        versus_d = "candidate_d 대비 뚜렷한 개선 부족"

    if h_core_wins >= 2 and h_core_losses <= 1:
        versus_h = "candidate_h 대비 convergence trade-off 완화"
    elif h_core_wins >= h_core_losses:
        versus_h = "candidate_h 대비 혼합이지만 더 안전한 신호"
    else:
        versus_h = "candidate_h 대비 trade-off 완화 근거 부족"

    if p_core_wins >= 2 or deap_core_wins >= 2:
        external = "external comparator gap 일부 축소"
    elif p_core_losses >= 3 and deap_core_losses >= 3:
        external = "external comparator 대비 핵심 metric 열세 유지"
    else:
        external = "external comparator 대비 혼합"
    return versus_d, versus_h, external


def _candidate_decision(
    paired_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    executed_problems: list[str],
    not_executed_benchmarks: list[dict[str, str]],
) -> dict[str, Any]:
    problem_rows: list[dict[str, Any]] = []
    zdt_positive = 0
    dtlz_positive = 0
    external_gap_closure = 0
    wfg_only = bool(executed_problems) and all(problem.startswith("wfg") for problem in executed_problems)

    for problem in executed_problems:
        versus_d, versus_h, external = _problem_read(paired_rows, problem)
        if "diversity 개선" in versus_d or "convergence 쪽 우세" in versus_d:
            if problem.startswith("zdt"):
                zdt_positive += 1
            if problem.startswith("dtlz"):
                dtlz_positive += 1
        if "gap 일부 축소" in external:
            external_gap_closure += 1
        problem_rows.append(
            {
                "candidate": "candidate_j_h_lite_retry2",
                "problem": problem,
                "candidate_d_vs": versus_d,
                "candidate_h_vs": versus_h,
                "external_vs": external,
            }
        )

    fairness_status = str(fairness_payload["status"])
    if fairness_status == "fail":
        final = "Needs parameter fairness rerun"
    elif wfg_only:
        final = "Hold for more benchmarks"
    elif zdt_positive >= 2 and dtlz_positive >= 2 and external_gap_closure >= 2 and not not_executed_benchmarks:
        final = "Promote to change request"
    elif zdt_positive >= 2 and dtlz_positive >= 1 and external_gap_closure >= 1:
        final = "Approved for opt-in experimental profile"
    elif zdt_positive >= 1:
        final = "Hold for more benchmarks"
    else:
        final = "Reject"

    return {
        "problem_rows": problem_rows,
        "fairness_status": fairness_status,
        "final_decision": final,
    }


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(PROJECT_ROOT / args.config)
    problem_names = [item.strip().lower() for item in str(args.problems).split(",") if item.strip()]
    suite_specs = mo_candidate_suite_specs()
    selected_specs = [suite_specs[name] for name in problem_names]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    variants = _candidate_variants()
    variant_map = {variant.candidate_id: variant for variant in variants}

    benchmark_rows = [
        {
            "problem": spec.problem,
            "source": "internal_problem_registry",
            "objectives": spec.objectives,
            "variables": spec.variables,
            "bounds": list(spec.bounds),
            "reference_front": spec.reference_front_name,
            "hv_reference_point": list(spec.hv_reference_point),
            "metric_limitations": (
                "pymoo-backed smoke reference front; distance/GD/IGD remain limited"
                if spec.problem.startswith("wfg")
                else "hypervolume_2d only for 2-objective problems"
            ),
        }
        for spec in selected_specs
    ]
    candidate_rows = [candidate_variant_metadata(variant) for variant in variants]
    requested_problem_names = {spec.problem for spec in selected_specs}
    not_executed_benchmarks = [
        {
            "problem": problem_name,
            "reason": "benchmark was not selected in this run",
        }
        for problem_name in ("wfg1", "wfg2")
        if problem_name not in requested_problem_names
    ]

    comparison_specs = [
        ("candidate_j_h_lite_retry2", "candidate_d_uniform_crossover", "candidate_j vs candidate_d"),
        ("candidate_j_h_lite_retry2", "candidate_h_uniform_dedup_mutation_boost", "candidate_j vs candidate_h"),
        ("candidate_j_h_lite_retry2", "internal_nsga2", "candidate_j vs internal baseline"),
        ("candidate_j_h_lite_retry2", "pymoo_nsga2", "candidate_j vs pymoo"),
        ("candidate_j_h_lite_retry2", "deap_nsga2", "candidate_j vs deap"),
        ("candidate_j_h_lite_retry2", "random_pareto_archive", "candidate_j vs random archive"),
        ("candidate_h_uniform_dedup_mutation_boost", "pymoo_nsga2", "candidate_h vs pymoo"),
        ("candidate_d_uniform_crossover", "pymoo_nsga2", "candidate_d vs pymoo"),
    ]

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in selected_specs:
        config = build_problem_config(base_config, spec)
        try:
            reference_front = reference_front_for_spec(spec, point_count=201)
        except ImportError as exc:
            not_executed_benchmarks.append(
                {
                    "problem": spec.problem,
                    "reason": str(exc),
                }
            )
            failures.append(
                {
                    "type": "skipped_problem",
                    "target": spec.problem,
                    "problem": spec.problem,
                    "seed": -1,
                    "message": str(exc),
                    "impact": "problem excluded from candidate_j extended validation",
                    "action": "install pymoo and rerun the selected WFG benchmark",
                }
            )
            continue
        reference_point = list(spec.hv_reference_point)
        problem_output_root = output_root / spec.problem
        problem_output_root.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            results = [
                run_internal_nsga2(config, seed=seed, output_root=str(problem_output_root)),
                *[_candidate_result(config, variant, seed=seed, output_root=problem_output_root) for variant in variants],
                run_pymoo_nsga2(config, seed=seed, budget=args.budget),
                run_deap_nsga2(config, seed=seed, budget=args.budget),
                run_random_archive_anchor(run_random_pareto_archive(config, seed=seed, budget=args.budget)),
            ]
            for result in results:
                row = result_to_front_row(result, reference_front=reference_front, reference_point=reference_point)
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
                            "problem": spec.problem,
                            "seed": seed,
                            "message": result.error_message,
                            "impact": "seed excluded from paired comparison",
                            "action": "review comparator/runtime failure before candidate_j promotion",
                        }
                    )

    fairness_payload = evaluate_parameter_fairness(
        raw_rows,
        benchmark_rows=benchmark_rows,
        candidate_rows=candidate_rows,
    )
    aggregate_rows = _aggregate_rows(raw_rows)
    paired_rows = _paired_rows(raw_rows, comparison_specs)
    decision_payload = _candidate_decision(
        paired_rows,
        fairness_payload,
        [spec.problem for spec in selected_specs],
        not_executed_benchmarks,
    )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "seeds": seeds,
        "budget": args.budget,
        "selected_problems": problem_names,
        "benchmark_rows": benchmark_rows,
        "candidate_rows": candidate_rows,
        "not_executed_benchmarks": not_executed_benchmarks,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "fairness": fairness_payload,
        "decision": decision_payload,
        "failures": failures,
    }

    artifact_bases = _artifact_base_names(problem_names)
    json_path = safe_artifact_path(artifact_root, artifact_bases["results"], args.artifact_suffix, ".json")
    csv_path = safe_artifact_path(artifact_root, artifact_bases["results"], args.artifact_suffix, ".csv")
    md_path = safe_artifact_path(artifact_root, artifact_bases["results"], args.artifact_suffix, ".md")
    report_path = safe_artifact_path(artifact_root, artifact_bases["report"], args.artifact_suffix, ".md")
    fairness_report_path = safe_artifact_path(artifact_root, artifact_bases["fairness"], args.artifact_suffix, ".md")

    BASE._write_json(json_path, payload)
    BASE._write_csv(
        csv_path,
        aggregate_rows,
        [
            "problem", "algorithm", "library", "status", "seeds", "successful_seeds", "mean_hv",
            "mean_distance", "mean_gd", "mean_igd", "mean_spacing", "mean_coverage",
            "mean_nondominated_count", "mean_duplicate_rate", "mean_archive_duplicate_rate",
            "mean_objective_duplicate_rate", "mean_decision_duplicate_rate", "mean_unique_decision_count",
            "mean_unique_objective_count", "mean_boundary_point_count", "mean_runtime_seconds",
            "mean_actual_evaluations", "success_rate",
        ],
    )

    wfg_only = bool(problem_names) and all(name.startswith("wfg") for name in problem_names)

    md_lines = [
        "# NSGA-II Candidate J WFG Results" if wfg_only else "# NSGA-II Candidate J Extended Results",
        "",
        "## Aggregate Results",
        "",
        *BASE._markdown_table(
            aggregate_rows,
            ["problem", "algorithm", "mean_hv", "mean_distance", "mean_igd", "mean_spacing", "mean_nondominated_count", "mean_duplicate_rate", "mean_runtime_seconds"],
        ),
        "",
        "## Candidate Decision",
        "",
        *BASE._markdown_table(
            decision_payload["problem_rows"],
            ["candidate", "problem", "candidate_d_vs", "candidate_h_vs", "external_vs"],
        ),
        "",
        f"Final decision: **{decision_payload['final_decision']}**",
        "",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# NSGA-II Candidate J WFG Smoke Report"
        if wfg_only
        else "# NSGA-II Candidate J Extended Validation Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: candidate_j_h_lite_retry2 generalization과 parameter fairness 자동 점검.",
        f"- candidate_j 현재 상태: `{candidate_rows[-1]['promotion_status']}` / default_changed={candidate_rows[-1]['default_changed']}",
        f"- 실행 benchmark: {', '.join(problem_names)}",
        f"- fairness checker 결과: **{fairness_payload['status']}**",
        f"- candidate_j 최종 판정: **{decision_payload['final_decision']}**",
        "- 기본 NSGA-II default는 변경하지 않았다.",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "- Scope: candidate_j extended validation, parameter fairness automation, ZDT/DTLZ small benchmark, external comparator comparison.",
        "- Non-Scope: default promotion, 새 후보 설계, WFG full suite, constrained MOO, checkpoint/resume, parallel evaluation, productization.",
        "",
        "## 3. Parameter Fairness Report",
        "",
        *BASE._markdown_table(
            fairness_payload["issues"],
            ["status", "issue_type", "problem", "algorithm", "message", "severity"],
        ),
        "",
        "## 4. Benchmark Definitions",
        "",
        *BASE._markdown_table(
            benchmark_rows,
            ["problem", "source", "objectives", "variables", "bounds", "reference_front", "metric_limitations"],
        ),
        "",
        "WFG benchmark notes:" if wfg_only else "WFG benchmark status:",
        "",
        *BASE._markdown_table(not_executed_benchmarks, ["problem", "reason"]),
        "",
        "## 5. Candidate Context",
        "",
        *BASE._markdown_table(
            [
                {
                    "candidate": "candidate_d_uniform_crossover",
                    "status": "approved_opt_in",
                    "strength": "ZDT HV/distance/coverage improvement",
                    "weakness": "spacing/nondominated_count weakness vs external",
                    "current_decision": "Approved for opt-in experimental profile",
                },
                {
                    "candidate": "candidate_h_uniform_dedup_mutation_boost",
                    "status": "hold",
                    "strength": "duplicate rate / spacing repair on some ZDT slices",
                    "weakness": "DTLZ convergence trade-off",
                    "current_decision": "Hold for more benchmarks",
                },
                {
                    "candidate": "candidate_j_h_lite_retry2",
                    "status": "under_validation",
                    "strength": "lighter duplicate retry with lower trade-off than candidate_h",
                    "weakness": "spacing/count gap vs external remains",
                    "current_decision": decision_payload["final_decision"],
                },
            ],
            ["candidate", "status", "strength", "weakness", "current_decision"],
        ),
        "",
        "## 6. Experiment Configuration",
        "",
        *BASE._markdown_table(
            [
                {
                    "problem": spec.problem,
                    "algorithms": "internal, candidate_d, candidate_h, candidate_j, pymoo, deap, random_archive",
                    "seeds": len(seeds),
                    "requested_budget": args.budget,
                    "actual_evaluations_summary": "see aggregate_rows / fairness report",
                    "key_settings": f"pop={base_config.population_size}, gen={base_config.generations}, crossover={base_config.crossover}, mutation={base_config.mutation}",
                }
                for spec in selected_specs
            ],
            ["problem", "algorithms", "seeds", "requested_budget", "actual_evaluations_summary", "key_settings"],
        ),
        "",
        "## 7. Results Summary",
        "",
        *BASE._markdown_table(
            aggregate_rows,
            ["problem", "algorithm", "mean_hv", "mean_distance", "mean_gd", "mean_igd", "mean_spacing", "mean_coverage", "mean_nondominated_count", "mean_duplicate_rate", "mean_runtime_seconds"],
        ),
        "",
        "## 8. Paired Comparisons",
        "",
        *BASE._markdown_table(
            paired_rows,
            ["problem", "comparison", "metric", "win", "tie", "loss", "mean_delta", "median_delta"],
        ),
        "",
        "## 9. Candidate J Decision",
        "",
        *BASE._markdown_table(
            [
                {
                    "candidate": "candidate_j_h_lite_retry2",
                    "zdt_results": f"positive_problems={sum(1 for row in decision_payload['problem_rows'] if row['problem'].startswith('zdt') and '개선' in row['candidate_d_vs'])}",
                    "dtlz_results": f"positive_problems={sum(1 for row in decision_payload['problem_rows'] if row['problem'].startswith('dtlz') and ('개선' in row['candidate_d_vs'] or '우세' in row['candidate_d_vs']))}",
                    "wfg_results": "see WFG aggregate rows" if wfg_only else "not executed in this pass",
                    "candidate_d_vs": "see problem rows",
                    "candidate_h_vs": "see problem rows",
                    "external_vs": "see problem rows",
                    "fairness": fairness_payload["status"],
                    "decision": decision_payload["final_decision"],
                }
            ],
            ["candidate", "zdt_results", "dtlz_results", "wfg_results", "candidate_d_vs", "candidate_h_vs", "external_vs", "fairness", "decision"],
        ),
        "",
        "## 10. What We Learned",
        "",
        "- candidate_j가 candidate_h의 trade-off를 줄였는지는 candidate_j vs candidate_h paired rows를 기준으로 본다.",
        "- candidate_j가 candidate_d의 diversity 약점을 줄였는지는 spacing/nondominated_count와 duplicate metrics를 함께 봐야 한다.",
        "- DTLZ2/DTLZ3/DTLZ4에서 안정성이 충분히 유지되는지 여부가 change-request 승격의 핵심이다.",
        "- WFG는 이번 패스에서 shared adapter를 추가하지 않아 해석 범위를 넓히지 않았다.",
        "- spacing/nondominated_count 약점이 남으면 default promotion은 여전히 금지다.",
        "",
        "## 11. Failures, Skips, and Mismatches",
        "",
        *BASE._markdown_table(
            failures or [{"type": "none", "target": "none", "problem": "none", "seed": 0, "message": "none", "impact": "none", "action": "none"}],
            ["type", "target", "problem", "seed", "message", "impact", "action"],
        ),
        "",
        "## 12. Regression Check",
        "",
        "| command | result | note |",
        "| --- | --- | --- |",
        "| python scripts/check_local_baseline.py --output-dir artifacts/candidate_j_extended_guard | pending_from_shell | executed separately in shell |",
        f"| python scripts/validate_nsga2_candidate_j_extended.py --artifact-suffix {args.artifact_suffix or 'none'} | success | current run |",
        "",
        "## 13. Maturity Impact",
        "",
        "- candidate_j가 좋아도 기본값을 바꾸지 않았으므로 알고리즘 maturity 상향은 주장하지 않는다.",
        "- parameter fairness automation이 안정적으로 추가되면 실험 툴킷으로서 Level 4 근거는 강화될 수 있다.",
        "- pymoo/DEAP 대비 여전히 약하면 범용 optimizer 성숙도 상향은 금지다.",
        "",
        "## 14. Recommended Next Work",
        "",
        "1. candidate_j가 Promote면 change request 작성 또는 보강.",
        "2. candidate_j가 Hold면 WFG/DTLZ 추가 validation.",
        "3. spacing/nondominated_count 개선용 survivor-pressure 가설 재설계.",
        "4. parameter fairness checker를 모든 comparison runner에 통합.",
        "5. constrained multi-objective contract, checkpoint/resume, parallel evaluation은 후속 과제로 유지.",
        "",
        f"이번 candidate_j extended validation 결과, candidate_j는 {', '.join(problem_names)} benchmark에서 trade-off를 점검했고, external comparator 대비 여전히 일부 gap이 남아 최종 판정은 {decision_payload['final_decision']}이다.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    fairness_lines = [
        "# NSGA-II Candidate J WFG Fairness Report"
        if wfg_only
        else "# NSGA-II Candidate J Fairness Report",
        "",
        f"- Overall fairness status: **{fairness_payload['status']}**",
        "",
        "## Policy",
        "",
        *BASE._markdown_table(
            fairness_payload["policy_rows"],
            ["fairness_item", "pass_rule", "warning_rule", "fail_rule"],
        ),
        "",
        "## Issue Rows",
        "",
        *BASE._markdown_table(
            fairness_payload["issues"],
            ["status", "issue_type", "problem", "algorithm", "message", "severity", "recommended_action"],
        ),
    ]
    fairness_report_path.write_text("\n".join(fairness_lines) + "\n", encoding="utf-8")

    print(json.dumps({"results_json": str(json_path), "report_md": str(report_path), "fairness_md": str(fairness_report_path), "decision": decision_payload["final_decision"], "fairness_status": fairness_payload["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
