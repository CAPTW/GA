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
    candidate_i_h_lite_retry1,
    candidate_j_h_lite_retry2,
    candidate_k_dedup_only_lite,
    candidate_variant_metadata,
)
from ga_lab.experiment.parameter_fairness import evaluate_parameter_fairness


def _load_base_helpers():
    helper_path = PROJECT_ROOT / "scripts" / "validate_nsga2_candidate_suite.py"
    spec = importlib.util.spec_from_file_location("_candidate_suite_base_h_lite", helper_path)
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
        description="Validate NSGA-II h-lite candidates after default drift audit passes."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problems", default="zdt1,zdt2,zdt3,dtlz2,dtlz3")
    parser.add_argument("--output-root", default="outputs/nsga2_h_lite_candidate_validation")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=9201)
    parser.add_argument("--budget", type=int, default=760)
    return parser.parse_args()


def _candidate_variants() -> list[NSGA2CandidateVariant]:
    return [
        candidate_d_uniform_crossover(),
        candidate_h_uniform_dedup_mutation_boost(),
        candidate_i_h_lite_retry1(),
        candidate_j_h_lite_retry2(),
        candidate_k_dedup_only_lite(),
    ]


def _decorate_row(row: dict[str, Any], *, reference_front: list[list[float]]) -> dict[str, Any]:
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
        int(row["seed"]): row for row in right_rows if row.get("success") and int(row.get("seed", -1)) >= 0
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
        if row["problem"] == problem and row["left_algorithm"] == left and row["right_algorithm"] == right and row["metric"] == metric:
            return row
    return None


def _candidate_decisions(aggregate_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], candidate_ids: list[str]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        statuses: dict[str, str] = {}
        zdt_positive = 0
        dtlz_stable = 0
        for problem in sorted({row["problem"] for row in aggregate_rows}):
            spacing_d = _pick(paired_rows, problem, candidate_id, "candidate_d_uniform_crossover", "spacing")
            count_d = _pick(paired_rows, problem, candidate_id, "candidate_d_uniform_crossover", "nondominated_count")
            dup_d = _pick(paired_rows, problem, candidate_id, "candidate_d_uniform_crossover", "archive_duplicate_rate")
            hv_h = _pick(paired_rows, problem, candidate_id, "candidate_h_uniform_dedup_mutation_boost", "hypervolume_2d")
            dist_h = _pick(paired_rows, problem, candidate_id, "candidate_h_uniform_dedup_mutation_boost", "reference_front_distance")
            igd_h = _pick(paired_rows, problem, candidate_id, "candidate_h_uniform_dedup_mutation_boost", "inverted_generational_distance")

            diversity_good = sum(int(row["win"]) > int(row["loss"]) for row in (spacing_d, count_d, dup_d) if row) >= 2
            tradeoff_reduced = sum(int(row["win"]) >= int(row["loss"]) for row in (hv_h, dist_h, igd_h) if row) >= 2

            if diversity_good and tradeoff_reduced:
                statuses[problem] = "candidate_h trade-off reduced with diversity preserved"
            elif diversity_good:
                statuses[problem] = "diversity gain with residual candidate_h trade-off"
            elif tradeoff_reduced:
                statuses[problem] = "trade-off reduced but diversity gain limited"
            else:
                statuses[problem] = "no material improvement vs candidate_h/candidate_d"

            if problem.startswith("zdt") and statuses[problem] == "candidate_h trade-off reduced with diversity preserved":
                zdt_positive += 1
            if problem.startswith("dtlz") and statuses[problem] in {
                "candidate_h trade-off reduced with diversity preserved",
                "trade-off reduced but diversity gain limited",
            }:
                dtlz_stable += 1

        if zdt_positive >= 2 and dtlz_stable >= 1:
            final = "Promote to change request"
        elif zdt_positive >= 1:
            final = "Hold for more benchmarks"
        elif any("trade-off reduced" in status for status in statuses.values()):
            final = "Merge idea into candidate_h follow-up"
        else:
            final = "Reject"

        decisions.append(
            {
                "candidate": candidate_id,
                "zdt1": statuses.get("zdt1", "n/a"),
                "zdt2": statuses.get("zdt2", "n/a"),
                "zdt3": statuses.get("zdt3", "n/a"),
                "dtlz2": statuses.get("dtlz2", "n/a"),
                "dtlz3": statuses.get("dtlz3", "n/a"),
                "candidate_d_vs": "; ".join(f"{k}:{v}" for k, v in statuses.items()),
                "candidate_h_vs": "see paired rows",
                "external_vs": "see paired rows",
                "final_decision": final,
            }
        )
    return decisions


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(PROJECT_ROOT / args.config)
    problem_names = [item.strip().lower() for item in str(args.problems).split(",") if item.strip()]
    selected_specs = [mo_candidate_suite_specs()[name] for name in problem_names]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    variants = _candidate_variants()

    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    variant_map = {variant.candidate_id: variant for variant in variants}
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    comparison_specs = [
        ("candidate_i_h_lite_retry1", "candidate_h_uniform_dedup_mutation_boost", "candidate_i vs candidate_h"),
        ("candidate_j_h_lite_retry2", "candidate_h_uniform_dedup_mutation_boost", "candidate_j vs candidate_h"),
        ("candidate_k_dedup_only_lite", "candidate_h_uniform_dedup_mutation_boost", "candidate_k vs candidate_h"),
        ("candidate_i_h_lite_retry1", "candidate_d_uniform_crossover", "candidate_i vs candidate_d"),
        ("candidate_j_h_lite_retry2", "candidate_d_uniform_crossover", "candidate_j vs candidate_d"),
        ("candidate_k_dedup_only_lite", "candidate_d_uniform_crossover", "candidate_k vs candidate_d"),
        ("candidate_i_h_lite_retry1", "internal_nsga2", "candidate_i vs internal baseline"),
        ("candidate_j_h_lite_retry2", "internal_nsga2", "candidate_j vs internal baseline"),
        ("candidate_k_dedup_only_lite", "internal_nsga2", "candidate_k vs internal baseline"),
        ("candidate_i_h_lite_retry1", "pymoo_nsga2", "candidate_i vs pymoo"),
        ("candidate_j_h_lite_retry2", "pymoo_nsga2", "candidate_j vs pymoo"),
        ("candidate_k_dedup_only_lite", "pymoo_nsga2", "candidate_k vs pymoo"),
        ("candidate_i_h_lite_retry1", "deap_nsga2", "candidate_i vs deap"),
        ("candidate_j_h_lite_retry2", "deap_nsga2", "candidate_j vs deap"),
        ("candidate_k_dedup_only_lite", "deap_nsga2", "candidate_k vs deap"),
        ("candidate_i_h_lite_retry1", "random_pareto_archive", "candidate_i vs random archive"),
        ("candidate_j_h_lite_retry2", "random_pareto_archive", "candidate_j vs random archive"),
        ("candidate_k_dedup_only_lite", "random_pareto_archive", "candidate_k vs random archive"),
        ("candidate_h_uniform_dedup_mutation_boost", "pymoo_nsga2", "candidate_h vs pymoo"),
        ("candidate_h_uniform_dedup_mutation_boost", "deap_nsga2", "candidate_h vs deap"),
    ]

    for spec in selected_specs:
        config = build_problem_config(base_config, spec)
        reference_front = reference_front_for_spec(spec, point_count=201)
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
                row = _decorate_row(row, reference_front=reference_front)
                row = decorate_fairness_row(
                    row,
                    spec=spec,
                    base_config=config,
                    requested_budget=args.budget,
                    variant_map=variant_map,
                )
                row["benchmark_reference_front"] = spec.reference_front_name
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
                            "action": "review comparator/runtime failure before promotion",
                        }
                    )

    aggregate_rows = _aggregate_rows(raw_rows)
    paired_rows = _paired_rows(raw_rows, comparison_specs)
    decision_rows = _candidate_decisions(
        aggregate_rows,
        paired_rows,
        [variant.candidate_id for variant in variants if variant.candidate_id not in {
            "candidate_d_uniform_crossover",
            "candidate_h_uniform_dedup_mutation_boost",
        }],
    )
    fairness_payload = evaluate_parameter_fairness(
        raw_rows,
        benchmark_rows=benchmark_rows,
        candidate_rows=candidate_rows,
    )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "seeds": seeds,
        "budget": args.budget,
        "selected_problems": problem_names,
        "candidate_rows": candidate_rows,
        "benchmark_rows": benchmark_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "decision_rows": decision_rows,
        "failures": failures,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
    }

    json_path = safe_artifact_path(artifact_root, "nsga2_h_lite_candidate_results", args.artifact_suffix, ".json")
    csv_path = safe_artifact_path(artifact_root, "nsga2_h_lite_candidate_results", args.artifact_suffix, ".csv")
    md_path = safe_artifact_path(artifact_root, "nsga2_h_lite_candidate_results", args.artifact_suffix, ".md")
    report_path = safe_artifact_path(artifact_root, "nsga2_h_lite_candidate_report", args.artifact_suffix, ".md")

    BASE._write_json(json_path, payload)
    BASE._write_csv(
        csv_path,
        aggregate_rows,
        [
            "problem","algorithm","library","status","seeds","successful_seeds","mean_hv","mean_distance","mean_gd",
            "mean_igd","mean_spacing","mean_coverage","mean_nondominated_count","mean_duplicate_rate",
            "mean_archive_duplicate_rate","mean_objective_duplicate_rate","mean_decision_duplicate_rate",
            "mean_unique_decision_count","mean_unique_objective_count","mean_boundary_point_count",
            "mean_runtime_seconds","mean_actual_evaluations","success_rate",
        ],
    )
    md_path.write_text(
        "\n".join(
            [
                "# NSGA-II H-Lite Candidate Results",
                "",
                "## Aggregate Results",
                "",
                *BASE._markdown_table(
                    aggregate_rows,
                    ["problem","algorithm","mean_hv","mean_distance","mean_igd","mean_spacing","mean_nondominated_count","mean_duplicate_rate","mean_runtime_seconds"],
                ),
                "",
                "## Candidate Decisions",
                "",
                *BASE._markdown_table(
                    decision_rows,
                    ["candidate","zdt1","zdt2","zdt3","dtlz2","dtlz3","final_decision"],
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
        ) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        "\n".join(
            [
                "# NSGA-II H-Lite Candidate Report",
                "",
                "## Benchmarks",
                "",
                *BASE._markdown_table(
                    benchmark_rows,
                    ["problem","variables","objectives","bounds","reference_front","hv_reference_point","notes"],
                ),
                "",
                "## Candidates",
                "",
                *BASE._markdown_table(
                    candidate_rows,
                    ["candidate_id","base_candidate_id","operator_change","default_changed","promotion_status","expected_gain","risk"],
                ),
                "",
                "## Fairness Summary",
                "",
                *BASE._markdown_table(
                    fairness_summary_rows(fairness_payload),
                    ["status", "pass", "warning", "fail"],
                ),
                "",
                *BASE._markdown_table(
                    fairness_payload["issues"],
                    ["status","issue_type","algorithm","problem","message"],
                ),
                "",
                "## Aggregate Results",
                "",
                *BASE._markdown_table(
                    aggregate_rows,
                    ["problem","algorithm","mean_hv","mean_distance","mean_igd","mean_spacing","mean_nondominated_count","mean_duplicate_rate","mean_runtime_seconds"],
                ),
                "",
                "## Paired Summary",
                "",
                *BASE._markdown_table(
                    paired_rows,
                    ["problem","comparison","metric","win","tie","loss","mean_delta","median_delta","comparable_seeds"],
                ),
                "",
                "## Candidate Decisions",
                "",
                *BASE._markdown_table(
                    decision_rows,
                    ["candidate","zdt1","zdt2","zdt3","dtlz2","dtlz3","candidate_d_vs","candidate_h_vs","external_vs","final_decision"],
                ),
                "",
                "## Failures",
                "",
                *BASE._markdown_table(
                    failures or [{"type":"none","target":"none","problem":"none","seed":0,"message":"none","impact":"none","action":"none"}],
                    ["type","target","problem","seed","message","impact","action"],
                ),
                "",
            ]
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
