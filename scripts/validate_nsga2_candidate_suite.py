from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from ga_lab.config import GAConfig, load_config
from ga_lab.experiment.external_mo_comparators import (
    METRIC_SPECS,
    ExternalMOComparatorResult,
    paired_metric_summary,
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
    apply_candidate_variant,
    candidate_d_uniform_crossover,
    candidate_variant_metadata,
)
from ga_lab.experiment.parameter_fairness import evaluate_parameter_fairness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate NSGA-II candidate variants across a small MO benchmark suite."
    )
    parser.add_argument(
        "--config",
        default="configs/smoke/zdt1_nsga2_smoke.json",
        help="Base NSGA-II config used as the internal baseline anchor.",
    )
    parser.add_argument(
        "--problems",
        default="zdt1,zdt2,zdt3,dtlz2",
        help="Comma-separated benchmark list.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/nsga2_candidate_suite_validation",
        help="Directory for timestamped raw outputs.",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Directory for stable validation artifacts.",
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Optional suffix for artifact names.",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of repeated seeds.")
    parser.add_argument("--seed-start", type=int, default=8101, help="First seed.")
    parser.add_argument("--budget", type=int, default=760, help="Requested evaluation budget.")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column) for column in columns} for row in rows])


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _summary_stat(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = _finite_values(rows, key)
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "std": 0.0 if len(values) == 1 else stdev(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _success_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    successful = [row for row in rows if row.get("success")]
    return len(successful) / len(rows)


def _format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    return lines


def _decorate_front_row(
    row: dict[str, Any],
    *,
    reference_front: list[list[float]],
) -> dict[str, Any]:
    if not row.get("success"):
        row["reference_front_coverage"] = None
        row["front_unique_count"] = 0
        row["front_duplicate_count"] = 0
        return row
    directions = [
        bool(value)
        for value in row.get("metadata", {}).get("objective_directions", [False, False])
    ]
    nondominated_front = row.get("nondominated_objective_vectors", [])
    objective_vectors = row.get("objective_vectors", [])
    row["reference_front_coverage"] = coverage_indicator(
        nondominated_front,
        reference_front,
        directions,
    )
    unique_count = len({tuple(vector) for vector in objective_vectors})
    row["front_unique_count"] = unique_count
    row["front_duplicate_count"] = len(objective_vectors) - unique_count
    return row


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    aggregates: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        successful = [row for row in bucket if row.get("success")]
        hv = _summary_stat(successful, "hypervolume_2d")
        distance = _summary_stat(successful, "reference_front_distance")
        gd = _summary_stat(successful, "generational_distance")
        igd = _summary_stat(successful, "inverted_generational_distance")
        spacing = _summary_stat(successful, "spacing")
        coverage = _summary_stat(successful, "reference_front_coverage")
        count = _summary_stat(successful, "nondominated_count")
        runtime = _summary_stat(successful, "runtime_seconds")
        evaluations = _summary_stat(successful, "actual_evaluations")
        statuses = {str(row.get("status", "unknown")) for row in bucket}
        if statuses == {"skipped"}:
            status = "skipped"
        elif "failed" in statuses and not successful:
            status = "failed"
        elif "failed" in statuses:
            status = "partial_failure"
        else:
            status = "success"
        aggregates.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "library": bucket[0].get("library"),
                "status": status,
                "seeds": len(bucket),
                "successful_seeds": len(successful),
                "mean_hv": hv["mean"],
                "mean_distance": distance["mean"],
                "mean_gd": gd["mean"],
                "mean_igd": igd["mean"],
                "mean_spacing": spacing["mean"],
                "mean_coverage": coverage["mean"],
                "mean_nondominated_count": count["mean"],
                "mean_runtime_seconds": runtime["mean"],
                "mean_actual_evaluations": evaluations["mean"],
                "success_rate": _success_rate(bucket),
            }
        )
    return aggregates


def _paired_rows(
    rows: list[dict[str, Any]],
    comparison_specs: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    paired: list[dict[str, Any]] = []
    for problem in sorted({str(row["problem"]) for row in rows}):
        for left, right, label in comparison_specs:
            left_rows = grouped.get((problem, left), [])
            right_rows = grouped.get((problem, right), [])
            for metric_name in METRIC_SPECS:
                summary = paired_metric_summary(
                    internal_rows=left_rows,
                    comparator_rows=right_rows,
                    metric_name=metric_name,
                )
                paired.append(
                    {
                        "problem": problem,
                        "comparison": label,
                        "left_algorithm": left,
                        "right_algorithm": right,
                        "metric": metric_name,
                        "win": summary["internal_win"],
                        "tie": summary["tie"],
                        "loss": summary["external_win"],
                        "mean_delta": summary["mean_delta"],
                        "median_delta": summary["median_delta"],
                        "comparable_seeds": summary["comparable_seed_count"],
                    }
                )
    return paired


def _make_candidate_result(
    base_config: GAConfig,
    *,
    seed: int,
    output_root: Path,
) -> ExternalMOComparatorResult:
    variant = candidate_d_uniform_crossover()
    candidate_config = apply_candidate_variant(base_config, variant)
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


def _candidate_problem_decisions(
    paired_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    core_metrics = {
        "hypervolume_2d",
        "reference_front_distance",
        "generational_distance",
        "inverted_generational_distance",
        "coverage_indicator",
    }
    problem_rows: list[dict[str, Any]] = []
    promotion_score = 0
    mixed_problems = 0
    fairness_issue = False

    for problem in sorted({str(row["problem"]) for row in paired_rows}):
        candidate_vs_internal = [
            row
            for row in paired_rows
            if row["problem"] == problem
            and row["comparison"] == "candidate_d vs internal baseline"
            and row["metric"] in core_metrics
        ]
        candidate_vs_pymoo = [
            row
            for row in paired_rows
            if row["problem"] == problem
            and row["comparison"] == "candidate_d vs pymoo"
            and row["metric"] in core_metrics
        ]
        candidate_vs_deap = [
            row
            for row in paired_rows
            if row["problem"] == problem
            and row["comparison"] == "candidate_d vs deap"
            and row["metric"] in core_metrics
        ]
        if any(int(row.get("comparable_seeds", 0)) == 0 for row in candidate_vs_internal):
            fairness_issue = True

        wins_internal = sum(int(row["win"]) > int(row["loss"]) for row in candidate_vs_internal)
        losses_internal = sum(int(row["win"]) < int(row["loss"]) for row in candidate_vs_internal)
        wins_pymoo = sum(int(row["win"]) > int(row["loss"]) for row in candidate_vs_pymoo)
        losses_pymoo = sum(int(row["win"]) < int(row["loss"]) for row in candidate_vs_pymoo)
        wins_deap = sum(int(row["win"]) > int(row["loss"]) for row in candidate_vs_deap)
        losses_deap = sum(int(row["win"]) < int(row["loss"]) for row in candidate_vs_deap)

        if wins_internal >= 4 and losses_internal == 0:
            baseline_read = "core metrics improved vs internal"
            promotion_score += 1
        elif wins_internal >= 3:
            baseline_read = "mixed but favorable vs internal"
            mixed_problems += 1
        else:
            baseline_read = "no robust gain vs internal"

        if wins_pymoo >= 3 or wins_deap >= 3:
            external_read = "competitive on some core metrics"
        elif losses_pymoo >= 4 and losses_deap >= 4:
            external_read = "still clearly behind external comparators"
        else:
            external_read = "mixed versus external comparators"

        if problem == "zdt1":
            verdict = "strong prior signal"
        elif wins_internal >= 4 and losses_internal == 0:
            verdict = "reproduced"
        elif wins_internal >= 2:
            verdict = "mixed"
        else:
            verdict = "not reproduced"

        problem_rows.append(
            {
                "candidate": "candidate_d_uniform_crossover",
                "problem": problem,
                "internal_baseline": baseline_read,
                "pymoo": external_read if candidate_vs_pymoo else "not compared",
                "deap": external_read if candidate_vs_deap else "not compared",
                "decision_hint": verdict,
                "fairness": (
                    "paired seeds available"
                    if not any(int(row.get("comparable_seeds", 0)) == 0 for row in candidate_vs_internal)
                    else "missing comparable seeds"
                ),
            }
        )

    if fairness_issue:
        decision = "Needs parameter fairness rerun"
    elif promotion_score >= 3:
        decision = "Promote to change request"
    elif promotion_score >= 1 or mixed_problems >= 1:
        decision = "Hold for more benchmarks"
    else:
        decision = "Reject"
    return problem_rows, decision


def _results_markdown(
    payload: dict[str, Any],
    *,
    benchmark_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# NSGA-II Candidate Suite Validation Results",
        "",
        "## Benchmarks",
        "",
        *_markdown_table(
            benchmark_rows,
            [
                "problem",
                "variables",
                "objectives",
                "bounds",
                "reference_front",
                "hv_reference_point",
                "notes",
            ],
        ),
        "",
        "## Candidate",
        "",
        *_markdown_table(
            candidate_rows,
            [
                "candidate_id",
                "operator_change",
                "default_changed",
                "promotion_status",
                "source_diagnosis_report",
                "expected_gain",
                "risk",
            ],
        ),
        "",
        "## Fairness Summary",
        "",
        *_markdown_table(
            fairness_summary_rows(payload["fairness"]),
            ["status", "pass", "warning", "fail"],
        ),
        "",
        "## Fairness Issues",
        "",
        *_markdown_table(
            payload["fairness"]["issues"],
            ["status", "issue_type", "algorithm", "problem", "message", "recommended_action"],
        ),
        "",
        "## Aggregate Results",
        "",
        *_markdown_table(
            payload["aggregate_rows"],
            [
                "problem",
                "algorithm",
                "status",
                "seeds",
                "mean_hv",
                "mean_distance",
                "mean_gd",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Paired Results",
        "",
        *_markdown_table(
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
            ],
        ),
        "",
        "## Candidate Decision Matrix",
        "",
        *_markdown_table(
            decision_rows,
            [
                "candidate",
                "problem",
                "internal_baseline",
                "pymoo",
                "deap",
                "decision_hint",
                "fairness",
            ],
        ),
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
        item.strip().lower() for item in str(args.problems).split(",") if item.strip()
    ]
    suite_specs = mo_candidate_suite_specs()
    selected_specs = [suite_specs[name] for name in requested_problem_names]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]

    variant = candidate_d_uniform_crossover()
    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows([variant])
    variant_map = {variant.candidate_id: variant}

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    comparison_specs = [
        ("candidate_d_uniform_crossover", "internal_nsga2", "candidate_d vs internal baseline"),
        ("candidate_d_uniform_crossover", "pymoo_nsga2", "candidate_d vs pymoo"),
        ("candidate_d_uniform_crossover", "deap_nsga2", "candidate_d vs deap"),
        ("candidate_d_uniform_crossover", "random_pareto_archive", "candidate_d vs random archive"),
        ("internal_nsga2", "pymoo_nsga2", "internal baseline vs pymoo"),
        ("internal_nsga2", "deap_nsga2", "internal baseline vs deap"),
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
                _make_candidate_result(config, seed=seed, output_root=problem_output_root),
                run_pymoo_nsga2(config, seed=seed, budget=args.budget),
                run_deap_nsga2(config, seed=seed, budget=args.budget),
                run_random_archive_anchor(run_random_pareto_archive(config, seed=seed, budget=args.budget)),
            ]
            for result in results:
                row = result_to_front_row(
                    result,
                    reference_front=reference_front,
                    reference_point=reference_point,
                )
                row = _decorate_front_row(row, reference_front=reference_front)
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
                            "impact": "seed excluded from paired metric comparison",
                            "action": "review comparator/runtime failure before promotion",
                        }
                    )

    aggregate_rows = _aggregate_rows(raw_rows)
    paired_rows = _paired_rows(raw_rows, comparison_specs)
    decision_rows, promotion_decision = _candidate_problem_decisions(paired_rows)
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
        "selected_problems": requested_problem_names,
        "candidate": candidate_rows[0],
        "benchmark_rows": benchmark_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "decision_rows": decision_rows,
        "promotion_decision": promotion_decision,
        "failures": failures,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
    }

    json_path = safe_artifact_path(
        artifact_root, "nsga2_candidate_suite_validation_results", args.artifact_suffix, ".json"
    )
    csv_path = safe_artifact_path(
        artifact_root, "nsga2_candidate_suite_validation_results", args.artifact_suffix, ".csv"
    )
    md_path = safe_artifact_path(
        artifact_root, "nsga2_candidate_suite_validation_results", args.artifact_suffix, ".md"
    )
    report_path = safe_artifact_path(
        artifact_root, "nsga2_candidate_suite_validation_report", args.artifact_suffix, ".md"
    )

    _write_json(json_path, payload)
    _write_csv(
        csv_path,
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
            "mean_gd",
            "mean_igd",
            "mean_spacing",
            "mean_coverage",
            "mean_nondominated_count",
            "mean_runtime_seconds",
            "mean_actual_evaluations",
            "success_rate",
        ],
    )
    md_path.write_text(
        _results_markdown(
            payload,
            benchmark_rows=benchmark_rows,
            candidate_rows=candidate_rows,
            decision_rows=decision_rows,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# NSGA-II Candidate Suite Validation Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- 검증 대상 candidate: `{variant.candidate_id}`",
        f"- 실행 benchmark: {', '.join(requested_problem_names)}",
        f"- seed 수: {len(seeds)}, requested budget: {args.budget}",
        f"- 최종 promotion decision: **{promotion_decision}**",
        "- 기본 internal NSGA-II 기본값은 변경하지 않았다.",
        "",
        "## 2. Benchmark Definitions",
        "",
        *_markdown_table(
            benchmark_rows,
            [
                "problem",
                "variables",
                "objectives",
                "bounds",
                "reference_front",
                "hv_reference_point",
                "notes",
            ],
        ),
        "",
        "## 3. Candidate Definition",
        "",
        *_markdown_table(
            candidate_rows,
            [
                "candidate_id",
                "operator_change",
                "default_changed",
                "promotion_status",
                "source_diagnosis_report",
                "expected_gain",
                "risk",
            ],
        ),
        "",
        "## 4. Fairness Summary",
        "",
        *_markdown_table(
            fairness_summary_rows(fairness_payload),
            ["status", "pass", "warning", "fail"],
        ),
        "",
        *_markdown_table(
            fairness_payload["issues"],
            ["status", "issue_type", "algorithm", "problem", "message"],
        ),
        "",
        "## 5. Results Summary",
        "",
        *_markdown_table(
            aggregate_rows,
            [
                "problem",
                "algorithm",
                "mean_hv",
                "mean_distance",
                "mean_gd",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## 6. Paired Comparisons",
        "",
        *_markdown_table(
            paired_rows,
            [
                "problem",
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
        "## 7. Candidate Promotion Decision",
        "",
        *_markdown_table(
            decision_rows,
            [
                "candidate",
                "problem",
                "internal_baseline",
                "pymoo",
                "deap",
                "decision_hint",
                "fairness",
            ],
        ),
        "",
        "최종 결정:",
        f"- **{promotion_decision}**",
        "",
        "## 8. Failures, Skips, and Mismatches",
        "",
        *(
            _markdown_table(
                failures,
                ["type", "target", "problem", "seed", "message", "impact", "action"],
            )
            if failures
            else ["- 없음"]
        ),
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({"results_json": str(json_path), "report_md": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
