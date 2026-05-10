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
    candidate_l_sparse_parent_bias_light,
    candidate_variant_metadata,
)
from ga_lab.experiment.parameter_fairness import evaluate_parameter_fairness


def _load_base_helpers():
    helper_path = PROJECT_ROOT / "scripts" / "validate_nsga2_candidate_suite.py"
    spec = importlib.util.spec_from_file_location("_candidate_suite_base_survivor_phase1", helper_path)
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
        description="Run Phase 1 ZDT repeated-seed validation for survivor-pressure candidate_l."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problems", default="zdt1,zdt2,zdt3")
    parser.add_argument("--output-root", default="outputs/nsga2_survivor_pressure_phase1")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=10201)
    parser.add_argument("--budget", type=int, default=760)
    parser.add_argument(
        "--drift-audit-json",
        default="artifacts/nsga2_candidate_l_default_drift_audit_results.json",
        help="Optional drift-audit JSON included in the Phase 1 report if it exists.",
    )
    parser.add_argument("--skip-pymoo", action="store_true")
    parser.add_argument("--skip-deap", action="store_true")
    parser.add_argument("--skip-random-archive", action="store_true")
    return parser.parse_args()


def _candidate_variants() -> list[NSGA2CandidateVariant]:
    return [
        candidate_d_uniform_crossover(),
        candidate_h_uniform_dedup_mutation_boost(),
        candidate_j_h_lite_retry2(),
        candidate_l_sparse_parent_bias_light(),
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


def _candidate_result(
    config: GAConfig,
    variant: NSGA2CandidateVariant,
    *,
    seed: int,
    output_root: Path,
) -> ExternalMOComparatorResult:
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


def _decorate_row(
    row: dict[str, Any],
    *,
    spec: MOBenchmarkSpec,
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
    else:
        row["reference_front_coverage"] = None
        row["decision_duplicate_rate"] = None
        row["objective_duplicate_rate"] = None
        row["archive_duplicate_rate"] = None
        row["unique_decision_count"] = None
        row["unique_objective_count"] = None
        row["boundary_point_count"] = None
        row["metric_calculation_success"] = False

    row = decorate_fairness_row(
        row,
        spec=spec,
        base_config=base_config,
        requested_budget=requested_budget,
        variant_map=variant_map,
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
                summary = _paired_metric_summary(
                    left_rows,
                    right_rows,
                    metric_name,
                    metric_spec["higher_is_better"],
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


def _metric_wins(row: dict[str, Any] | None) -> bool:
    return row is not None and int(row["win"]) > int(row["loss"])


def _metric_losses(row: dict[str, Any] | None) -> bool:
    return row is not None and int(row["win"]) < int(row["loss"])


def _candidate_gate_rows(
    rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    default_rows = [row for row in rows if row["algorithm"] == "internal_nsga2"]
    candidate_l_rows = [row for row in rows if row["algorithm"] == "candidate_l_sparse_parent_bias_light"]

    metadata_ok = all(
        row.get("metadata", {}).get("candidate_id") == "candidate_l_sparse_parent_bias_light"
        and row.get("metadata", {}).get("base_candidate_id") == "candidate_j_h_lite_retry2"
        and row.get("metadata", {}).get("mechanism") == "sparse_region_parent_bias_light"
        for row in candidate_l_rows
    )
    default_changed_ok = all(
        row.get("metadata", {}).get("default_changed") is False for row in candidate_l_rows
    )
    candidate_isolation_ok = all(
        "candidate_id" not in row.get("metadata", {})
        and "default_changed" not in row.get("metadata", {})
        for row in default_rows
    )
    evaluations_ok = all(
        row.get("requested_budget") == row.get("actual_evaluations")
        for row in candidate_l_rows
        if row.get("success")
    )
    metric_ok = all(bool(row.get("metric_calculation_success")) for row in candidate_l_rows if row.get("success"))
    fairness_fail_free = fairness_payload.get("summary_counts", {}).get("fail", 0) == 0
    catastrophic_regression = any(
        isinstance(row.get("hypervolume_2d"), int | float) and float(row["hypervolume_2d"]) <= 0.0
        for row in candidate_l_rows
        if row.get("success")
    )

    return [
        {
            "gate": "candidate metadata",
            "result": metadata_ok,
            "interpretation": "candidate_l metadata present with the expected base candidate and mechanism",
        },
        {
            "gate": "default_changed=false",
            "result": default_changed_ok,
            "interpretation": "candidate_l rows keep default_changed=false",
        },
        {
            "gate": "candidate isolation",
            "result": candidate_isolation_ok,
            "interpretation": "default internal NSGA-II rows stay candidate-metadata free",
        },
        {
            "gate": "fairness fail 없음",
            "result": fairness_fail_free,
            "interpretation": f"fairness summary={fairness_payload.get('summary_counts', {})}",
        },
        {
            "gate": "actual evaluations",
            "result": evaluations_ok,
            "interpretation": "candidate_l actual_evaluations match requested budget",
        },
        {
            "gate": "metric calculation success",
            "result": metric_ok,
            "interpretation": "Phase 1 core MO metrics stayed finite for candidate_l",
        },
        {
            "gate": "catastrophic regression",
            "result": not catastrophic_regression,
            "interpretation": "no zero-or-negative hypervolume collapse observed for candidate_l",
        },
    ]


def _phase1_problem_rows(
    paired_rows: list[dict[str, Any]],
    problems: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for problem in problems:
        spacing_j = _pick(
            paired_rows,
            problem,
            "candidate_l_sparse_parent_bias_light",
            "candidate_j_h_lite_retry2",
            "spacing",
        )
        count_j = _pick(
            paired_rows,
            problem,
            "candidate_l_sparse_parent_bias_light",
            "candidate_j_h_lite_retry2",
            "nondominated_count",
        )
        hv_j = _pick(
            paired_rows,
            problem,
            "candidate_l_sparse_parent_bias_light",
            "candidate_j_h_lite_retry2",
            "hypervolume_2d",
        )
        distance_j = _pick(
            paired_rows,
            problem,
            "candidate_l_sparse_parent_bias_light",
            "candidate_j_h_lite_retry2",
            "reference_front_distance",
        )
        igd_j = _pick(
            paired_rows,
            problem,
            "candidate_l_sparse_parent_bias_light",
            "candidate_j_h_lite_retry2",
            "inverted_generational_distance",
        )
        coverage_j = _pick(
            paired_rows,
            problem,
            "candidate_l_sparse_parent_bias_light",
            "candidate_j_h_lite_retry2",
            "coverage_indicator",
        )
        duplicate_j = _pick(
            paired_rows,
            problem,
            "candidate_l_sparse_parent_bias_light",
            "candidate_j_h_lite_retry2",
            "archive_duplicate_rate",
        )

        diversity_positive = _metric_wins(spacing_j) or _metric_wins(count_j)
        duplicate_positive = _metric_wins(duplicate_j)
        core_regressions = sum(
            _metric_losses(row) for row in (hv_j, distance_j, igd_j, coverage_j)
        )

        if diversity_positive and core_regressions == 0:
            read = "spacing 또는 nondominated_count 개선이 있었고 핵심 수렴 metric 회귀는 크지 않았다"
        elif diversity_positive:
            read = "diversity 신호는 있으나 HV/distance/IGD/coverage 일부 후퇴가 동반되었다"
        elif duplicate_positive and core_regressions == 0:
            read = "duplicate rate는 줄었지만 spacing/count 개선 신호는 약하다"
        elif core_regressions >= 3:
            read = "candidate_j 대비 핵심 수렴 metric 후퇴가 커서 부정 신호다"
        else:
            read = "candidate_j 대비 차이가 작거나 혼합 신호다"

        rows.append(
            {
                "problem": problem,
                "diversity_positive": diversity_positive,
                "duplicate_positive": duplicate_positive,
                "core_regressions": core_regressions,
                "interpretation": read,
            }
        )
    return rows


def _phase1_decision(
    gate_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    problem_rows: list[dict[str, Any]],
) -> str:
    if fairness_payload.get("summary_counts", {}).get("fail", 0) > 0:
        return "Needs fairness rerun"
    if not all(bool(row["result"]) for row in gate_rows):
        return "Fix required"

    positive_problems = sum(bool(row["diversity_positive"]) for row in problem_rows)
    catastrophic_problems = sum(int(row["core_regressions"]) >= 3 for row in problem_rows)
    mixed_problems = sum(
        bool(row["diversity_positive"]) and int(row["core_regressions"]) > 0 for row in problem_rows
    )

    if positive_problems >= 2 and catastrophic_problems == 0:
        return "Phase 1 passed, eligible for Phase 2 planning"
    if positive_problems >= 1 and catastrophic_problems == 0:
        return "Phase 1 passed with trade-offs"
    if positive_problems >= 1:
        return "Hold for more evidence"
    if catastrophic_problems >= 2 or mixed_problems == 0:
        return "Reject"
    return "Hold for more evidence"


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
        "# NSGA-II Survivor-Pressure Phase 1 Results",
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
                "mean_gd",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_duplicate_rate",
                "mean_runtime_seconds",
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
        *BASE._markdown_table(fairness_summary_rows(payload["fairness"]), ["status", "pass", "warning", "fail"]),
        "",
        "## Decision",
        "",
        f"- `{payload['phase1_decision']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def _fairness_report_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# NSGA-II Survivor-Pressure Phase 1 Fairness Report",
        "",
        "## Fairness Summary",
        "",
        *BASE._markdown_table(fairness_summary_rows(payload["fairness"]), ["status", "pass", "warning", "fail"]),
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
    drift_detected = bool(drift_payload.get("overall", {}).get("drift_detected")) if isinstance(drift_payload, dict) else None
    candidate_definition = {
        "candidate_id": "candidate_l_sparse_parent_bias_light",
        "base_candidate": "candidate_j_h_lite_retry2",
        "mechanism": "sparse_region_parent_bias_light",
        "default_changed": False,
        "promotion_status": "phase0_sanity",
        "allowed_use": "phase0_sanity_only",
        "disallowed_use": "default_replacement",
    }
    lines: list[str] = [
        "# NSGA-II Survivor-Pressure Phase 1 ZDT Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- 이번 작업의 목표: `candidate_l_sparse_parent_bias_light`의 ZDT Phase 1 반복 검증",
        f"- default drift audit 결과: `{('DRIFT DETECTED' if drift_detected else 'NO DRIFT') if drift_detected is not None else 'not available'}`",
        f"- candidate isolation 결과: `{'pass' if any(row['gate'] == 'candidate isolation' and row['result'] for row in payload['gate_rows']) else 'fail'}`",
        f"- 실행한 benchmark: {', '.join(payload['selected_problems'])}",
        f"- candidate_l vs candidate_j 핵심 결과: `{payload['phase1_decision']}` 이전 단계의 정성 해석은 paired summary에 기록",
        f"- fairness 결과: pass={payload['fairness_summary'].get('pass', 0)}, warning={payload['fairness_summary'].get('warning', 0)}, fail={payload['fairness_summary'].get('fail', 0)}",
        f"- Phase 1 decision: **{payload['phase1_decision']}**",
        "- 기본값 변경 여부: 없음",
        "- Level 판정 변화 여부: Level 판정 유지 또는 실험 툴킷 근거 강화 범위에서만 해석",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "- Scope: candidate_l_sparse_parent_bias_light, ZDT1/ZDT2/ZDT3, 10-seed repeated validation, candidate_l vs candidate_j, fairness check, default drift check",
        "- Non-Scope: default promotion, candidate_l change request, DTLZ/WFG validation, new survivor-pressure families, productization",
        "",
        "## 3. Default Drift and Candidate Isolation",
        "",
        *BASE._markdown_table(
            [
                {
                    "gate": "default metadata contamination",
                    "result": "pass" if drift_detected is False else "fail" if drift_detected else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "selection.py 변경 이후에도 default path에서 candidate metadata leak가 없어야 한다",
                },
                {
                    "gate": "default_changed=false",
                    "result": "pass" if any(row["gate"] == "default_changed=false" and row["result"] for row in payload["gate_rows"]) else "fail",
                    "evidence": "candidate_l metadata",
                    "interpretation": "candidate_l는 default_changed=false를 유지해야 한다",
                },
                {
                    "gate": "actual evaluations",
                    "result": "pass" if any(row["gate"] == "actual evaluations" and row["result"] for row in payload["gate_rows"]) else "fail",
                    "evidence": "phase1 raw_rows",
                    "interpretation": "candidate_l actual evaluations가 requested budget과 일치해야 한다",
                },
                {
                    "gate": "candidate isolation",
                    "result": "pass" if any(row["gate"] == "candidate isolation" and row["result"] for row in payload["gate_rows"]) else "fail",
                    "evidence": "default internal rows + candidate rows",
                    "interpretation": "explicit opt-in 경로에서만 candidate metadata가 나타나야 한다",
                },
                {
                    "gate": "selection.py drift",
                    "result": "pass" if drift_detected is False else "fail" if drift_detected else "n/a",
                    "evidence": payload.get("drift_audit_path") or "n/a",
                    "interpretation": "selection.py 변경이 default internal NSGA-II 결과를 바꾸지 않아야 한다",
                },
                {
                    "gate": "local baseline governance",
                    "result": "external command",
                    "evidence": "see regression check section",
                    "interpretation": "runner 밖에서 check_local_baseline command로 확인한다",
                },
            ],
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
                "problem",
                "algorithm",
                "mean_hv",
                "mean_distance",
                "mean_gd",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_duplicate_rate",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## 7. Paired Comparisons",
        "",
        *BASE._markdown_table(
            [
                row
                for row in payload["paired_rows"]
                if row["comparison"]
                in {
                    "candidate_l vs candidate_j",
                    "candidate_l vs candidate_h",
                    "candidate_l vs candidate_d",
                    "candidate_l vs pymoo",
                    "candidate_l vs deap",
                }
            ],
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
        "## 8. Candidate L Decision",
        "",
        f"- **{payload['phase1_decision']}**",
        "",
        "## 9. What We Learned",
        "",
        *[
            f"- {row['problem']}: {row['interpretation']}"
            for row in payload["problem_rows"]
        ],
        "- sparse parent bias는 spacing / nondominated_count를 겨냥했지만, HV / distance / IGD / coverage와 함께 읽어야 한다.",
        "- candidate_h보다 trade-off가 작아졌는지는 candidate_l vs candidate_h paired rows로 별도 확인한다.",
        "- pymoo / DEAP 대비 우세 주장으로 해석하지 않고, gap이 남으면 그대로 기록한다.",
        "",
        "## 10. Failures and Warnings",
        "",
        *BASE._markdown_table(
            payload["failures"] or [{"type": "none", "target": "none", "seed": None, "message": "none", "impact": "none", "action": "none"}],
            ["type", "target", "seed", "message", "impact", "action"],
        ),
        "",
        "## 11. Regression Check",
        "",
        *BASE._markdown_table(
            [
                {
                    "command": "python scripts/audit_nsga2_default_drift.py ...",
                    "result": "see drift artifact",
                    "note": payload.get("drift_audit_path") or "n/a",
                },
                {
                    "command": "python scripts/validate_nsga2_survivor_pressure_phase1.py ...",
                    "result": "success",
                    "note": "current run",
                },
            ],
            ["command", "result", "note"],
        ),
        "",
        "## 12. Maturity Impact",
        "",
        "- Level 판정 유지.",
        "- Phase 1은 candidate evidence이지 default algorithm maturity 상향 근거가 아니다.",
        "- fairness / isolation / default drift gate가 유지되면 실험 툴킷으로서 Level 4 근거는 강화될 수 있다.",
        "- pymoo / DEAP 대비 약점이 남아 있으면 범용 optimizer 성숙도 상향은 금지한다.",
        "",
        "## 13. Recommended Next Work",
        "",
        "1. Phase 1 passed이면 Phase 2 DTLZ planning 작성",
        "2. Phase 1 passed with trade-offs이면 candidate_l 설계 조정 여부 검토",
        "3. Hold이면 seed 수 또는 ZDT stress 재검토",
        "4. Reject이면 sparse parent bias family 폐기 또는 backlog로 회수",
        "5. candidate_j opt-in documentation 유지",
        "6. fairness checker single-objective runner 확장 검토",
        "7. constrained multi-objective contract",
        "8. checkpoint/resume",
        "9. parallel evaluation",
        "",
        f"이번 Phase 1 결과, candidate_l_sparse_parent_bias_light는 ZDT 계열에서 {payload['phase1_decision']} 신호를 보였고, candidate_j 대비 mixed signal 여부는 paired rows에 기록되었으며, 최종 판정은 {payload['phase1_decision']}이다.",
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


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(PROJECT_ROOT / args.config)
    selected_problems = [item.strip().lower() for item in str(args.problems).split(",") if item.strip()]
    selected_specs = [mo_candidate_suite_specs()[name] for name in selected_problems]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    variants = _candidate_variants()
    variant_map = {variant.candidate_id: variant for variant in variants}

    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    comparison_specs = [
        ("candidate_l_sparse_parent_bias_light", "candidate_j_h_lite_retry2", "candidate_l vs candidate_j"),
        ("candidate_l_sparse_parent_bias_light", "candidate_h_uniform_dedup_mutation_boost", "candidate_l vs candidate_h"),
        ("candidate_l_sparse_parent_bias_light", "candidate_d_uniform_crossover", "candidate_l vs candidate_d"),
        ("candidate_l_sparse_parent_bias_light", "internal_nsga2", "candidate_l vs internal baseline"),
        ("candidate_l_sparse_parent_bias_light", "pymoo_nsga2", "candidate_l vs pymoo"),
        ("candidate_l_sparse_parent_bias_light", "deap_nsga2", "candidate_l vs deap"),
        ("candidate_l_sparse_parent_bias_light", "random_pareto_archive", "candidate_l vs random archive"),
        ("candidate_j_h_lite_retry2", "pymoo_nsga2", "candidate_j vs pymoo"),
        ("candidate_h_uniform_dedup_mutation_boost", "candidate_j_h_lite_retry2", "candidate_h vs candidate_j"),
    ]

    for spec in selected_specs:
        config = build_problem_config(base_config, spec)
        config = _retarget_budget(config, args.budget)
        reference_front = reference_front_for_spec(spec, point_count=201)
        reference_point = list(spec.hv_reference_point)
        problem_output_root = output_root / spec.problem
        problem_output_root.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            results: list[ExternalMOComparatorResult] = [
                run_internal_nsga2(config, seed=seed, output_root=str(problem_output_root)),
                *[_candidate_result(config, variant, seed=seed, output_root=problem_output_root) for variant in variants],
            ]
            if not args.skip_pymoo:
                results.append(run_pymoo_nsga2(config, seed=seed, budget=args.budget))
            if not args.skip_deap:
                results.append(run_deap_nsga2(config, seed=seed, budget=args.budget))
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
                            "action": "review comparator/runtime failure before Phase 1 conclusion",
                        }
                    )

    fairness_payload = evaluate_parameter_fairness(
        raw_rows,
        benchmark_rows=benchmark_rows,
        candidate_rows=candidate_rows,
    )
    aggregate_rows = _aggregate_rows(raw_rows)
    paired_rows = _paired_rows(raw_rows, comparison_specs)
    gate_rows = _candidate_gate_rows(raw_rows, fairness_payload)
    problem_rows = _phase1_problem_rows(paired_rows, selected_problems)
    phase1_decision = _phase1_decision(gate_rows, fairness_payload, problem_rows)
    drift_audit_path = PROJECT_ROOT / args.drift_audit_json
    drift_payload = _load_drift_payload(drift_audit_path, selected_problems)

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
        "paired_rows": paired_rows,
        "gate_rows": gate_rows,
        "problem_rows": problem_rows,
        "phase1_decision": phase1_decision,
        "failures": failures,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
        "drift_audit_path": str(drift_audit_path) if drift_payload is not None else None,
        "drift_audit": drift_payload,
    }

    results_json = safe_artifact_path(
        artifact_root,
        "nsga2_survivor_pressure_phase1_results",
        args.artifact_suffix,
        ".json",
    )
    results_csv = safe_artifact_path(
        artifact_root,
        "nsga2_survivor_pressure_phase1_results",
        args.artifact_suffix,
        ".csv",
    )
    results_md = safe_artifact_path(
        artifact_root,
        "nsga2_survivor_pressure_phase1_results",
        args.artifact_suffix,
        ".md",
    )
    report_md = safe_artifact_path(
        artifact_root,
        "nsga2_survivor_pressure_phase1_report",
        args.artifact_suffix,
        ".md",
    )
    fairness_md = safe_artifact_path(
        artifact_root,
        "nsga2_survivor_pressure_phase1_fairness_report",
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
            "mean_gd",
            "mean_igd",
            "mean_spacing",
            "mean_coverage",
            "mean_nondominated_count",
            "mean_duplicate_rate",
            "mean_archive_duplicate_rate",
            "mean_objective_duplicate_rate",
            "mean_decision_duplicate_rate",
            "mean_unique_decision_count",
            "mean_unique_objective_count",
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
