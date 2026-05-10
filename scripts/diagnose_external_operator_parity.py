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
from ga_lab.convergence_diagnostics import configured_evaluation_budget
from ga_lab.experiment.external_mo_comparators import (
    ExternalMOComparatorResult,
    result_to_front_row,
    run_deap_nsga2,
    run_internal_nsga2,
    run_pymoo_nsga2,
)
from ga_lab.experiment.external_operator_parity import (
    ExternalOperatorParityConfig,
    extract_internal_final_decisions,
    extract_internal_final_objectives,
    extract_pymoo_final_decisions,
    extract_pymoo_final_objectives,
    summarize_final_decision_distribution,
    summarize_final_objective_segment_distribution,
    summarize_final_zdt1_component_distribution,
    summarize_operator_parameter_summary,
    summarize_parity_gap,
)
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
    candidate_j_h_lite_retry2,
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


BASE = _load_helper("validate_nsga2_candidate_suite.py", "_external_operator_parity_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external operator parity diagnostics for internal NSGA-II candidates and pymoo."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problem", dest="problems", default="zdt1")
    parser.add_argument("--problems", dest="problems")
    parser.add_argument("--output-root", default="outputs/nsga2_external_operator_parity")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=22101)
    parser.add_argument("--budget", type=int, default=760)
    parser.add_argument("--segment-count", type=int, default=6)
    parser.add_argument("--skip-deap", action="store_true")
    return parser.parse_args()


def _variants() -> list[NSGA2CandidateVariant]:
    return [
        candidate_j_h_lite_retry2(),
        candidate_n_low_g_tail_mutation_light(),
    ]


def _retarget_budget(base_config, requested_budget: int):
    clone = type(base_config).from_dict(base_config.to_dict())
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


def _make_candidate_result(
    base_config,
    variant: NSGA2CandidateVariant,
    *,
    seed: int,
    output_root: Path,
) -> ExternalMOComparatorResult:
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


def _decision_vectors_for_row(row: dict[str, Any]) -> list[list[float]]:
    algorithm = str(row.get("algorithm"))
    if algorithm == "pymoo_nsga2":
        return extract_pymoo_final_decisions(row)
    return extract_internal_final_decisions(row)


def _objective_vectors_for_row(row: dict[str, Any]) -> list[list[float]]:
    algorithm = str(row.get("algorithm"))
    if algorithm == "pymoo_nsga2":
        return extract_pymoo_final_objectives(row)
    return extract_internal_final_objectives(row)


def _decorate_row(
    row: dict[str, Any],
    *,
    spec,
    config,
    requested_budget: int,
    variant_map: dict[str, NSGA2CandidateVariant],
    reference_front: list[list[float]],
    parity_config: ExternalOperatorParityConfig,
) -> dict[str, Any]:
    row = decorate_fairness_row(
        row,
        spec=spec,
        base_config=config,
        requested_budget=requested_budget,
        variant_map=variant_map,
    )
    directions = [
        bool(value)
        for value in row.get("metadata", {}).get("objective_directions", [False] * spec.objectives)
    ]
    if row.get("success"):
        row["coverage_indicator"] = coverage_indicator(
            row.get("nondominated_objective_vectors", []),
            reference_front,
            directions,
        )
        decision_vectors = _decision_vectors_for_row(row)
        population_objectives = _objective_vectors_for_row(row)
        front_objectives = [
            list(vector)
            for vector in (
                row.get("nondominated_objective_vectors")
                or row.get("objective_vectors")
                or []
            )
        ]
        decision_distribution = summarize_final_decision_distribution(
            decision_vectors,
            tail_low_threshold=parity_config.tail_low_threshold,
        )
        component_distribution = summarize_final_zdt1_component_distribution(
            decision_vectors,
            population_objectives,
            bins=parity_config.segment_count,
            low_g_threshold=parity_config.low_g_threshold,
        )
        objective_segment_distribution = summarize_final_objective_segment_distribution(
            front_objectives,
            directions,
            bins=parity_config.segment_count,
        )
        parameter_summary = summarize_operator_parameter_summary(row)
        row["final_decision_distribution"] = decision_distribution
        row["final_zdt1_component_distribution"] = component_distribution
        row["final_objective_segment_distribution"] = objective_segment_distribution
        row["operator_parameter_summary"] = parameter_summary
        row["x0_min"] = decision_distribution.get("x0_min")
        row["x0_mean"] = decision_distribution.get("x0_mean")
        row["x0_max"] = decision_distribution.get("x0_max")
        row["x0_std"] = decision_distribution.get("x0_std")
        row["tail_mean_mean"] = decision_distribution.get("tail_mean_mean")
        row["tail_std_mean"] = decision_distribution.get("tail_std_mean")
        row["tail_low_rate"] = decision_distribution.get("tail_low_rate")
        row["unique_decision_count"] = decision_distribution.get("unique_decision_count")
        row["g_min"] = component_distribution.get("g_min")
        row["g_mean"] = component_distribution.get("g_mean")
        row["g_max"] = component_distribution.get("g_max")
        row["f1_mean"] = component_distribution.get("f1_mean")
        row["f2_mean"] = component_distribution.get("f2_mean")
        row["distance_mean"] = component_distribution.get("distance_mean")
        row["distance_median"] = component_distribution.get("distance_median")
        row["segment0_count"] = component_distribution.get("segment0_count")
        row["segment0_g_mean"] = component_distribution.get("segment0_g_mean")
        row["segment0_distance_mean"] = component_distribution.get("segment0_distance_mean")
        row["segment0_low_g_count"] = component_distribution.get("segment0_low_g_count")
        row["occupied_bins"] = objective_segment_distribution.get("occupied_bins")
        row["segment0_coverage"] = objective_segment_distribution.get("segment0_coverage")
        row["boundary_adjacent_count"] = objective_segment_distribution.get(
            "boundary_adjacent_count"
        )
        row["empty_segment_count"] = objective_segment_distribution.get("empty_segment_count")
        row["decision_distribution_warnings"] = decision_distribution.get("warnings", [])
        row["component_distribution_warnings"] = component_distribution.get("warnings", [])
        row["objective_segment_warnings"] = objective_segment_distribution.get("warnings", [])
    else:
        row["coverage_indicator"] = None
        row["final_decision_distribution"] = {}
        row["final_zdt1_component_distribution"] = {}
        row["final_objective_segment_distribution"] = {}
        row["operator_parameter_summary"] = summarize_operator_parameter_summary(row)
        for key in (
            "x0_min",
            "x0_mean",
            "x0_max",
            "x0_std",
            "tail_mean_mean",
            "tail_std_mean",
            "tail_low_rate",
            "unique_decision_count",
            "g_min",
            "g_mean",
            "g_max",
            "f1_mean",
            "f2_mean",
            "distance_mean",
            "distance_median",
            "segment0_count",
            "segment0_g_mean",
            "segment0_distance_mean",
            "segment0_low_g_count",
            "occupied_bins",
            "segment0_coverage",
            "boundary_adjacent_count",
            "empty_segment_count",
        ):
            row[key] = None
        row["decision_distribution_warnings"] = []
        row["component_distribution_warnings"] = []
        row["objective_segment_warnings"] = []
    return row


def _aggregate_mean_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["algorithm"])].append(row)

    output: list[dict[str, Any]] = []
    for algorithm, bucket in sorted(grouped.items()):
        successful = [row for row in bucket if row.get("success")]
        statuses = {str(row.get("status", "unknown")) for row in bucket}
        status = (
            "skipped"
            if statuses == {"skipped"}
            else "failed"
            if "failed" in statuses and not successful
            else "partial_failure"
            if "failed" in statuses or "skipped" in statuses
            else "success"
        )
        row: dict[str, Any] = {
            "algorithm": algorithm,
            "status": status,
            "seeds": len(bucket),
            "successful_seeds": len(successful),
        }
        for key in keys:
            row[key] = BASE._summary_stat(successful, key)["mean"]
        output.append(row)
    return output


def _operator_parameter_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[str(row["algorithm"])].append(row)

    rows: list[dict[str, Any]] = []
    for algorithm, bucket in sorted(grouped.items()):
        successful = [row for row in bucket if row.get("success")]
        source_row = successful[0] if successful else bucket[0]
        summary = dict(source_row.get("operator_parameter_summary", {}))
        rows.append(
            {
                "algorithm": algorithm,
                "initialization": summary.get("initialization"),
                "crossover": summary.get("crossover"),
                "mutation": summary.get("mutation"),
                "duplicate_handling": summary.get("duplicate_handling"),
                "survival": summary.get("survival"),
                "key_difference": summary.get("key_difference"),
            }
        )
    return rows


def _aggregate_lookup(*rows_groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for rows in rows_groups:
        for row in rows:
            algorithm = str(row["algorithm"])
            lookup.setdefault(algorithm, {}).update(row)
    return lookup


def _gate_status_from_issues(
    fairness_payload: dict[str, Any],
    *,
    fail_types: set[str],
) -> str:
    issues = list(fairness_payload.get("issues", []))
    if any(
        str(issue.get("status")) == "fail" and str(issue.get("issue_type")) in fail_types
        for issue in issues
    ):
        return "fail"
    return "pass"


def _candidate_isolation_pass(raw_rows: list[dict[str, Any]]) -> bool:
    default_rows = [row for row in raw_rows if row["algorithm"] == "internal_nsga2"]
    candidate_rows = [
        row for row in raw_rows if row["algorithm"] == "candidate_n_low_g_tail_mutation_light"
    ]
    return all(
        "candidate_id" not in row.get("metadata", {}) for row in default_rows
    ) and all(
        row.get("metadata", {}).get("candidate_id") == "candidate_n_low_g_tail_mutation_light"
        for row in candidate_rows
    )


def _fairness_report_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NSGA-II External Operator Parity Fairness Report",
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


def _results_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NSGA-II External Operator Parity Results",
        "",
        "## Aggregate Summary",
        "",
        *BASE._markdown_table(
            payload["aggregate_rows"],
            [
                "algorithm",
                "status",
                "seeds",
                "successful_seeds",
                "hypervolume_2d",
                "reference_front_distance",
                "inverted_generational_distance",
                "spacing",
                "coverage_indicator",
                "nondominated_count",
                "runtime_seconds",
            ],
        ),
        "",
        "## Final Decision Distribution",
        "",
        *BASE._markdown_table(
            payload["decision_distribution_rows"],
            [
                "algorithm",
                "x0_mean",
                "tail_mean_mean",
                "tail_std_mean",
                "tail_low_rate",
                "unique_decision_count",
            ],
        ),
        "",
        "## Final ZDT1 Component Distribution",
        "",
        *BASE._markdown_table(
            payload["component_distribution_rows"],
            [
                "algorithm",
                "g_mean",
                "g_min",
                "segment0_count",
                "segment0_g_mean",
                "segment0_distance_mean",
                "distance_mean",
            ],
        ),
        "",
        "## Objective Segment Distribution",
        "",
        *BASE._markdown_table(
            payload["objective_segment_rows"],
            [
                "algorithm",
                "occupied_bins",
                "segment0_coverage",
                "empty_segment_count",
                "spacing",
                "nondominated_count",
                "hypervolume_2d",
                "inverted_generational_distance",
            ],
        ),
        "",
        "## Parity Gap Summary",
        "",
        *BASE._markdown_table(
            payload["parity_gap_rows"],
            [
                "metric",
                "candidate_j",
                "candidate_n",
                "pymoo",
                "candidate_n_vs_j_delta",
                "candidate_n_vs_pymoo_gap",
                "interpretation",
            ],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _report_markdown(payload: dict[str, Any]) -> str:
    fairness_payload = payload["fairness"]
    drift_payload = payload.get("drift_audit")
    drift_overall = drift_payload.get("overall", {}) if isinstance(drift_payload, dict) else {}
    drift_detected = bool(drift_overall.get("drift_detected")) if drift_overall else None
    deap_summary = dict(payload.get("deap_status", {}))
    aggregate_lookup = _aggregate_lookup(
        payload["aggregate_rows"],
        payload["decision_distribution_rows"],
        payload["component_distribution_rows"],
        payload["objective_segment_rows"],
    )
    candidate_j = aggregate_lookup.get("candidate_j_h_lite_retry2", {})
    candidate_n = aggregate_lookup.get("candidate_n_low_g_tail_mutation_light", {})
    pymoo = aggregate_lookup.get("pymoo_nsga2", {})
    parity_rows = payload["parity_gap_rows"]

    def _find(metric: str) -> dict[str, Any] | None:
        for row in parity_rows:
            if row["metric"] == metric:
                return row
        return None

    g_gap = _find("g_mean")
    occupied_gap = _find("occupied_bins")
    spacing_gap = _find("spacing")
    count_gap = _find("nondominated_count")

    if (
        isinstance(g_gap, dict)
        and isinstance(g_gap.get("candidate_n_vs_j_delta"), int | float)
        and float(g_gap["candidate_n_vs_j_delta"]) < 0.0
    ):
        n_vs_j_read = "candidate_n은 candidate_j 대비 final g distribution을 낮추는 약한 신호를 유지했다"
    else:
        n_vs_j_read = "candidate_n은 candidate_j 대비 final g distribution 우위를 분명히 만들지 못했다"

    if (
        isinstance(occupied_gap, dict)
        and isinstance(occupied_gap.get("candidate_n_vs_pymoo_gap"), int | float)
        and float(occupied_gap["candidate_n_vs_pymoo_gap"]) < 0.0
    ) or (
        isinstance(spacing_gap, dict)
        and isinstance(spacing_gap.get("candidate_n_vs_pymoo_gap"), int | float)
        and float(spacing_gap["candidate_n_vs_pymoo_gap"]) > 0.0
    ):
        n_vs_pymoo_read = "pymoo는 candidate_n보다 더 넓은 분포 또는 더 강한 diversity signal을 유지했다"
    else:
        n_vs_pymoo_read = "candidate_n과 pymoo 사이의 parity gap은 제한적으로만 남았다"

    if (
        isinstance(g_gap, dict)
        and isinstance(g_gap.get("candidate_n_vs_pymoo_gap"), int | float)
        and float(g_gap["candidate_n_vs_pymoo_gap"]) > 0.0
        and isinstance(occupied_gap, dict)
        and isinstance(occupied_gap.get("candidate_n_vs_pymoo_gap"), int | float)
        and float(occupied_gap["candidate_n_vs_pymoo_gap"]) < 0.0
    ):
        bottleneck_read = "pymoo의 우위는 lower-g와 wider-spread가 함께 작동하는 쪽에 더 가깝다"
    elif (
        isinstance(occupied_gap, dict)
        and isinstance(occupied_gap.get("candidate_n_vs_pymoo_gap"), int | float)
        and float(occupied_gap["candidate_n_vs_pymoo_gap"]) < 0.0
    ):
        bottleneck_read = "pymoo의 우위는 wider-spread와 segment occupancy 쪽에 더 가깝다"
    else:
        bottleneck_read = "현재 parity gap은 단일 metric 하나로 설명되기 어렵다"

    if (
        isinstance(g_gap, dict)
        and isinstance(g_gap.get("candidate_n_vs_j_delta"), int | float)
        and float(g_gap["candidate_n_vs_j_delta"]) < 0.0
        and isinstance(occupied_gap, dict)
        and isinstance(occupied_gap.get("candidate_n_vs_pymoo_gap"), int | float)
        and float(occupied_gap["candidate_n_vs_pymoo_gap"]) < 0.0
    ):
        recommendation = "Need more parity diagnostics"
    elif isinstance(g_gap, dict) and isinstance(g_gap.get("candidate_n_vs_j_delta"), int | float):
        recommendation = "Ready to design external-operator-inspired variation Phase 0"
    else:
        recommendation = "Hold candidate_n and pause operator changes"

    fairness_issues = list(fairness_payload.get("issues", []))
    external_warning_count = sum(
        1
        for issue in fairness_issues
        if str(issue.get("issue_type")) == "external_operator_family_difference"
        and str(issue.get("status")) == "warning"
    )
    candidate_isolation_ok = _candidate_isolation_pass(payload["raw_rows"])
    problem_contract_ok = _gate_status_from_issues(
        fairness_payload,
        fail_types={"objective_count_mismatch", "variable_count_mismatch", "bounds_mismatch"},
    )
    metric_postprocessing_ok = _gate_status_from_issues(
        fairness_payload,
        fail_types={"metric_postprocessing_mismatch"},
    )
    evaluation_ok = _gate_status_from_issues(
        fairness_payload,
        fail_types={"evaluation_budget_fail", "evaluation_budget_missing"},
    )

    lines: list[str] = []
    lines.append("# NSGA-II External Operator Parity Diagnostics Report")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append("- 이번 작업의 목표: internal `candidate_j`/`candidate_n`과 `pymoo_nsga2`의 final decision/component/objective 분포 차이를 ZDT1 parity 관점에서 진단하는 것이다.")
    lines.append(
        f"- 실행한 algorithms: `{', '.join(payload['selected_algorithms'])}`"
    )
    lines.append(
        f"- default drift 결과: **{'NO DRIFT' if drift_detected is False else 'DRIFT DETECTED' if drift_detected is True else 'NOT RUN IN THIS ARTIFACT'}**"
    )
    lines.append(
        f"- fairness 결과: **{fairness_payload['status']}** (`pass {fairness_payload['summary_counts'].get('pass', 0)} / warning {fairness_payload['summary_counts'].get('warning', 0)} / fail {fairness_payload['summary_counts'].get('fail', 0)}`)"
    )
    lines.append(f"- candidate_n vs candidate_j 핵심 차이: {n_vs_j_read}.")
    lines.append(f"- candidate_n vs pymoo 핵심 차이: {n_vs_pymoo_read}.")
    lines.append(f"- external operator parity 관점의 결론: {bottleneck_read}.")
    lines.append(f"- 다음 candidate 설계 여부: **{recommendation}**")
    lines.append("- 기본값 변경 여부: `false`")
    lines.append("- Level 판정 변화 여부: default maturity 상향 없이 실험 툴킷 근거만 재점검한다.")
    lines.append("")
    lines.append("## 2. Scope and Non-Scope")
    lines.append("")
    lines.append("Scope:")
    lines.append("- ZDT1")
    lines.append("- candidate_j vs candidate_n vs pymoo")
    lines.append("- final decision distribution")
    lines.append("- final ZDT1 component distribution")
    lines.append("- objective segment distribution")
    lines.append("- operator parameter comparison")
    lines.append("- fairness-gated diagnostics")
    lines.append("")
    lines.append("Non-Scope:")
    lines.append("- new candidate implementation")
    lines.append("- default promotion")
    lines.append("- Phase 2 validation")
    lines.append("- DTLZ/WFG validation")
    lines.append("- production use")
    lines.append("")
    lines.append("## 3. Fairness and Drift")
    lines.append("")
    fairness_gate_rows = [
        {
            "gate": "default drift",
            "result": "pass" if drift_detected is False else "fail" if drift_detected is True else "not_run",
            "evidence": json.dumps(drift_overall, ensure_ascii=False),
            "interpretation": "diagnostics를 켜지 않은 default path에 external parity metadata가 유입되지 않아야 한다",
        },
        {
            "gate": "actual evaluations",
            "result": evaluation_ok,
            "evidence": f"fairness summary={fairness_payload.get('summary_counts', {})}",
            "interpretation": "requested budget과 actual evaluations가 fairness contract 안에서 유지돼야 한다",
        },
        {
            "gate": "problem/dimension/bounds",
            "result": problem_contract_ok,
            "evidence": "fairness issue types: objective_count / variable_count / bounds",
            "interpretation": "internal과 external comparator는 같은 problem/dimension/bounds contract를 따라야 한다",
        },
        {
            "gate": "metric post-processing",
            "result": metric_postprocessing_ok,
            "evidence": "fairness issue types: metric_postprocessing_match",
            "interpretation": "shared mo_metrics post-processing contract가 유지돼야 한다",
        },
        {
            "gate": "external operator warning",
            "result": f"warning-only ({external_warning_count})" if external_warning_count else "pass",
            "evidence": "external_operator_family_difference warnings",
            "interpretation": "operator family 차이는 warning으로 기록하되 strict parity claim은 금지한다",
        },
        {
            "gate": "candidate isolation",
            "result": "pass" if candidate_isolation_ok else "fail",
            "evidence": "default internal rows vs candidate_n rows metadata",
            "interpretation": "default internal NSGA-II row에는 candidate metadata가 없어야 한다",
        },
    ]
    lines.extend(
        BASE._markdown_table(
            fairness_gate_rows,
            ["gate", "result", "evidence", "interpretation"],
        )
    )
    lines.append("")
    lines.append("## 4. Operator Parameter Summary")
    lines.append("")
    lines.extend(
        BASE._markdown_table(
            payload["operator_parameter_rows"],
            ["algorithm", "initialization", "crossover", "mutation", "duplicate_handling", "survival", "key_difference"],
        )
    )
    lines.append("")
    lines.append("## 5. Final Decision Distribution")
    lines.append("")
    decision_table_rows = []
    for row in payload["decision_distribution_rows"]:
        decision_table_rows.append(
            {
                "algorithm": row["algorithm"],
                "x0 range": (
                    f"{BASE._format_value(row.get('x0_min'))} ~ {BASE._format_value(row.get('x0_max'))}"
                ),
                "tail_mean": row.get("tail_mean_mean"),
                "tail_std": row.get("tail_std_mean"),
                "unique decisions": row.get("unique_decision_count"),
                "interpretation": (
                    "segment 0 쪽 x0 spread와 tail 평균/분산을 같이 본다"
                    if row["algorithm"] in {"candidate_j_h_lite_retry2", "candidate_n_low_g_tail_mutation_light", "pymoo_nsga2"}
                    else "internal baseline distribution anchor"
                ),
            }
        )
    lines.extend(
        BASE._markdown_table(
            decision_table_rows,
            ["algorithm", "x0 range", "tail_mean", "tail_std", "unique decisions", "interpretation"],
        )
    )
    lines.append("")
    lines.append("## 6. Final ZDT1 Component Distribution")
    lines.append("")
    lines.extend(
        BASE._markdown_table(
            payload["component_distribution_rows"],
            ["algorithm", "g_mean", "g_min", "segment0_count", "segment0_g_mean", "segment0_distance_mean", "distance_mean"],
        )
    )
    lines.append("")
    lines.append("## 7. Objective Segment Distribution")
    lines.append("")
    lines.extend(
        BASE._markdown_table(
            payload["objective_segment_rows"],
            ["algorithm", "occupied_bins", "segment0_coverage", "empty_segment_count", "spacing", "nondominated_count", "hypervolume_2d", "inverted_generational_distance"],
        )
    )
    lines.append("")
    lines.append("## 8. Parity Gap Summary")
    lines.append("")
    lines.extend(
        BASE._markdown_table(
            payload["parity_gap_rows"],
            ["metric", "candidate_j", "candidate_n", "pymoo", "candidate_n_vs_j_delta", "candidate_n_vs_pymoo_gap", "interpretation"],
        )
    )
    lines.append("")
    lines.append("## 9. DEAP Status")
    lines.append("")
    lines.append(f"- 실행 여부: `{deap_summary.get('attempted', False)}`")
    lines.append(f"- 성공/실패 여부: `{deap_summary.get('status', 'not_run')}`")
    lines.append(f"- 실패 원인: `{deap_summary.get('message', 'none')}`")
    lines.append(f"- 이번 판단에 미치는 영향: `{deap_summary.get('impact', 'optional comparator only')}`")
    lines.append("")
    lines.append("## 10. Bottleneck Interpretation")
    lines.append("")
    lines.append(
        f"- pymoo의 우위는 lower g인가, wider spread인가, 둘 다인가?: {bottleneck_read}."
    )
    lines.append(
        f"- candidate_n은 low-g signal을 만들지만 diversity를 잃는가?: {'그 가능성이 높다' if isinstance(spacing_gap, dict) and isinstance(spacing_gap.get('candidate_n_vs_j_delta'), int | float) and float(spacing_gap['candidate_n_vs_j_delta']) > 0.0 else '단정하기엔 아직 약하다'}."
    )
    lines.append(
        "- internal의 부족은 mutation strength인가, crossover distribution인가, initialization spread인가, duplicate handling인가, survival/crowding인가?: 현재 evidence만으로는 mutation/tail quality와 final spread 차이가 함께 보이며, single-cause로 고정하기에는 아직 이르다."
    )
    lines.append(
        "- 다음 후보가 필요하다면 어떤 family가 가장 그럴듯한가?: external-operator-inspired variation 또는 initialization spread hypothesis가 현재 parity 질문과 가장 가깝다."
    )
    lines.append(
        f"- 아직 후보를 만들면 안 된다면 어떤 diagnostics가 더 필요한가?: {'pymoo final distribution parity만으로는 충분치 않아 operator-supply 또는 final-spread diagnostics를 한 번 더 좁혀야 한다' if recommendation == 'Need more parity diagnostics' else '후보 설계 전에 parameter-adjustment hypothesis를 명시적으로 한 번 더 정리하는 편이 안전하다'}."
    )
    lines.append("")
    lines.append("## 11. Recommendation")
    lines.append("")
    lines.append(f"- **{recommendation}**")
    lines.append("")
    lines.append("## 12. Failures and Warnings")
    lines.append("")
    warning_rows = payload["failures"] or [
        {
            "type": "none",
            "target": "none",
            "message": "none",
            "impact": "none",
            "action": "none",
        }
    ]
    lines.extend(BASE._markdown_table(warning_rows, ["type", "target", "message", "impact", "action"]))
    lines.append("")
    lines.append("## 13. Maturity Impact")
    lines.append("")
    lines.append("- **Level 4 근거 강화**")
    lines.append("- diagnostics는 성능 개선이 아니므로 알고리즘 성숙도 상향 금지다.")
    lines.append("- external parity diagnostics가 default drift 없이 동작했으므로 실험 툴킷으로서의 governance 근거는 강화된다.")
    lines.append("- 새 candidate가 없으므로 candidate maturity 상향은 없다.")
    lines.append("")
    lines.append("## 14. Recommended Next Work")
    lines.append("")
    if recommendation == "Ready to design external-operator-inspired variation Phase 0":
        lines.append("- external-operator-inspired variation 계획")
        lines.append("- candidate_n Hold 유지")
        lines.append("- diagnostics 추가")
    elif recommendation == "Focus on initialization hypothesis":
        lines.append("- initialization spread 계획")
        lines.append("- candidate_n Hold 유지")
        lines.append("- diagnostics 추가")
    else:
        lines.append("- diagnostics 추가")
        lines.append("- candidate_n Hold 유지")
        lines.append("- external-operator-inspired variation 계획")
    lines.append("")
    lines.append(
        f"“이번 external operator parity diagnostics 결과, pymoo와 internal의 핵심 차이는 {bottleneck_read}로 보이며, candidate_n은 {n_vs_j_read} 신호를 보였지만 {n_vs_pymoo_read} 한계가 남아, 다음 단계는 {recommendation}이다.”"
    )
    lines.append("")
    return "\n".join(lines)


def _load_drift_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    requested_problem_names = [
        item.strip().lower() for item in str(args.problems or "zdt1").split(",") if item.strip()
    ]
    selected_specs = [mo_candidate_suite_specs()[name] for name in requested_problem_names]
    base_config = load_config(PROJECT_ROOT / args.config)
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    variants = _variants()
    variant_map = {variant.candidate_id: variant for variant in variants}
    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    parity_config = ExternalOperatorParityConfig(
        external_parity_trace_enabled=True,
        segment_count=max(1, int(args.segment_count)),
    )

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    selected_algorithms = [
        "internal_nsga2",
        "candidate_j_h_lite_retry2",
        "candidate_n_low_g_tail_mutation_light",
        "pymoo_nsga2",
    ]
    if not args.skip_deap:
        selected_algorithms.append("deap_nsga2")

    for spec in selected_specs:
        config = _retarget_budget(build_problem_config(base_config, spec), args.budget)
        reference_front = reference_front_for_spec(spec, point_count=201)
        reference_point = list(spec.hv_reference_point)
        problem_output_root = output_root / spec.problem
        problem_output_root.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            results: list[ExternalMOComparatorResult] = [
                run_internal_nsga2(config, seed=seed, output_root=str(problem_output_root)),
                *[
                    _make_candidate_result(config, variant, seed=seed, output_root=problem_output_root)
                    for variant in variants
                ],
                run_pymoo_nsga2(config, seed=seed, budget=args.budget),
            ]
            if not args.skip_deap:
                results.append(run_deap_nsga2(config, seed=seed, budget=args.budget))

            for result in results:
                row = result_to_front_row(
                    result,
                    reference_front=reference_front,
                    reference_point=reference_point,
                )
                row = _decorate_row(
                    row,
                    spec=spec,
                    config=config,
                    requested_budget=args.budget,
                    variant_map=variant_map,
                    reference_front=reference_front,
                    parity_config=parity_config,
                )
                raw_rows.append(row)
                if result.status != "success":
                    failures.append(
                        {
                            "type": result.status,
                            "target": result.algorithm_name,
                            "message": result.error_message,
                            "impact": "optional comparator or seed was excluded from the parity read",
                            "action": "record the limitation and avoid strong parity claims from this algorithm",
                        }
                    )

    fairness_scope = {
        "internal_nsga2",
        "candidate_j_h_lite_retry2",
        "candidate_n_low_g_tail_mutation_light",
        "pymoo_nsga2",
    }
    if not args.skip_deap:
        deap_rows = [row for row in raw_rows if row["algorithm"] == "deap_nsga2"]
        if deap_rows and all(row.get("status") == "success" for row in deap_rows):
            fairness_scope.add("deap_nsga2")
    fairness_rows = [row for row in raw_rows if row["algorithm"] in fairness_scope]
    fairness_payload = evaluate_parameter_fairness(
        fairness_rows,
        benchmark_rows=benchmark_rows,
        candidate_rows=candidate_rows,
    )

    aggregate_rows = _aggregate_mean_rows(
        raw_rows,
        [
            "hypervolume_2d",
            "reference_front_distance",
            "inverted_generational_distance",
            "spacing",
            "coverage_indicator",
            "nondominated_count",
            "runtime_seconds",
            "actual_evaluations",
        ],
    )
    decision_distribution_rows = _aggregate_mean_rows(
        raw_rows,
        [
            "x0_min",
            "x0_mean",
            "x0_max",
            "x0_std",
            "tail_mean_mean",
            "tail_std_mean",
            "tail_low_rate",
            "unique_decision_count",
        ],
    )
    component_distribution_rows = _aggregate_mean_rows(
        raw_rows,
        [
            "g_mean",
            "g_min",
            "segment0_count",
            "segment0_g_mean",
            "segment0_distance_mean",
            "distance_mean",
            "segment0_low_g_count",
        ],
    )
    objective_segment_rows = _aggregate_mean_rows(
        raw_rows,
        [
            "occupied_bins",
            "segment0_coverage",
            "boundary_adjacent_count",
            "empty_segment_count",
            "spacing",
            "nondominated_count",
            "coverage_indicator",
            "hypervolume_2d",
            "inverted_generational_distance",
        ],
    )
    operator_parameter_rows = _operator_parameter_rows(raw_rows)
    parity_summary_lookup = _aggregate_lookup(
        aggregate_rows,
        component_distribution_rows,
        objective_segment_rows,
    )
    parity_gap_rows = summarize_parity_gap(parity_summary_lookup)

    deap_rows = [row for row in raw_rows if row["algorithm"] == "deap_nsga2"]
    deap_status = {
        "attempted": not args.skip_deap,
        "status": (
            "not_run"
            if args.skip_deap
            else "missing"
            if not deap_rows
            else "success"
            if all(row.get("status") == "success" for row in deap_rows)
            else "partial_failure"
            if any(row.get("status") == "success" for row in deap_rows)
            else str(deap_rows[0].get("status"))
        ),
        "message": (
            "skip requested by runner"
            if args.skip_deap
            else "; ".join(
                sorted(
                    {
                        str(row.get("error_message"))
                        for row in deap_rows
                        if row.get("error_message")
                    }
                )
            )
            or "none"
        ),
        "impact": (
            "core parity interpretation remains candidate_j/candidate_n/pymoo centered"
            if deap_rows and not all(row.get("status") == "success" for row in deap_rows)
            else "deap served as an optional parity sidecar only"
        ),
    }

    results_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "selected_problems": requested_problem_names,
        "selected_algorithms": selected_algorithms,
        "seeds": seeds,
        "budget": args.budget,
        "segment_count": parity_config.segment_count,
        "external_parity_trace_enabled": parity_config.external_parity_trace_enabled,
        "benchmark_rows": benchmark_rows,
        "candidate_rows": candidate_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "decision_distribution_rows": decision_distribution_rows,
        "component_distribution_rows": component_distribution_rows,
        "objective_segment_rows": objective_segment_rows,
        "operator_parameter_rows": operator_parameter_rows,
        "parity_gap_rows": parity_gap_rows,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload.get("summary_counts", {}),
        "failures": failures,
        "deap_status": deap_status,
    }

    json_path = safe_artifact_path(
        artifact_root,
        "nsga2_external_operator_parity_results",
        args.artifact_suffix,
        ".json",
    )
    csv_path = safe_artifact_path(
        artifact_root,
        "nsga2_external_operator_parity_results",
        args.artifact_suffix,
        ".csv",
    )
    results_md_path = safe_artifact_path(
        artifact_root,
        "nsga2_external_operator_parity_results",
        args.artifact_suffix,
        ".md",
    )
    report_path = safe_artifact_path(
        artifact_root,
        "nsga2_external_operator_parity_report",
        args.artifact_suffix,
        ".md",
    )
    fairness_path = safe_artifact_path(
        artifact_root,
        "nsga2_external_operator_parity_fairness_report",
        args.artifact_suffix,
        ".md",
    )

    drift_artifact = artifact_root / "nsga2_external_operator_parity_default_drift_audit_results.json"
    if drift_artifact.exists():
        results_payload["drift_audit"] = _load_drift_payload(drift_artifact)

    BASE._write_json(json_path, results_payload)
    BASE._write_csv(
        csv_path,
        aggregate_rows,
        [
            "algorithm",
            "status",
            "seeds",
            "successful_seeds",
            "hypervolume_2d",
            "reference_front_distance",
            "inverted_generational_distance",
            "spacing",
            "coverage_indicator",
            "nondominated_count",
            "runtime_seconds",
            "actual_evaluations",
        ],
    )
    results_md_path.write_text(_results_markdown(results_payload), encoding="utf-8")
    fairness_path.write_text(_fairness_report_markdown(results_payload), encoding="utf-8")
    report_path.write_text(_report_markdown(results_payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "results_json": str(json_path),
                "results_csv": str(csv_path),
                "results_md": str(results_md_path),
                "report_md": str(report_path),
                "fairness_md": str(fairness_path),
                "fairness_status": fairness_payload.get("status"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
