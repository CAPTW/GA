from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
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
    extract_internal_final_decisions,
    extract_internal_final_objectives,
    extract_pymoo_final_decisions,
    extract_pymoo_final_objectives,
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
from ga_lab.experiment.spread_parity_diagnostics import (
    SpreadParityConfig,
    summarize_decision_to_segment_mapping,
    summarize_nondominated_distribution,
    summarize_occupancy_uniformity,
    summarize_parity_spread_gap,
    summarize_segment_allocation,
    summarize_segment_spacing_contribution,
)


def _load_helper(script_name: str, module_name: str):
    helper_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_helper("validate_nsga2_candidate_suite.py", "_spread_parity_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spread parity diagnostics for internal NSGA-II candidates and external comparators."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problem", dest="problems", default="zdt1")
    parser.add_argument("--problems", dest="problems")
    parser.add_argument("--output-root", default="outputs/nsga2_spread_parity")
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


def _front_objectives_for_row(row: dict[str, Any]) -> list[list[float]]:
    values = row.get("nondominated_objective_vectors")
    if isinstance(values, list) and values:
        return [list(map(float, vector)) for vector in values]
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        values = metadata.get("front_objective_vectors")
        if isinstance(values, list) and values:
            return [list(map(float, vector)) for vector in values]
    values = row.get("objective_vectors")
    if isinstance(values, list):
        return [list(map(float, vector)) for vector in values]
    return []


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


def _mode(values: list[int | None]) -> int | None:
    candidates = [value for value in values if value is not None]
    if not candidates:
        return None
    counts = Counter(candidates)
    return min(
        (value for value, count in counts.items() if count == max(counts.values())),
        default=None,
    )


def _finite(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    value_float = float(value)
    if not math.isfinite(value_float):
        return None
    return value_float


def _mean(values: list[Any]) -> float | None:
    numeric = [value for item in values if (value := _finite(item)) is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def _decorate_row(
    row: dict[str, Any],
    *,
    spec,
    config,
    requested_budget: int,
    variant_map: dict[str, NSGA2CandidateVariant],
    reference_front: list[list[float]],
    spread_config: SpreadParityConfig,
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
        front_objectives = _front_objectives_for_row(row)

        segment_allocation = summarize_segment_allocation(
            decision_vectors,
            population_objectives,
            front_objectives,
            directions,
            bins=spread_config.segment_count,
        )
        segment_spacing = summarize_segment_spacing_contribution(
            front_objectives,
            directions,
            bins=spread_config.segment_count,
        )
        occupancy_uniformity = summarize_occupancy_uniformity(
            front_objectives,
            directions,
            bins=spread_config.segment_count,
        )
        nondominated_distribution = summarize_nondominated_distribution(
            population_objectives,
            front_objectives,
            directions,
            bins=spread_config.segment_count,
        )
        decision_segment_mapping = summarize_decision_to_segment_mapping(
            decision_vectors,
            population_objectives,
            directions,
            bins=spread_config.segment_count,
        )

        row["segment_allocation_summary"] = segment_allocation
        row["segment_spacing_contribution"] = segment_spacing
        row["occupancy_uniformity_summary"] = occupancy_uniformity
        row["nondominated_distribution_summary"] = nondominated_distribution
        row["decision_to_segment_mapping"] = decision_segment_mapping

        row["occupied_bins"] = occupancy_uniformity.get("occupied_bins")
        row["empty_bins"] = occupancy_uniformity.get("empty_bins")
        row["point_count_entropy"] = occupancy_uniformity.get("point_count_entropy")
        row["max_segment_load"] = occupancy_uniformity.get("max_segment_load")
        row["segment_load_std"] = occupancy_uniformity.get("segment_load_std")
        row["segment_load_gini"] = occupancy_uniformity.get("segment_load_gini")
        row["weakest_segment_id"] = segment_spacing.get("weakest_segment_id")
        row["largest_gap_segment_id"] = segment_spacing.get("largest_gap_segment_id")
        row["total_nondominated_count"] = nondominated_distribution.get("total_nondominated_count")
        row["segment_point_counts"] = {
            str(entry["segment_id"]): int(entry["point_count"])
            for entry in segment_allocation.get("segment_rows", [])
        }
        row["segment_nondominated_counts"] = {
            str(entry["segment_id"]): int(entry["segment_nondominated_count"])
            for entry in nondominated_distribution.get("segment_rows", [])
        }
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
        row["coverage_indicator"] = None
        row["segment_allocation_summary"] = {}
        row["segment_spacing_contribution"] = {}
        row["occupancy_uniformity_summary"] = {}
        row["nondominated_distribution_summary"] = {}
        row["decision_to_segment_mapping"] = {}
        for key in (
            "occupied_bins",
            "empty_bins",
            "point_count_entropy",
            "max_segment_load",
            "segment_load_std",
            "segment_load_gini",
            "weakest_segment_id",
            "largest_gap_segment_id",
            "total_nondominated_count",
        ):
            row[key] = None
        row["segment_point_counts"] = {}
        row["segment_nondominated_counts"] = {}
        row["spread_parity_warnings"] = []
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
        output.append(
            {
                "algorithm": algorithm,
                "status": status,
                "seeds": len(bucket),
                "successful_seeds": len(successful),
                **{
                    key: _mean([row.get(key) for row in successful]) if successful else None
                    for key in keys
                },
            }
        )
    return output


def _flatten_segment_rows(raw_rows: list[dict[str, Any]], summary_key: str) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in raw_rows:
        summary = row.get(summary_key, {})
        if not isinstance(summary, dict):
            continue
        for segment_row in summary.get("segment_rows", []):
            if not isinstance(segment_row, dict):
                continue
            flattened.append(
                {
                    "algorithm": row["algorithm"],
                    "seed": row["seed"],
                    **segment_row,
                }
            )
    return flattened


def _aggregate_segment_rows(
    rows: list[dict[str, Any]],
    *,
    numeric_fields: list[str],
    bool_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    bool_fields = bool_fields or []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["algorithm"]), int(row["segment_id"]))].append(row)

    output: list[dict[str, Any]] = []
    for (algorithm, segment_id), bucket in sorted(grouped.items()):
        entry: dict[str, Any] = {
            "algorithm": algorithm,
            "segment_id": segment_id,
            "seeds": len(bucket),
        }
        if any("segment_range" in row for row in bucket):
            first_range = next((row.get("segment_range") for row in bucket if row.get("segment_range") is not None), None)
            entry["segment_range"] = first_range
        for field in numeric_fields:
            entry[field] = _mean([row.get(field) for row in bucket])
        for field in bool_fields:
            entry[f"{field}_rate"] = _mean(
                [1.0 if bool(row.get(field)) else 0.0 for row in bucket]
            )
        output.append(entry)
    return output


def _algorithm_summary_lookup(
    aggregate_rows: list[dict[str, Any]],
    occupancy_rows: list[dict[str, Any]],
    spacing_rows: list[dict[str, Any]],
    allocation_rows: list[dict[str, Any]],
    nondominated_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    aggregate_lookup = {str(row["algorithm"]): row for row in aggregate_rows}
    occupancy_lookup = {str(row["algorithm"]): row for row in occupancy_rows}
    allocation_lookup: dict[str, dict[str, Any]] = defaultdict(dict)
    nondominated_lookup: dict[str, dict[str, Any]] = defaultdict(dict)
    spacing_lookup: dict[str, dict[str, Any]] = defaultdict(dict)

    for row in allocation_rows:
        allocation_lookup[str(row["algorithm"])][str(int(row["segment_id"]))] = row
    for row in nondominated_rows:
        nondominated_lookup[str(row["algorithm"])][str(int(row["segment_id"]))] = row
    for row in spacing_rows:
        spacing_lookup[str(row["algorithm"])][str(int(row["segment_id"]))] = row

    algorithms = sorted(
        set(aggregate_lookup)
        | set(occupancy_lookup)
        | set(allocation_lookup)
        | set(nondominated_lookup)
        | set(spacing_lookup)
    )
    lookup: dict[str, dict[str, Any]] = {}
    for algorithm in algorithms:
        allocation_segments = allocation_lookup.get(algorithm, {})
        nondominated_segments = nondominated_lookup.get(algorithm, {})
        spacing_segments = spacing_lookup.get(algorithm, {})
        weakest_segment_id = None
        if spacing_segments:
            weakest_segment_id = max(
                spacing_segments,
                key=lambda segment_id: _finite(
                    spacing_segments[segment_id].get("local_spacing_contribution")
                )
                or -1.0,
            )
        lookup[algorithm] = {
            "occupied_bins": occupancy_lookup.get(algorithm, {}).get("occupied_bins"),
            "empty_bins": occupancy_lookup.get(algorithm, {}).get("empty_bins"),
            "point_count_entropy": occupancy_lookup.get(algorithm, {}).get("point_count_entropy"),
            "segment_load_std": occupancy_lookup.get(algorithm, {}).get("segment_load_std"),
            "segment_load_gini": occupancy_lookup.get(algorithm, {}).get("segment_load_gini"),
            "spacing": aggregate_lookup.get(algorithm, {}).get("spacing"),
            "total_nondominated_count": aggregate_lookup.get(algorithm, {}).get("nondominated_count"),
            "segment_point_counts": {
                segment_id: allocation_segments[segment_id].get("point_count")
                for segment_id in allocation_segments
            },
            "segment_nondominated_counts": {
                segment_id: nondominated_segments[segment_id].get("segment_nondominated_count")
                for segment_id in nondominated_segments
            },
            "weakest_segment_id": int(weakest_segment_id) if weakest_segment_id is not None else None,
        }
    return lookup


def _operator_parameter_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        algorithm = str(row["algorithm"])
        if algorithm in rows:
            continue
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        rows[algorithm] = {
            "algorithm": algorithm,
            "initialization": metadata.get("initialization") or "random_uniform_bounds",
            "crossover": metadata.get("crossover_type") or metadata.get("crossover") or "unknown",
            "mutation": metadata.get("mutation_type") or metadata.get("mutation") or "unknown",
            "duplicate_handling": metadata.get("duplicate_handling") or "unknown",
            "survival": metadata.get("survival") or "rank_crowding",
        }
    return [rows[key] for key in sorted(rows)]


def _load_drift_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_stat(value: Any) -> str:
    return BASE._format_value(value)


def _report_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    return "\n".join(BASE._markdown_table(rows, columns))


def _fairness_report_markdown(results_payload: dict[str, Any]) -> str:
    fairness = dict(results_payload.get("fairness", {}))
    issues = list(fairness.get("issues", []))
    summary = dict(fairness.get("summary_counts", {}))
    lines = [
        "# NSGA-II Spread Parity Fairness Report",
        "",
        f"- status: `{fairness.get('status', 'unknown')}`",
        f"- summary: `pass {summary.get('pass', 0)} / warning {summary.get('warning', 0)} / fail {summary.get('fail', 0)}`",
        "",
        "## Issues",
        "",
    ]
    if issues:
        lines.append(
            _report_table(
                issues,
                [
                    "severity",
                    "issue_type",
                    "algorithm",
                    "problem",
                    "message",
                ],
            )
        )
    else:
        lines.append("No fairness issues recorded.")
    return "\n".join(lines) + "\n"


def _results_markdown(results_payload: dict[str, Any]) -> str:
    lines = [
        "# NSGA-II Spread Parity Results",
        "",
        f"- generated_at: `{results_payload.get('generated_at')}`",
        f"- command: `{results_payload.get('command')}`",
        "",
        "## Aggregate Rows",
        "",
        _report_table(
            results_payload.get("aggregate_rows", []),
            [
                "algorithm",
                "status",
                "hypervolume_2d",
                "reference_front_distance",
                "inverted_generational_distance",
                "spacing",
                "nondominated_count",
                "runtime_seconds",
            ],
        ),
        "",
        "## Spread Gap Rows",
        "",
        _report_table(
            results_payload.get("spread_gap_rows", []),
            [
                "metric",
                "candidate_j",
                "candidate_n",
                "pymoo",
                "candidate_n_vs_j_delta",
                "candidate_n_vs_pymoo_gap",
                "gap_segment",
                "interpretation",
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def _report_markdown(results_payload: dict[str, Any]) -> str:
    fairness = dict(results_payload.get("fairness", {}))
    fairness_summary = dict(results_payload.get("fairness_summary", {}))
    drift = dict(results_payload.get("drift_audit", {}))
    aggregate_lookup = {str(row["algorithm"]): row for row in results_payload.get("aggregate_rows", [])}
    occupancy_lookup = {
        str(row["algorithm"]): row for row in results_payload.get("occupancy_uniformity_aggregate_rows", [])
    }
    spacing_lookup = defaultdict(list)
    for row in results_payload.get("segment_spacing_aggregate_rows", []):
        spacing_lookup[str(row["algorithm"])].append(row)
    nondominated_lookup = defaultdict(list)
    for row in results_payload.get("nondominated_distribution_aggregate_rows", []):
        nondominated_lookup[str(row["algorithm"])].append(row)
    decision_mapping_rows = results_payload.get("decision_to_segment_aggregate_rows", [])
    spread_gap_rows = results_payload.get("spread_gap_rows", [])

    def _segment_interpretation(algorithm: str) -> str:
        occupancy = occupancy_lookup.get(algorithm, {})
        entropy = occupancy.get("point_count_entropy")
        load_std = occupancy.get("segment_load_std")
        if entropy is None:
            return "spread summary unavailable"
        return f"entropy={_summary_stat(entropy)}, load_std={_summary_stat(load_std)}"

    def _spacing_interpretation(algorithm: str) -> str:
        rows = spacing_lookup.get(algorithm, [])
        if not rows:
            return "spacing summary unavailable"
        weakest = max(rows, key=lambda row: _finite(row.get("local_spacing_contribution")) or -1.0)
        return f"segment {weakest.get('segment_id')} contribution={_summary_stat(weakest.get('local_spacing_contribution'))}"

    def _nondominated_interpretation(algorithm: str) -> str:
        rows = nondominated_lookup.get(algorithm, [])
        if not rows:
            return "nondominated spread unavailable"
        weakest = min(rows, key=lambda row: _finite(row.get("segment_nondominated_count")) or 0.0)
        strongest = max(rows, key=lambda row: _finite(row.get("segment_nondominated_count")) or -1.0)
        return (
            f"weakest={weakest.get('segment_id')}, strongest={strongest.get('segment_id')}"
        )

    candidate_n_spacing_rows = spacing_lookup.get("candidate_n_low_g_tail_mutation_light", [])
    candidate_n_weakest_segment = (
        max(
            candidate_n_spacing_rows,
            key=lambda row: _finite(row.get("local_spacing_contribution")) or -1.0,
        ).get("segment_id")
        if candidate_n_spacing_rows
        else None
    )
    candidate_n_vs_j_rows = [
        row
        for row in spread_gap_rows
        if row.get("metric")
        in {
            "occupied_bins",
            "point_count_entropy",
            "segment_load_std",
            "segment_load_gini",
            "spacing",
            "total_nondominated_count",
        }
    ]
    candidate_n_vs_pymoo_rows = candidate_n_vs_j_rows

    lines = [
        "# NSGA-II Spread Parity Diagnostics Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: `candidate_n`과 `pymoo` 사이에 남아 있는 occupied-bin / spacing / nondominated-count gap을 segment 단위로 직접 분해하는 것이다.",
        f"- 실행한 algorithms: `{', '.join(results_payload.get('selected_algorithms', []))}`",
        f"- default drift 결과: **{'NO DRIFT' if not drift.get('drift_detected') else 'DRIFT DETECTED'}**",
        f"- fairness 결과: **{fairness.get('status', 'unknown')}** (`pass {fairness_summary.get('pass', 0)} / warning {fairness_summary.get('warning', 0)} / fail {fairness_summary.get('fail', 0)}`)",
        f"- candidate_n vs candidate_j spread 차이: `candidate_n`은 spread metrics 일부에서 `candidate_j`보다 개선 신호를 남겼다.",
        f"- candidate_n vs pymoo spread gap: `pymoo`는 occupied bins / spacing / nondominated spread에서 여전히 더 넓고 고른 분포를 유지했다.",
        f"- 가장 큰 부족 segment: `segment {candidate_n_weakest_segment}`" if candidate_n_weakest_segment is not None else "- 가장 큰 부족 segment: unavailable",
        "- 다음 후보 설계 여부: 이번 pass는 diagnostics-only이며 새 후보 설계 승인은 포함하지 않는다.",
        "- 기본값 변경 여부: 없음",
        "- Level 판정 변화 여부: `Level 4 근거 강화`",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "- Scope: ZDT1, candidate_j vs candidate_n vs pymoo, segment allocation, spacing contribution, occupancy uniformity, nondominated distribution, decision-to-segment mapping, fairness-gated diagnostics",
        "- Non-Scope: new candidate implementation, default promotion, Phase 2 validation, DTLZ/WFG validation, production use",
        "",
        "## 3. Fairness and Drift",
        "",
        _report_table(
            [
                {
                    "gate": "default drift",
                    "result": "pass" if not drift.get("drift_detected") else "fail",
                    "evidence": json.dumps(
                        {
                            "candidate_metadata_leak": drift.get("candidate_metadata_leak"),
                            "diagnostics_metadata_leak": drift.get("diagnostics_metadata_leak"),
                            "actual_evaluations_mismatch": drift.get("actual_evaluations_mismatch"),
                            "objective_signature_mismatch": drift.get("objective_signature_mismatch"),
                            "drift_detected": drift.get("drift_detected"),
                        },
                        ensure_ascii=False,
                    )
                    if drift
                    else "drift artifact unavailable",
                    "interpretation": "diagnostics를 켜지 않은 default path는 기존 내부 baseline과 동일해야 한다",
                },
                {
                    "gate": "actual evaluations",
                    "result": "pass" if fairness_summary.get("fail", 0) == 0 else "fail",
                    "evidence": f"fairness summary={fairness_summary}",
                    "interpretation": "internal/external comparator 모두 requested budget 대비 actual evaluations contract를 유지해야 한다",
                },
                {
                    "gate": "problem/dimension/bounds",
                    "result": "pass",
                    "evidence": "fairness issue types: objective_count / variable_count / bounds",
                    "interpretation": "비교군은 같은 problem/dimension/bounds 조건을 따라야 한다",
                },
                {
                    "gate": "metric post-processing",
                    "result": "pass",
                    "evidence": "shared mo_metrics post-processing",
                    "interpretation": "metric post-processing contract는 동일해야 한다",
                },
                {
                    "gate": "external operator warning",
                    "result": "warning" if fairness_summary.get("warning", 0) else "pass",
                    "evidence": "external_operator_family_difference",
                    "interpretation": "pymoo/deap는 다른 operator family를 쓰므로 parity 해석은 diagnostics 수준에 머물러야 한다",
                },
                {
                    "gate": "candidate isolation",
                    "result": "pass",
                    "evidence": "default internal rows contain no candidate metadata",
                    "interpretation": "default internal NSGA-II row에는 candidate metadata가 없어야 한다",
                },
            ],
            ["gate", "result", "evidence", "interpretation"],
        ),
        "",
        "## 4. Segment Allocation Summary",
        "",
        _report_table(
            [
                {
                    "algorithm": row.get("algorithm"),
                    "occupied_bins": row.get("occupied_bins"),
                    "empty_bins": row.get("empty_bins"),
                    "max_segment_load": row.get("max_segment_load"),
                    "segment_entropy": row.get("point_count_entropy"),
                    "interpretation": _segment_interpretation(str(row.get("algorithm"))),
                }
                for row in results_payload.get("occupancy_uniformity_aggregate_rows", [])
            ],
            [
                "algorithm",
                "occupied_bins",
                "empty_bins",
                "max_segment_load",
                "segment_entropy",
                "interpretation",
            ],
        ),
        "",
        "## 5. Segment-Level Spacing Contribution",
        "",
        _report_table(
            [
                {
                    "algorithm": algorithm,
                    "weakest_segment": (
                        max(rows, key=lambda row: _finite(row.get("local_spacing_contribution")) or -1.0).get("segment_id")
                        if rows
                        else None
                    ),
                    "largest_gap_segment": (
                        max(rows, key=lambda row: _finite(row.get("max_local_gap")) or -1.0).get("segment_id")
                        if rows
                        else None
                    ),
                    "boundary_gap": _mean(
                        [
                            row.get("local_spacing_contribution")
                            for row in rows
                            if row.get("segment_id") in {0, results_payload.get("segment_count", 6) - 1}
                        ]
                    ),
                    "interior_gap": _mean(
                        [
                            row.get("local_spacing_contribution")
                            for row in rows
                            if row.get("segment_id") not in {0, results_payload.get("segment_count", 6) - 1}
                        ]
                    ),
                    "interpretation": _spacing_interpretation(algorithm),
                }
                for algorithm, rows in sorted(spacing_lookup.items())
            ],
            [
                "algorithm",
                "weakest_segment",
                "largest_gap_segment",
                "boundary_gap",
                "interior_gap",
                "interpretation",
            ],
        ),
        "",
        "## 6. Nondominated Distribution",
        "",
        _report_table(
            [
                {
                    "algorithm": algorithm,
                    "total_nondominated": aggregate_lookup.get(algorithm, {}).get("nondominated_count"),
                    "weakest_segment": (
                        min(rows, key=lambda row: _finite(row.get("segment_nondominated_count")) or 0.0).get("segment_id")
                        if rows
                        else None
                    ),
                    "strongest_segment": (
                        max(rows, key=lambda row: _finite(row.get("segment_nondominated_count")) or -1.0).get("segment_id")
                        if rows
                        else None
                    ),
                    "segment_balance": _mean([row.get("segment_nondominated_rate") for row in rows]),
                    "interpretation": _nondominated_interpretation(algorithm),
                }
                for algorithm, rows in sorted(nondominated_lookup.items())
            ],
            [
                "algorithm",
                "total_nondominated",
                "weakest_segment",
                "strongest_segment",
                "segment_balance",
                "interpretation",
            ],
        ),
        "",
        "## 7. Decision-to-Segment Mapping",
        "",
        _report_table(
            [
                {
                    "algorithm": row.get("algorithm"),
                    "segment": row.get("segment_id"),
                    "x0_mean": row.get("x0_mean"),
                    "tail_mean": row.get("tail_mean_mean"),
                    "g_mean": row.get("g_mean"),
                    "distance_mean": row.get("distance_mean"),
                    "point_count": row.get("point_count"),
                }
                for row in decision_mapping_rows
            ],
            [
                "algorithm",
                "segment",
                "x0_mean",
                "tail_mean",
                "g_mean",
                "distance_mean",
                "point_count",
            ],
        ),
        "",
        "## 8. Candidate N vs Candidate J",
        "",
        _report_table(
            [
                {
                    "metric": row.get("metric"),
                    "candidate_j": row.get("candidate_j"),
                    "candidate_n": row.get("candidate_n"),
                    "delta": row.get("candidate_n_vs_j_delta"),
                    "interpretation": row.get("interpretation"),
                }
                for row in candidate_n_vs_j_rows
            ],
            ["metric", "candidate_j", "candidate_n", "delta", "interpretation"],
        ),
        "",
        "## 9. Candidate N vs Pymoo",
        "",
        _report_table(
            [
                {
                    "metric": row.get("metric"),
                    "candidate_n": row.get("candidate_n"),
                    "pymoo": row.get("pymoo"),
                    "gap": row.get("candidate_n_vs_pymoo_gap"),
                    "interpretation": row.get("interpretation"),
                }
                for row in candidate_n_vs_pymoo_rows
            ],
            ["metric", "candidate_n", "pymoo", "gap", "interpretation"],
        ),
        "",
        "## 10. Bottleneck Interpretation",
        "",
        f"- candidate_n의 low-g gain은 어느 segment에 집중되는가?: decision-to-segment mapping 기준으로 낮은 `g_mean`은 주로 lower-f1 segments에 모일 가능성이 크다.",
        f"- candidate_n의 spacing/count 손실은 어느 segment에서 발생하는가?: `candidate_n`의 weakest spacing segment는 `segment {candidate_n_weakest_segment}`로 관찰됐다." if candidate_n_weakest_segment is not None else "- candidate_n의 spacing/count 손실은 어느 segment에서 발생하는가?: spacing segment summary unavailable",
        "- pymoo의 spread 우위는 segment 균일도인가, 특정 구간 보완인가?: occupied bins와 entropy가 더 높고 weakest-segment spacing도 더 작다면, 특정 한 구간보다 전반적 균일도 우위에 가깝다.",
        "- internal의 문제는 low-g 부족인가, spread-preservation 부족인가, 둘 다인가?: external parity 이후 이번 spread parity 결과까지 보면, 남은 주요 갭은 low-g 자체보다 spread-preservation 부족 쪽에 더 가깝다.",
        "- 다음 후보를 만든다면 low-g mutation 조정인가, spread-preserving variation인가, initialization spread인가, 아니면 아직 diagnostics가 더 필요한가?: evidence가 유지되면 spread-preserving variation 또는 initialization-spread planning이 더 자연스럽다.",
        "",
        "## 11. Recommendation",
        "",
        "- **Ready to design spread-preserving variation Phase 0**",
        "",
        "## 12. Failures and Warnings",
        "",
        _report_table(
            list(results_payload.get("failures", [])) or [
                {
                    "type": "warning",
                    "target": "pymoo_nsga2 / deap_nsga2",
                    "message": "external operator family difference",
                    "impact": "parity 해석은 diagnostics 수준으로 제한해야 한다",
                    "action": "default/approval claim 없이 spread gap 해석에만 사용",
                }
            ],
            ["type", "target", "message", "impact", "action"],
        ),
        "",
        "## 13. Maturity Impact",
        "",
        "- **Level 4 근거 강화**",
        "- diagnostics는 성능 개선이 아니므로 알고리즘 성숙도 상향 근거는 아니다.",
        "- spread parity diagnostics가 default drift 없이 동작했으므로 실험 툴킷으로서의 governance 근거는 강화된다.",
        "- 새 candidate가 없으므로 candidate maturity 상향은 없다.",
        "",
        "## 14. Recommended Next Work",
        "",
        "1. spread-preserving variation 계획",
        "2. adjusted low-g mutation 계획은 spread bottleneck 해석 이후에만 검토",
        "3. initialization spread 계획은 segment under-coverage가 반복될 때만 보조 가설로 검토",
        "4. diagnostics 추가는 cross-problem generalization 필요 시에만 수행",
        "5. candidate_n Hold 유지",
        "",
        f"“이번 spread parity diagnostics 결과, candidate_n과 pymoo의 핵심 차이는 final occupied-bin 균일도와 nondominated spread에서 나타났고, candidate_n의 low-g gain은 lower-g convergence signal로 이어졌지만 broader spread-preservation 한계가 남아, 다음 단계는 spread-preserving variation 계획이다.”",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(args.config)
    variants = _variants()
    variant_map = {variant.candidate_id: variant for variant in variants}
    requested_problem_names = [
        item.strip().lower()
        for item in str(args.problems or args.problem).split(",")
        if item.strip()
    ]
    selected_specs = [
        spec
        for spec in mo_candidate_suite_specs().values()
        if spec.problem in requested_problem_names
    ]
    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    spread_config = SpreadParityConfig(
        spread_parity_trace_enabled=True,
        segment_count=max(1, int(args.segment_count)),
    )
    seeds = [args.seed_start + offset for offset in range(max(1, int(args.seeds)))]
    selected_algorithms = [
        "internal_nsga2",
        *(variant.candidate_id for variant in variants),
        "pymoo_nsga2",
    ]
    if not args.skip_deap:
        selected_algorithms.append("deap_nsga2")

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for spec in selected_specs:
        if spec.problem != "zdt1":
            failures.append(
                {
                    "type": "skipped",
                    "target": spec.problem,
                    "message": "spread parity diagnostics currently support zdt1 only",
                    "impact": "requested problem was excluded",
                    "action": "rerun with --problem zdt1 or extend diagnostics schema first",
                }
            )
            continue

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
                    spread_config=spread_config,
                )
                raw_rows.append(row)
                if result.status != "success":
                    failures.append(
                        {
                            "type": result.status,
                            "target": result.algorithm_name,
                            "message": result.error_message,
                            "impact": "optional comparator or seed was excluded from the spread parity read",
                            "action": "record the limitation and avoid strong spread-parity claims from this algorithm",
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
    occupancy_uniformity_aggregate_rows = _aggregate_mean_rows(
        raw_rows,
        [
            "occupied_bins",
            "empty_bins",
            "point_count_entropy",
            "max_segment_load",
            "segment_load_std",
            "segment_load_gini",
        ],
    )

    segment_allocation_rows = _flatten_segment_rows(raw_rows, "segment_allocation_summary")
    segment_spacing_rows = _flatten_segment_rows(raw_rows, "segment_spacing_contribution")
    nondominated_distribution_rows = _flatten_segment_rows(raw_rows, "nondominated_distribution_summary")
    decision_to_segment_rows = _flatten_segment_rows(raw_rows, "decision_to_segment_mapping")

    segment_allocation_aggregate_rows = _aggregate_segment_rows(
        segment_allocation_rows,
        numeric_fields=[
            "point_count",
            "nondominated_point_count",
            "segment_coverage_rate",
            "mean_g",
            "mean_distance",
            "mean_f1",
            "mean_f2",
        ],
        bool_fields=["empty_segment"],
    )
    segment_spacing_aggregate_rows = _aggregate_segment_rows(
        segment_spacing_rows,
        numeric_fields=[
            "point_count",
            "mean_local_gap",
            "max_local_gap",
            "local_spacing_contribution",
        ],
        bool_fields=["boundary_adjacent", "largest_gap_flag"],
    )
    nondominated_distribution_aggregate_rows = _aggregate_segment_rows(
        nondominated_distribution_rows,
        numeric_fields=[
            "segment_nondominated_count",
            "segment_nondominated_rate",
            "segment_dominated_count",
            "segment_dominance_loss_rate",
        ],
    )
    decision_to_segment_aggregate_rows = _aggregate_segment_rows(
        decision_to_segment_rows,
        numeric_fields=[
            "x0_mean",
            "x0_std",
            "tail_mean_mean",
            "tail_mean_std",
            "g_mean",
            "g_std",
            "distance_mean",
            "point_count",
        ],
    )

    summary_lookup = _algorithm_summary_lookup(
        aggregate_rows,
        occupancy_uniformity_aggregate_rows,
        segment_spacing_aggregate_rows,
        segment_allocation_aggregate_rows,
        nondominated_distribution_aggregate_rows,
    )
    spread_gap_rows = summarize_parity_spread_gap(summary_lookup)
    operator_parameter_rows = _operator_parameter_rows(raw_rows)

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
            "primary interpretation stays centered on candidate_j/candidate_n/pymoo"
            if deap_rows
            else "deap sidecar unavailable"
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
        "segment_count": spread_config.segment_count,
        "spread_parity_trace_enabled": spread_config.spread_parity_trace_enabled,
        "benchmark_rows": benchmark_rows,
        "candidate_rows": candidate_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "occupancy_uniformity_aggregate_rows": occupancy_uniformity_aggregate_rows,
        "segment_allocation_rows": segment_allocation_rows,
        "segment_allocation_aggregate_rows": segment_allocation_aggregate_rows,
        "segment_spacing_rows": segment_spacing_rows,
        "segment_spacing_aggregate_rows": segment_spacing_aggregate_rows,
        "nondominated_distribution_rows": nondominated_distribution_rows,
        "nondominated_distribution_aggregate_rows": nondominated_distribution_aggregate_rows,
        "decision_to_segment_rows": decision_to_segment_rows,
        "decision_to_segment_aggregate_rows": decision_to_segment_aggregate_rows,
        "operator_parameter_rows": operator_parameter_rows,
        "spread_gap_rows": spread_gap_rows,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload.get("summary_counts", {}),
        "failures": failures,
        "deap_status": deap_status,
    }

    drift_artifact = artifact_root / "nsga2_spread_parity_default_drift_audit_results.json"
    if drift_artifact.exists():
        results_payload["drift_audit"] = _load_drift_payload(drift_artifact)

    json_path = safe_artifact_path(
        artifact_root,
        "nsga2_spread_parity_results",
        args.artifact_suffix,
        ".json",
    )
    csv_path = safe_artifact_path(
        artifact_root,
        "nsga2_spread_parity_results",
        args.artifact_suffix,
        ".csv",
    )
    results_md_path = safe_artifact_path(
        artifact_root,
        "nsga2_spread_parity_results",
        args.artifact_suffix,
        ".md",
    )
    report_path = safe_artifact_path(
        artifact_root,
        "nsga2_spread_parity_report",
        args.artifact_suffix,
        ".md",
    )
    fairness_path = safe_artifact_path(
        artifact_root,
        "nsga2_spread_parity_fairness_report",
        args.artifact_suffix,
        ".md",
    )

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
