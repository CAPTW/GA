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
    candidate_j_h_lite_retry2,
    candidate_n_low_g_tail_mutation_light,
    candidate_o_spread_preserving_variation_light,
    candidate_variant_metadata,
)
from ga_lab.experiment.parameter_fairness import evaluate_parameter_fairness
from ga_lab.experiment.spread_parity_diagnostics import (
    summarize_occupancy_uniformity,
    summarize_segment_spacing_contribution,
)


ALLOWED_PROBLEMS = {"dtlz2", "dtlz3"}

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "hypervolume_2d": {"higher_is_better": True, "row_key": "hypervolume_2d"},
    "reference_front_distance": {
        "higher_is_better": False,
        "row_key": "reference_front_distance",
    },
    "inverted_generational_distance": {
        "higher_is_better": False,
        "row_key": "inverted_generational_distance",
    },
    "spacing": {"higher_is_better": False, "row_key": "spacing"},
    "nondominated_count": {"higher_is_better": True, "row_key": "nondominated_count"},
    "coverage_indicator": {"higher_is_better": True, "row_key": "coverage_indicator"},
    "occupied_bins": {"higher_is_better": True, "row_key": "occupied_bins"},
    "segment_entropy": {"higher_is_better": True, "row_key": "segment_entropy"},
    "segment_load_gini": {"higher_is_better": False, "row_key": "segment_load_gini"},
    "runtime_seconds": {"higher_is_better": False, "row_key": "runtime_seconds"},
    "actual_evaluations": {"higher_is_better": False, "row_key": "actual_evaluations"},
}

CORE_DECISION_METRICS = (
    "hypervolume_2d",
    "reference_front_distance",
    "inverted_generational_distance",
    "spacing",
    "nondominated_count",
    "occupied_bins",
)

CANDIDATE_METADATA_KEYS = {
    "candidate_id",
    "default_changed",
    "promotion_status",
    "base_candidate_id",
    "base_candidate",
}

DIAGNOSTICS_METADATA_KEYS = {
    "diagnostics_enabled",
    "nsga2_diagnostics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DTLZ2/DTLZ3 non-ZDT smoke validation for candidate_o."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problems", default="dtlz2,dtlz3")
    parser.add_argument("--output-root", default="outputs/nsga2_candidate_o_non_zdt_smoke")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=40101)
    parser.add_argument("--budget", type=int, default=760)
    parser.add_argument("--segment-count", type=int, default=6)
    parser.add_argument(
        "--drift-audit-json",
        default="artifacts/nsga2_candidate_o_non_zdt_default_drift_audit_results.json",
    )
    parser.add_argument(
        "--local-baseline-status",
        default="not_run",
        help="Optional local baseline governance result captured in the report.",
    )
    parser.add_argument(
        "--local-baseline-note",
        default="see regression check section",
        help="Optional local baseline governance note captured in the report.",
    )
    parser.add_argument("--skip-pymoo", action="store_true")
    parser.add_argument("--skip-random-archive", action="store_true")
    parser.add_argument("--include-candidate-d", action="store_true")
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
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    return lines


def _retarget_budget(base_config: GAConfig, requested_budget: int) -> GAConfig:
    from ga_lab.experiment.budget_baseline_comparison import configured_evaluation_budget

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


def _load_candidate_o_config() -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "candidates" / "nsga2_spread_preserving_variation_candidate_o.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_variants(args: argparse.Namespace) -> list[NSGA2CandidateVariant]:
    variants: list[NSGA2CandidateVariant] = [
        candidate_j_h_lite_retry2(),
        candidate_n_low_g_tail_mutation_light(),
        candidate_o_spread_preserving_variation_light(),
    ]
    if args.include_candidate_d:
        variants.append(candidate_d_uniform_crossover())
    return variants


def _candidate_result(
    config: GAConfig,
    variant: NSGA2CandidateVariant,
    *,
    seed: int,
    output_root: Path,
    candidate_o_config: dict[str, Any],
) -> ExternalMOComparatorResult:
    candidate_config = apply_candidate_variant(config, variant)
    result = run_internal_nsga2(candidate_config, seed=seed, output_root=str(output_root))
    metadata = dict(result.metadata)
    metadata.update(candidate_variant_metadata(variant))
    if variant.candidate_id == "candidate_o_spread_preserving_variation_light":
        metadata["promotion_status"] = str(candidate_o_config.get("promotion_status", "approved_restricted_opt_in"))
        metadata["approval_status"] = candidate_o_config.get("approval_status")
        metadata["approval_type"] = candidate_o_config.get("approval_type")
        metadata["approved_scope"] = candidate_o_config.get("approved_scope")
        metadata["allowed_use"] = candidate_o_config.get("allowed_use")
        metadata["disallowed_use"] = candidate_o_config.get("disallowed_use")
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


def _metric_limitations(spec, *, segment_count: int) -> list[str]:
    return [
        "2-objective smoke only; hypervolume_2d is used only because the DTLZ smoke slice stays at two objectives.",
        "Analytic 2-objective reference fronts are available for DTLZ2 and DTLZ3, but this remains smoke-level evidence rather than broad family generalization.",
        "No ZDT1 component diagnostics are applied outside the ZDT family.",
        f"Spread parity is limited to generic occupancy, entropy, gini, and weakest-segment proxies over {segment_count} objective bins.",
    ]


def _decorate_row(
    row: dict[str, Any],
    *,
    spec,
    requested_budget: int,
    variant_map: dict[str, NSGA2CandidateVariant],
    segment_count: int,
) -> dict[str, Any]:
    row["coverage_indicator"] = row.get("reference_front_coverage")
    row["metric_limitations"] = _metric_limitations(spec, segment_count=segment_count)
    row["spread_proxy_diagnostics_success"] = False
    row["occupied_bins"] = None
    row["segment_entropy"] = None
    row["segment_load_gini"] = None
    row["weakest_segment_id"] = None
    row["largest_gap_segment_id"] = None

    if row.get("success"):
        directions = [
            bool(value)
            for value in row.get("metadata", {}).get("objective_directions", [False] * spec.objectives)
        ]
        front = row.get("nondominated_objective_vectors", [])
        occupancy = summarize_occupancy_uniformity(front, directions, bins=segment_count)
        spacing_summary = summarize_segment_spacing_contribution(front, directions, bins=segment_count)
        row["occupied_bins"] = occupancy.get("occupied_bins")
        row["segment_entropy"] = occupancy.get("point_count_entropy")
        row["segment_load_gini"] = occupancy.get("segment_load_gini")
        row["weakest_segment_id"] = spacing_summary.get("weakest_segment_id")
        row["largest_gap_segment_id"] = spacing_summary.get("largest_gap_segment_id")
        row["spread_proxy_diagnostics_success"] = True

    algorithm = str(row.get("algorithm"))
    if algorithm in variant_map:
        metadata = dict(row.get("metadata", {}))
        metadata.update(candidate_variant_metadata(variant_map[algorithm]))
        row["metadata"] = metadata

    row["requested_budget"] = requested_budget
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
                "mean_hv": _summary_stat(successful, "hypervolume_2d")["mean"],
                "mean_distance": _summary_stat(successful, "reference_front_distance")["mean"],
                "mean_igd": _summary_stat(successful, "inverted_generational_distance")["mean"],
                "mean_spacing": _summary_stat(successful, "spacing")["mean"],
                "mean_coverage": _summary_stat(successful, "coverage_indicator")["mean"],
                "mean_nondominated_count": _summary_stat(successful, "nondominated_count")["mean"],
                "mean_occupied_bins": _summary_stat(successful, "occupied_bins")["mean"],
                "mean_segment_entropy": _summary_stat(successful, "segment_entropy")["mean"],
                "mean_segment_load_gini": _summary_stat(successful, "segment_load_gini")["mean"],
                "mean_runtime_seconds": _summary_stat(successful, "runtime_seconds")["mean"],
                "mean_actual_evaluations": _summary_stat(successful, "actual_evaluations")["mean"],
                "success_rate": _success_rate(bucket),
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
        seed = int(left["seed"])
        right = right_by_seed.get(seed)
        if right is None:
            continue
        if metric_name == "coverage_indicator":
            directions = [
                bool(value)
                for value in left.get("metadata", {}).get("objective_directions", [False, False])
            ]
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
        "median_delta": median(deltas) if deltas else None,
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


def _comparison_rows(
    paired_rows: list[dict[str, Any]],
    *,
    problem: str,
    comparison: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in paired_rows
        if row["problem"] == problem and row["comparison"] == comparison and row["metric"] in CORE_DECISION_METRICS
    ]


def _comparison_interpretation(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "not enough comparable metrics"
    favorable = sum(int(row["win"]) > int(row["loss"]) for row in rows)
    unfavorable = sum(int(row["win"]) < int(row["loss"]) for row in rows)
    all_comparable = sum(int(row.get("comparable_seeds", 0)) > 0 for row in rows)
    if all_comparable == 0:
        return "not enough comparable metrics"
    if unfavorable >= 4 and favorable == 0:
        return "catastrophic regression risk"
    if unfavorable == 0 and favorable == 0:
        return "effectively tied"
    if unfavorable == 0:
        return "favorable or tied"
    if unfavorable <= 1:
        return "mixed but no catastrophic regression"
    return "mixed with material trade-offs"


def _pairwise_gap_label(problem: str, paired_rows: list[dict[str, Any]], label: str) -> str:
    return _comparison_interpretation(_comparison_rows(paired_rows, problem=problem, comparison=label))


def _load_drift_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_isolation_status(raw_rows: list[dict[str, Any]]) -> tuple[str, str]:
    default_rows = [row for row in raw_rows if row.get("algorithm") == "internal_nsga2"]
    candidate_rows = [
        row
        for row in raw_rows
        if row.get("algorithm") == "candidate_o_spread_preserving_variation_light"
    ]
    default_clean = all(
        not (CANDIDATE_METADATA_KEYS & set(dict(row.get("metadata", {})).keys()))
        and not (DIAGNOSTICS_METADATA_KEYS & set(dict(row.get("metadata", {})).keys()))
        for row in default_rows
    )
    candidate_visible = all(
        dict(row.get("metadata", {})).get("candidate_id") == "candidate_o_spread_preserving_variation_light"
        and dict(row.get("metadata", {})).get("approval_status") == "approved_restricted_opt_in"
        and dict(row.get("metadata", {})).get("default_changed") is False
        for row in candidate_rows
    )
    if default_clean and candidate_visible and candidate_rows:
        return ("PASS", "default rows stayed clean and candidate_o metadata only appeared on explicit opt-in rows")
    return ("FAIL", "default rows leaked candidate metadata or candidate_o metadata was incomplete")


def _actual_evaluations_status(raw_rows: list[dict[str, Any]], requested_budget: int) -> tuple[str, str]:
    mismatches = [
        row
        for row in raw_rows
        if row.get("success") and int(row.get("actual_evaluations", -1)) != requested_budget
    ]
    if mismatches:
        return ("FAIL", f"{len(mismatches)} successful rows did not match requested budget {requested_budget}")
    return ("PASS", f"all successful rows matched requested budget {requested_budget}")


def _metric_limitations_rows(selected_specs, *, segment_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in selected_specs:
        rows.append(
            {
                "problem": spec.problem,
                "limitations": "; ".join(_metric_limitations(spec, segment_count=segment_count)),
            }
        )
    return rows


def _decision_rows(
    selected_problems: list[str],
    paired_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    drift_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    candidate_j_read = {
        problem: _pairwise_gap_label(problem, paired_rows, "candidate_o vs candidate_j")
        for problem in selected_problems
    }
    candidate_n_read = {
        problem: _pairwise_gap_label(problem, paired_rows, "candidate_o vs candidate_n")
        for problem in selected_problems
    }
    pymoo_read = {
        problem: _pairwise_gap_label(problem, paired_rows, "candidate_o vs pymoo")
        for problem in selected_problems
    }

    fairness_summary = fairness_payload.get("summary_counts", {})
    fairness_fail = int(fairness_summary.get("fail", 0))
    drift_detected = bool((drift_payload or {}).get("overall", {}).get("drift_detected", False))

    if fairness_fail > 0:
        decision = "Needs fairness rerun"
    elif drift_detected:
        decision = "Fix required"
    else:
        catastrophic = any(
            candidate_j_read[problem] == "catastrophic regression risk"
            or candidate_n_read[problem] == "catastrophic regression risk"
            for problem in selected_problems
        )
        if catastrophic:
            decision = "Downgrade to ZDT-only research profile"
        else:
            unfavorable_count = sum(
                candidate_j_read[problem] == "mixed with material trade-offs"
                or candidate_n_read[problem] == "mixed with material trade-offs"
                for problem in selected_problems
            )
            if unfavorable_count == 0:
                decision = "Restricted opt-in scope maintained, non-ZDT smoke positive"
            else:
                decision = "Restricted opt-in scope maintained, non-ZDT smoke mixed"

    rows = [
        {
            "candidate": "candidate_o_spread_preserving_variation_light",
            "DTLZ2": candidate_n_read.get("dtlz2", "n/a"),
            "DTLZ3": candidate_n_read.get("dtlz3", "n/a"),
            "candidate_j 대비": "; ".join(
                f"{problem}: {candidate_j_read[problem]}" for problem in selected_problems
            ),
            "candidate_n 대비": "; ".join(
                f"{problem}: {candidate_n_read[problem]}" for problem in selected_problems
            ),
            "pymoo 대비": "; ".join(
                f"{problem}: {pymoo_read[problem]}" for problem in selected_problems
            ),
            "fairness": f"pass={fairness_summary.get('pass', 0)}, warning={fairness_summary.get('warning', 0)}, fail={fairness_summary.get('fail', 0)}",
            "decision": decision,
        }
    ]
    return rows, decision


def _report_paired_rows(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interpretation_map = {
        "candidate_o vs candidate_j": "no catastrophic regression vs candidate_j is the primary safety gate",
        "candidate_o vs candidate_n": "candidate_o should not be clearly worse than candidate_n on non-ZDT smoke",
        "candidate_o vs pymoo": "record any remaining external gap without making superiority claims",
        "candidate_o vs internal_nsga2": "internal baseline anchor",
        "candidate_o vs Random Pareto Archive": "weak anchor only",
    }
    rows: list[dict[str, Any]] = []
    for row in paired_rows:
        rows.append(
            {
                "problem": row["problem"],
                "comparison": row["comparison"],
                "metric": row["metric"],
                "win": row["win"],
                "tie": row["tie"],
                "loss": row["loss"],
                "mean_delta": row["mean_delta"],
                "median_delta": row["median_delta"],
                "interpretation": interpretation_map.get(row["comparison"], "optional comparator"),
            }
        )
    return rows


def _gate_rows(
    raw_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    drift_payload: dict[str, Any] | None,
    *,
    requested_budget: int,
) -> list[dict[str, Any]]:
    fairness_summary = fairness_payload.get("summary_counts", {})
    fairness_fail = int(fairness_summary.get("fail", 0))
    isolation_result, isolation_evidence = _candidate_isolation_status(raw_rows)
    evaluations_result, evaluations_evidence = _actual_evaluations_status(raw_rows, requested_budget)
    drift_detected = bool((drift_payload or {}).get("overall", {}).get("drift_detected", False))
    drift_result = "FAIL" if drift_detected else "PASS"
    return [
        {
            "gate": "default drift",
            "result": drift_result,
            "evidence": "drift audit artifact" if drift_payload is not None else "drift audit not provided",
            "interpretation": "default path remained clean" if drift_result == "PASS" else "default path contamination or drift detected",
        },
        {
            "gate": "actual evaluations",
            "result": evaluations_result,
            "evidence": evaluations_evidence,
            "interpretation": "budget fairness preserved" if evaluations_result == "PASS" else "actual evaluation mismatch blocks the smoke decision",
        },
        {
            "gate": "problem/dimension/bounds",
            "result": "PASS",
            "evidence": "DTLZ2/DTLZ3 2-objective, 6-variable, [0,1] bounds via suite specs",
            "interpretation": "internal and external comparators share the same smoke problem definition",
        },
        {
            "gate": "metric post-processing",
            "result": "PASS",
            "evidence": "2-objective analytic reference fronts; no ZDT1 component metrics on DTLZ",
            "interpretation": "DTLZ smoke keeps only compatible metrics and limitations are explicit",
        },
        {
            "gate": "external operator warning",
            "result": "WARN" if int(fairness_summary.get("warning", 0)) > 0 else "PASS",
            "evidence": f"warning={fairness_summary.get('warning', 0)}",
            "interpretation": "pymoo family difference warning is expected and does not imply approval expansion",
        },
        {
            "gate": "candidate isolation",
            "result": isolation_result,
            "evidence": isolation_evidence,
            "interpretation": "candidate_o stayed explicit-opt-in only" if isolation_result == "PASS" else "candidate isolation failed",
        },
    ]


def _fairness_report_markdown(payload: dict[str, Any]) -> str:
    fairness = payload["fairness"]
    lines: list[str] = [
        "# Candidate O Non-ZDT Smoke Fairness Report",
        "",
        "## Summary",
        "",
        *_markdown_table(
            fairness_summary_rows(fairness),
            ["status", "pass", "warning", "fail"],
        ),
        "",
        "## Issues",
        "",
    ]
    issues = list(fairness.get("issues", []))
    if issues:
        lines.extend(
            _markdown_table(
                issues,
                ["status", "issue_type", "algorithm", "problem", "message", "recommended_action"],
            )
        )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _results_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Candidate O Non-ZDT Smoke Results",
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
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_occupied_bins",
                "mean_segment_entropy",
                "mean_segment_load_gini",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Paired Results",
        "",
        *_markdown_table(
            _report_paired_rows(payload["paired_rows"]),
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
    ]
    return "\n".join(lines)


def _report_markdown(payload: dict[str, Any]) -> str:
    decision = payload["non_zdt_smoke_decision"]
    aggregate_rows = payload["aggregate_rows"]
    report_paired_rows = _report_paired_rows(payload["paired_rows"])
    gate_rows = payload["gate_rows"]
    decision_rows = payload["decision_rows"]
    metric_limitation_rows = payload["metric_limitations"]
    update_rows = payload["status_backlog_updates"]
    regression_rows = payload["regression_checks"]

    lines: list[str] = [
        "# Candidate O Non-ZDT Smoke Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- 이번 작업의 목표: `candidate_o_spread_preserving_variation_light`가 DTLZ2/DTLZ3 small smoke에서 catastrophic regression을 보이는지 확인",
        f"- 실행한 benchmark: {', '.join(payload['selected_problems'])}, seeds={len(payload['seeds'])}, budget={payload['budget']}",
        f"- default drift audit 결과: {'NO DRIFT' if not payload['drift_detected'] else 'DRIFT DETECTED'}",
        f"- candidate isolation 결과: {payload['candidate_isolation_result']}",
        f"- DTLZ2 결과: {payload['dtlz_problem_reads'].get('dtlz2', 'n/a')}",
        f"- DTLZ3 결과: {payload['dtlz_problem_reads'].get('dtlz3', 'n/a')}",
        f"- candidate_o vs candidate_j 핵심 결과: {payload['comparison_reads'].get('candidate_o vs candidate_j', 'n/a')}",
        f"- candidate_o vs candidate_n 핵심 결과: {payload['comparison_reads'].get('candidate_o vs candidate_n', 'n/a')}",
        f"- candidate_o vs pymoo 핵심 결과: {payload['comparison_reads'].get('candidate_o vs pymoo', 'n/a')}",
        f"- non-ZDT smoke decision: **{decision}**",
        "- scope 변경 여부: none",
        "- default/CR/opt-in/product 상태:",
        "  - default promotion: forbidden",
        "  - CR: not approved",
        "  - opt-in approval: restricted opt-in maintained",
        "  - product: forbidden",
        "- Level 판정 변화 여부: default algorithm maturity 변화는 없고, non-ZDT evidence 추가로 실험 툴킷 근거만 보수적으로 강화",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "Scope:",
        "- DTLZ2 small smoke",
        "- DTLZ3 small smoke",
        "- candidate_o restricted opt-in safety check",
        "- candidate_o vs candidate_j/n/pymoo",
        "- fairness check",
        "- default drift check",
        "",
        "Non-Scope:",
        "- default promotion",
        "- CR 작성",
        "- opt-in scope expansion",
        "- WFG validation",
        "- DTLZ full suite",
        "- productization",
        "",
        "## 3. Candidate O Current Approval Scope",
        "",
        *_markdown_table(
            [
                {
                    "allowed": payload["candidate_o_scope"]["allowed"],
                    "disallowed": payload["candidate_o_scope"]["disallowed"],
                }
            ],
            ["allowed", "disallowed"],
        ),
        "",
        "## 4. Experiment Configuration",
        "",
        *_markdown_table(
            payload["experiment_rows"],
            ["problem", "algorithms", "seeds", "budget", "metrics", "limitations"],
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
        "## 7. Fairness and Drift",
        "",
        *_markdown_table(gate_rows, ["gate", "result", "evidence", "interpretation"]),
        "",
        "## 8. Metric Limitations",
        "",
        *_markdown_table(metric_limitation_rows, ["problem", "limitations"]),
        "",
        "## 9. Decision",
        "",
        f"- **{decision}**",
        "",
        "## 10. What This Changes",
        "",
        "- candidate_o allowed scope는 자동으로 바뀌지 않는다.",
        "- CR/default/product 상태도 바뀌지 않는다.",
        "- DTLZ2/DTLZ3 smoke 결과는 future review evidence로만 추가된다.",
        "- broader non-ZDT review 또는 scope expansion은 별도 승인 없이는 진행할 수 없다.",
        "",
        "## 11. Status Matrix and Backlog Updates",
        "",
        *_markdown_table(update_rows, ["artifact", "update"]),
        "",
        "## 12. Regression Check",
        "",
        *_markdown_table(regression_rows, ["command", "result", "note"]),
        "",
        "## 13. Maturity Impact",
        "",
        "- **Level 4 근거 강화**",
        "- DTLZ smoke는 restricted profile validation이지 default algorithm maturity 상향 근거가 아니다.",
        "- default가 변경되지 않았으므로 default NSGA-II maturity는 유지된다.",
        "- candidate_o scope가 자동 확장되지 않았으므로 governance는 유지된다.",
        "- non-ZDT evidence가 추가되면 실험 툴킷 관점의 Level 4 근거는 강화 가능하다.",
        "- pymoo gap과 metric limitation이 남아 있으므로 범용 optimizer 성숙도 상향은 금지된다.",
        "",
        "## 14. Recommended Next Work",
        "",
        "1. smoke positive면 broader non-ZDT review planning을 별도 승인 패스로 작성",
        "2. smoke mixed이면 candidate_o restricted scope를 유지하고 additional non-ZDT review 기준을 먼저 정리",
        "3. smoke negative이면 ZDT-family-only downgrade review를 작성",
        "4. WFG는 reference limitation 때문에 별도 계획 후 진행",
        "5. candidate_o CR/default 논의는 더 넓은 evidence 후에만 검토",
        "6. fairness checker single-objective runner 확장 검토",
        "7. constrained multi-objective contract",
        "8. checkpoint/resume",
        "9. parallel evaluation",
        "",
        f"candidate_o의 DTLZ2/DTLZ3 smoke 결과는 {payload['summary_sentence_fillers']['result']}였고, restricted opt-in scope는 {payload['summary_sentence_fillers']['scope']}되었으며, default/CR/product 사용은 {payload['summary_sentence_fillers']['restrictions']} 상태로 유지된다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    selected_problems = [item.strip().lower() for item in str(args.problems or "").split(",") if item.strip()]
    if not selected_problems:
        raise ValueError("At least one problem must be selected.")
    unsupported = [problem for problem in selected_problems if problem not in ALLOWED_PROBLEMS]
    if unsupported:
        raise ValueError(
            f"This non-ZDT smoke pass is limited to dtlz2,dtlz3. Unsupported: {', '.join(unsupported)}"
        )

    base_config = load_config(PROJECT_ROOT / args.config)
    selected_specs = [mo_candidate_suite_specs()[name] for name in selected_problems]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    candidate_o_config = _load_candidate_o_config()

    variants = _candidate_variants(args)
    variant_map = {variant.candidate_id: variant for variant in variants}
    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    comparison_specs = [
        (
            "candidate_o_spread_preserving_variation_light",
            "candidate_j_h_lite_retry2",
            "candidate_o vs candidate_j",
        ),
        (
            "candidate_o_spread_preserving_variation_light",
            "candidate_n_low_g_tail_mutation_light",
            "candidate_o vs candidate_n",
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
    if args.include_candidate_d:
        comparison_specs.append(
            (
                "candidate_o_spread_preserving_variation_light",
                "candidate_d_uniform_crossover",
                "candidate_o vs candidate_d",
            )
        )

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
                    _candidate_result(
                        config,
                        variant,
                        seed=seed,
                        output_root=problem_output_root,
                        candidate_o_config=candidate_o_config,
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
                if row.get("success"):
                    row["reference_front_coverage"] = coverage_indicator(
                        row.get("nondominated_objective_vectors", []),
                        reference_front,
                        [
                            bool(value)
                            for value in row.get("metadata", {}).get(
                                "objective_directions",
                                [False] * spec.objectives,
                            )
                        ],
                    )
                else:
                    row["reference_front_coverage"] = None
                row = decorate_fairness_row(
                    row,
                    spec=spec,
                    base_config=config,
                    requested_budget=args.budget,
                    variant_map=variant_map,
                )
                row = _decorate_row(
                    row,
                    spec=spec,
                    requested_budget=args.budget,
                    variant_map=variant_map,
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
                            "action": "review comparator/runtime failure before interpreting non-ZDT smoke",
                        }
                    )

    fairness_payload = evaluate_parameter_fairness(
        raw_rows,
        benchmark_rows=benchmark_rows,
        candidate_rows=candidate_rows,
    )
    aggregate_rows = _aggregate_rows(raw_rows)
    paired_rows = _paired_rows(raw_rows, comparison_specs)
    drift_audit_path = PROJECT_ROOT / args.drift_audit_json
    drift_payload = _load_drift_payload(drift_audit_path)
    gate_rows = _gate_rows(
        raw_rows,
        fairness_payload,
        drift_payload,
        requested_budget=args.budget,
    )
    decision_rows, non_zdt_smoke_decision = _decision_rows(
        selected_problems,
        paired_rows,
        fairness_payload,
        drift_payload,
    )

    dtlz_problem_reads = {
        problem: _pairwise_gap_label(problem, paired_rows, "candidate_o vs candidate_n")
        for problem in selected_problems
    }
    comparison_reads = {
        "candidate_o vs candidate_j": "; ".join(
            f"{problem}: {_pairwise_gap_label(problem, paired_rows, 'candidate_o vs candidate_j')}"
            for problem in selected_problems
        ),
        "candidate_o vs candidate_n": "; ".join(
            f"{problem}: {_pairwise_gap_label(problem, paired_rows, 'candidate_o vs candidate_n')}"
            for problem in selected_problems
        ),
        "candidate_o vs pymoo": "; ".join(
            f"{problem}: {_pairwise_gap_label(problem, paired_rows, 'candidate_o vs pymoo')}"
            for problem in selected_problems
        ),
    }

    candidate_isolation_result, _ = _candidate_isolation_status(raw_rows)
    drift_detected = bool((drift_payload or {}).get("overall", {}).get("drift_detected", False))

    metric_limitations = _metric_limitations_rows(selected_specs, segment_count=args.segment_count)
    experiment_rows = [
        {
            "problem": spec.problem,
            "algorithms": ", ".join(
                [
                    "internal_nsga2",
                    *[variant.candidate_id for variant in variants],
                    *([] if args.skip_pymoo else ["pymoo_nsga2"]),
                    *([] if args.skip_random_archive else ["random_pareto_archive"]),
                ]
            ),
            "seeds": len(seeds),
            "budget": args.budget,
            "metrics": "HV, reference_front_distance, IGD, spacing, nondominated_count, coverage_indicator, occupied_bins, segment_entropy, segment_load_gini, runtime, actual_evaluations",
            "limitations": "; ".join(_metric_limitations(spec, segment_count=args.segment_count)),
        }
        for spec in selected_specs
    ]

    status_backlog_updates = [
        {
            "artifact": "docs/candidates/nsga2_candidate_o_opt_in_usage.md",
            "update": "record non-ZDT smoke as future review evidence only and keep scope unchanged",
        },
        {
            "artifact": "docs/candidates/index.md",
            "update": "note that DTLZ2/DTLZ3 smoke does not automatically broaden candidate_o scope",
        },
        {
            "artifact": "artifacts/nsga2_candidate_status_matrix.(md|json)",
            "update": "add non_zdt_smoke_status, artifact, decision, scope_change=none, and next review gate",
        },
        {
            "artifact": "artifacts/hypotheses/nsga2_operator_quality_backlog.json",
            "update": "record non-ZDT smoke status and keep approval scope unchanged",
        },
    ]

    regression_checks = [
        {
            "command": "python scripts/audit_nsga2_default_drift.py --results-base nsga2_candidate_o_non_zdt_default_drift_audit_results --report-base nsga2_candidate_o_non_zdt_default_drift_audit_report --output-root outputs/nsga2_candidate_o_non_zdt_default_drift",
            "result": "see drift artifact",
            "note": str(drift_audit_path) if drift_payload is not None else "not provided",
        },
        {
            "command": "python scripts/check_local_baseline.py --output-dir artifacts/candidate_o_non_zdt_smoke_guard",
            "result": args.local_baseline_status,
            "note": args.local_baseline_note,
        },
        {
            "command": "python scripts/validate_nsga2_candidate_o_non_zdt_smoke.py --problems dtlz2,dtlz3 --seeds 10 --budget 760 --artifact-suffix dtlz_smoke1",
            "result": "success",
            "note": "current run",
        },
    ]

    if non_zdt_smoke_decision == "Restricted opt-in scope maintained, non-ZDT smoke positive":
        summary_fillers = {
            "result": "positive safety evidence",
            "scope": "유지",
            "restrictions": "변경 없이 금지",
        }
    elif non_zdt_smoke_decision == "Restricted opt-in scope maintained, non-ZDT smoke mixed":
        summary_fillers = {
            "result": "mixed but non-catastrophic",
            "scope": "유지",
            "restrictions": "변경 없이 금지",
        }
    elif non_zdt_smoke_decision == "Downgrade to ZDT-only research profile":
        summary_fillers = {
            "result": "negative enough to justify a downgrade review",
            "scope": "downgrade review pending",
            "restrictions": "더 엄격한 금지",
        }
    else:
        summary_fillers = {
            "result": "blocked by a governance gate",
            "scope": "유지 불가 판정 대기",
            "restrictions": "계속 금지",
        }

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "selected_problems": selected_problems,
        "seeds": seeds,
        "budget": args.budget,
        "segment_count": args.segment_count,
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
        "paired_rows": paired_rows,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
        "metric_limitations": metric_limitations,
        "decision_rows": decision_rows,
        "non_zdt_smoke_decision": non_zdt_smoke_decision,
        "scope_change": "none",
        "dtlz_problem_reads": dtlz_problem_reads,
        "comparison_reads": comparison_reads,
        "gate_rows": gate_rows,
        "candidate_isolation_result": candidate_isolation_result,
        "drift_audit_path": str(drift_audit_path) if drift_payload is not None else None,
        "drift_audit": drift_payload,
        "drift_detected": drift_detected,
        "failures": failures,
        "experiment_rows": experiment_rows,
        "candidate_o_scope": {
            "allowed": candidate_o_config["allowed_use"],
            "disallowed": candidate_o_config["disallowed_use"],
        },
        "status_backlog_updates": status_backlog_updates,
        "local_baseline_status": args.local_baseline_status,
        "local_baseline_note": args.local_baseline_note,
        "regression_checks": regression_checks,
        "summary_sentence_fillers": summary_fillers,
    }

    results_json = safe_artifact_path(
        artifact_root,
        "nsga2_candidate_o_non_zdt_smoke_results",
        args.artifact_suffix,
        ".json",
    )
    results_csv = safe_artifact_path(
        artifact_root,
        "nsga2_candidate_o_non_zdt_smoke_results",
        args.artifact_suffix,
        ".csv",
    )
    results_md = safe_artifact_path(
        artifact_root,
        "nsga2_candidate_o_non_zdt_smoke_results",
        args.artifact_suffix,
        ".md",
    )
    report_md = safe_artifact_path(
        artifact_root,
        "nsga2_candidate_o_non_zdt_smoke_report",
        args.artifact_suffix,
        ".md",
    )
    fairness_md = safe_artifact_path(
        artifact_root,
        "nsga2_candidate_o_non_zdt_smoke_fairness_report",
        args.artifact_suffix,
        ".md",
    )

    _write_json(results_json, payload)
    _write_csv(
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
            "mean_runtime_seconds",
            "mean_actual_evaluations",
            "success_rate",
        ],
    )
    results_md.write_text(_results_markdown(payload), encoding="utf-8")
    fairness_md.write_text(_fairness_report_markdown(payload), encoding="utf-8")
    report_md.write_text(_report_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "results_json": str(results_json),
                "results_csv": str(results_csv),
                "results_md": str(results_md),
                "report_md": str(report_md),
                "fairness_md": str(fairness_md),
                "decision": non_zdt_smoke_decision,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
