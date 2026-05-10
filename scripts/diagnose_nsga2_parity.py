from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
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
from ga_lab.experiment.budget_baseline_comparison import configured_evaluation_budget
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
from ga_lab.experiment.mo_metrics import coverage_indicator, zdt1_reference_front


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    label: str
    changed: str
    hypothesis: str
    expected_metric: str
    risk: str
    overrides: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose internal NSGA-II parity gaps against external comparators."
    )
    parser.add_argument(
        "--config",
        default="configs/smoke/zdt1_nsga2_smoke.json",
        help="Base NSGA-II config used for ZDT1 parity diagnosis.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/nsga2_parity_diagnosis",
        help="Directory for timestamped raw parity-diagnosis outputs.",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Directory for stable parity-diagnosis artifacts.",
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Optional suffix for artifact names.",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of repeated seeds.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=7101,
        help="First seed for parity diagnosis.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Optional explicit evaluation budget override for external comparators and random anchor.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column) for column in columns} for row in rows])


def _safe_artifact_path(root: Path, base_name: str, suffix: str | None, extension: str) -> Path:
    if suffix:
        safe_suffix = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in suffix.strip()
        ).strip("_")
        if safe_suffix:
            return root / f"{base_name}_{safe_suffix}{extension}"
    candidate = root / f"{base_name}{extension}"
    if candidate.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return root / f"{base_name}_{timestamp}{extension}"
    return candidate


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
    successes = [row for row in rows if row.get("success")]
    return len(successes) / len(rows)


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


def _candidate_specs(base_config: GAConfig) -> list[CandidateSpec]:
    sigma = float(base_config.mutation_options.get("sigma", 0.2))
    return [
        CandidateSpec(
            candidate_id="candidate_a_gen31_pop8",
            label="Candidate A: generation-parity contour",
            changed="population 20 -> 8, generations 12 -> 31",
            hypothesis="The internal loop may be under-evolving because the same budget only yields 12 survivor generations.",
            expected_metric="reference_front_distance / GD / IGD",
            risk="smaller population may reduce diversity and boundary coverage.",
            overrides={"population_size": 8, "generations": 31},
        ),
        CandidateSpec(
            candidate_id="candidate_b_sigma005",
            label="Candidate B: milder gaussian mutation",
            changed=f"gaussian sigma {sigma} -> 0.05",
            hypothesis="The current gaussian mutation may be too disruptive for smooth ZDT1 convergence.",
            expected_metric="reference_front_distance / HV",
            risk="too little mutation can collapse nondominated count and spacing.",
            overrides={"mutation_options": {"sigma": 0.05}},
        ),
        CandidateSpec(
            candidate_id="candidate_c_pop38_gen6",
            label="Candidate C: larger-pop diversity contour",
            changed="population 20 -> 38, generations 12 -> 6",
            hypothesis="The internal path may need wider simultaneous front coverage rather than more generations.",
            expected_metric="HV / spacing / nondominated_count",
            risk="fewer generations can hurt convergence distance badly.",
            overrides={"population_size": 38, "generations": 6},
        ),
        CandidateSpec(
            candidate_id="candidate_d_uniform_crossover",
            label="Candidate D: crossover diversity ablation",
            changed="crossover arithmetic -> uniform",
            hypothesis="Arithmetic crossover may contract the front too aggressively on ZDT1.",
            expected_metric="spacing / nondominated_count / HV",
            risk="uniform crossover can preserve diversity but weaken smooth convergence.",
            overrides={"crossover": "uniform"},
        ),
    ]


def _clone_candidate_config(base_config: GAConfig, spec: CandidateSpec) -> GAConfig:
    clone = GAConfig.from_dict(base_config.to_dict())
    clone.run_name = f"{base_config.run_name}_{spec.candidate_id}"
    for key, value in spec.overrides.items():
        if key in {"mutation_options", "representation_options", "algorithm_options", "selection_options", "crossover_options"}:
            current = deepcopy(getattr(clone, key))
            current.update(deepcopy(value))
            setattr(clone, key, current)
        else:
            setattr(clone, key, deepcopy(value))
    return clone


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
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["algorithm"])].append(row)

    aggregates: list[dict[str, Any]] = []
    for algorithm, bucket in sorted(grouped.items()):
        successful = [row for row in bucket if row.get("success")]
        hv = _summary_stat(successful, "hypervolume_2d")
        distance = _summary_stat(successful, "reference_front_distance")
        gd = _summary_stat(successful, "generational_distance")
        igd = _summary_stat(successful, "inverted_generational_distance")
        spacing = _summary_stat(successful, "spacing")
        coverage = _summary_stat(successful, "reference_front_coverage")
        nondominated = _summary_stat(successful, "nondominated_count")
        runtime = _summary_stat(successful, "runtime_seconds")
        evaluations = _summary_stat(successful, "actual_evaluations")
        duplicate_count = _summary_stat(successful, "front_duplicate_count")
        unique_count = _summary_stat(successful, "front_unique_count")
        status_set = {str(row.get("status", "unknown")) for row in bucket}
        if status_set == {"skipped"}:
            status = "skipped"
        elif "failed" in status_set and not successful:
            status = "failed"
        elif "failed" in status_set:
            status = "partial_failure"
        else:
            status = "success"
        aggregates.append(
            {
                "problem": bucket[0]["problem"],
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
                "mean_nondominated_count": nondominated["mean"],
                "mean_runtime_seconds": runtime["mean"],
                "mean_actual_evaluations": evaluations["mean"],
                "mean_front_unique_count": unique_count["mean"],
                "mean_front_duplicate_count": duplicate_count["mean"],
                "success_rate": _success_rate(bucket),
            }
        )
    return aggregates


def _pairwise_rows(
    rows: list[dict[str, Any]],
    comparison_specs: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["algorithm"])].append(row)

    paired_rows: list[dict[str, Any]] = []
    for left, right, label in comparison_specs:
        for metric_name in METRIC_SPECS:
            summary = paired_metric_summary(
                internal_rows=grouped.get(left, []),
                comparator_rows=grouped.get(right, []),
                metric_name=metric_name,
            )
            paired_rows.append(
                {
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
    return paired_rows


def _run_internal_candidate(
    base_config: GAConfig,
    spec: CandidateSpec,
    *,
    seed: int,
    output_root: Path,
) -> ExternalMOComparatorResult:
    candidate_config = _clone_candidate_config(base_config, spec)
    result = run_internal_nsga2(candidate_config, seed=seed, output_root=str(output_root))
    metadata = dict(result.metadata)
    metadata.update(
        {
            "candidate_id": spec.candidate_id,
            "candidate_label": spec.label,
            "candidate_changed": spec.changed,
            "candidate_hypothesis": spec.hypothesis,
            "candidate_expected_metric": spec.expected_metric,
        }
    )
    return ExternalMOComparatorResult(
        problem_name=result.problem_name,
        algorithm_name=spec.candidate_id,
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


def _results_markdown(
    payload: dict[str, Any],
    *,
    candidate_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# NSGA-II Parity Diagnosis Results",
        "",
        "## Candidate Variants",
        "",
        *_markdown_table(
            candidate_rows,
            ["candidate", "changed", "hypothesis", "expected_metric", "risk"],
        ),
        "",
        "## Aggregate Results",
        "",
        *_markdown_table(
            payload["aggregate_rows"],
            [
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
                "mean_actual_evaluations",
                "mean_front_duplicate_count",
            ],
        ),
        "",
        "## Pairwise Results",
        "",
        *_markdown_table(
            payload["paired_rows"],
            [
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
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    budget = args.budget or configured_evaluation_budget(config)
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    reference_point = [
        float(value)
        for value in config.algorithm_options.get("hypervolume_reference_point", [1.05, 10.5])
    ]
    reference_front = zdt1_reference_front(201)
    candidates = _candidate_specs(config)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = PROJECT_ROOT / args.output_root / f"{timestamp}_{config.problem}_nsga2_parity"
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        raw_rows.append(
            _decorate_front_row(
                result_to_front_row(
                    run_internal_nsga2(config, seed=seed, output_root=str(output_root)),
                    reference_front=reference_front,
                    reference_point=reference_point,
                ),
                reference_front=reference_front,
            )
        )
        raw_rows.append(
            _decorate_front_row(
                result_to_front_row(
                    run_random_archive_anchor(
                        run_random_pareto_archive(config, seed=seed, budget=budget)
                    ),
                    reference_front=reference_front,
                    reference_point=reference_point,
                ),
                reference_front=reference_front,
            )
        )
        raw_rows.append(
            _decorate_front_row(
                result_to_front_row(
                    run_pymoo_nsga2(config, seed=seed, budget=budget),
                    reference_front=reference_front,
                    reference_point=reference_point,
                ),
                reference_front=reference_front,
            )
        )
        raw_rows.append(
            _decorate_front_row(
                result_to_front_row(
                    run_deap_nsga2(config, seed=seed, budget=budget),
                    reference_front=reference_front,
                    reference_point=reference_point,
                ),
                reference_front=reference_front,
            )
        )
        for spec in candidates:
            raw_rows.append(
                _decorate_front_row(
                    result_to_front_row(
                        _run_internal_candidate(config, spec, seed=seed, output_root=output_root),
                        reference_front=reference_front,
                        reference_point=reference_point,
                    ),
                    reference_front=reference_front,
                )
            )

    candidate_rows = [
        {
            "candidate": spec.candidate_id,
            "changed": spec.changed,
            "hypothesis": spec.hypothesis,
            "expected_metric": spec.expected_metric,
            "risk": spec.risk,
        }
        for spec in candidates
    ]

    comparison_specs = [
        ("internal_nsga2", "pymoo_nsga2", "internal baseline vs pymoo"),
        ("internal_nsga2", "deap_nsga2", "internal baseline vs DEAP"),
        ("internal_nsga2", "random_pareto_archive", "internal baseline vs random archive"),
    ]
    for spec in candidates:
        comparison_specs.extend(
            [
                (spec.candidate_id, "internal_nsga2", f"{spec.candidate_id} vs internal baseline"),
                (spec.candidate_id, "pymoo_nsga2", f"{spec.candidate_id} vs pymoo"),
                (spec.candidate_id, "deap_nsga2", f"{spec.candidate_id} vs DEAP"),
                (spec.candidate_id, "random_pareto_archive", f"{spec.candidate_id} vs random archive"),
            ]
        )

    aggregate_rows = _aggregate_rows(raw_rows)
    paired_rows = _pairwise_rows(raw_rows, comparison_specs)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str((PROJECT_ROOT / args.config).resolve()),
        "problem": config.problem,
        "seeds": seeds,
        "requested_budget": budget,
        "reference_point": reference_point,
        "reference_front_point_count": len(reference_front),
        "config_summary": {
            "population_size": config.population_size,
            "generations": config.generations,
            "genome_length": config.genome_length,
            "representation": config.representation,
            "selection": config.selection,
            "crossover": config.crossover,
            "crossover_rate": config.crossover_rate,
            "mutation": config.mutation,
            "mutation_rate": config.mutation_rate,
            "mutation_options": dict(config.mutation_options),
            "objective_directions": list(config.objective_directions),
        },
        "candidate_specs": [asdict(spec) for spec in candidates],
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "runtime_seconds": time.perf_counter() - started,
    }

    results_json_path = _safe_artifact_path(
        artifact_root,
        "nsga2_parity_diagnosis_results",
        args.artifact_suffix,
        ".json",
    )
    results_csv_path = _safe_artifact_path(
        artifact_root,
        "nsga2_parity_diagnosis_results",
        args.artifact_suffix,
        ".csv",
    )
    results_md_path = _safe_artifact_path(
        artifact_root,
        "nsga2_parity_diagnosis_results",
        args.artifact_suffix,
        ".md",
    )
    report_md_path = _safe_artifact_path(
        artifact_root,
        "nsga2_parity_diagnosis_report",
        args.artifact_suffix,
        ".md",
    )

    _write_json(results_json_path, payload)
    _write_csv(
        results_csv_path,
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
            "mean_front_unique_count",
            "mean_front_duplicate_count",
            "success_rate",
        ],
    )
    results_md_path.write_text(
        _results_markdown(payload, candidate_rows=candidate_rows),
        encoding="utf-8",
    )
    report_md_path.write_text(
        "# NSGA-II Parity Diagnosis Report\n\n"
        "See `nsga2_parity_diagnosis_results.json` for machine-readable diagnostics.\n",
        encoding="utf-8",
    )

    snapshot_root = output_root / "artifact_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    _write_json(snapshot_root / "nsga2_parity_diagnosis_results.json", payload)

    print(results_json_path)
    print(report_md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
