from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
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
    ExternalMOComparatorResult,
    METRIC_SPECS,
    optional_library_status,
    paired_metric_summary,
    result_to_front_row,
    run_deap_nsga2,
    run_internal_nsga2,
    run_pymoo_nsga2,
    run_random_archive_anchor,
)
from ga_lab.experiment.mo_baselines import run_random_pareto_archive
from ga_lab.experiment.mo_metrics import zdt1_reference_front
from ga_lab.experiment.parameter_fairness import (
    METRIC_POSTPROCESSING_ID,
    evaluate_parameter_fairness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare internal NSGA-II against optional external MOEA libraries."
    )
    parser.add_argument(
        "--config",
        default="configs/smoke/zdt1_nsga2_smoke.json",
        help="ZDT1 NSGA-II config used as the external comparison anchor.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/external_mo_comparison",
        help="Directory for timestamped raw comparison outputs.",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Directory for stable external comparison artifacts.",
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Optional suffix for stable artifact names, e.g. 'installed'.",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of repeated seeds.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=7101,
        help="First seed for the repeated external comparison.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Optional explicit evaluation budget override.",
    )
    parser.add_argument(
        "--skip-random-archive",
        action="store_true",
        help="Skip the internal random Pareto archive anchor.",
    )
    parser.add_argument(
        "--skip-pymoo",
        action="store_true",
        help="Skip the optional pymoo NSGA-II comparator.",
    )
    parser.add_argument(
        "--skip-deap",
        action="store_true",
        help="Skip the optional DEAP NSGA-II comparator.",
    )
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
    statuses = {str(row.get("status", "unknown")) for row in rows}
    if statuses == {"skipped"}:
        return None
    successes = [row for row in rows if row.get("success")]
    return len(successes) / len(rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    aggregates: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        successful = [row for row in bucket if row.get("success")]
        hv = _summary_stat(successful, "hypervolume_2d")
        distance = _summary_stat(successful, "reference_front_distance")
        igd = _summary_stat(successful, "inverted_generational_distance")
        spacing = _summary_stat(successful, "spacing")
        nondominated = _summary_stat(successful, "nondominated_count")
        runtime_summary = _summary_stat(successful, "runtime_seconds")
        evaluations_summary = _summary_stat(successful, "actual_evaluations")
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
                "std_hv": hv["std"],
                "median_hv": hv["median"],
                "mean_distance": distance["mean"],
                "mean_igd": igd["mean"],
                "mean_spacing": spacing["mean"],
                "mean_nondominated_count": nondominated["mean"],
                "success_rate": _success_rate(bucket),
                "mean_runtime_seconds": runtime_summary["mean"],
                "mean_actual_evaluations": evaluations_summary["mean"],
            }
        )
    return aggregates


def _paired_rows(
    rows: list[dict[str, Any]],
    *,
    internal_algorithm: str,
    comparator_algorithms: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["algorithm"])].append(row)
    internal_rows = grouped.get(internal_algorithm, [])

    paired: list[dict[str, Any]] = []
    for comparator in comparator_algorithms:
        comparator_rows = grouped.get(comparator, [])
        for metric_name in METRIC_SPECS:
            summary = paired_metric_summary(
                internal_rows=internal_rows,
                comparator_rows=comparator_rows,
                metric_name=metric_name,
            )
            if summary["comparable_seed_count"] == 0:
                interpretation = "comparable seed 없음"
            elif summary["internal_win"] > summary["external_win"]:
                interpretation = "internal NSGA-II 우세 경향"
            elif summary["internal_win"] < summary["external_win"]:
                interpretation = "external/comparator 우세 경향"
            else:
                interpretation = "대체로 동급 또는 혼재"
            paired.append(
                {
                    "problem": internal_rows[0]["problem"] if internal_rows else "unknown",
                    "comparator": comparator,
                    "metric": metric_name,
                    "internal_nsga2_win": summary["internal_win"],
                    "tie": summary["tie"],
                    "external_win": summary["external_win"],
                    "mean_delta": summary["mean_delta"],
                    "median_delta": summary["median_delta"],
                    "comparable_seed_count": summary["comparable_seed_count"],
                    "interpretation": interpretation,
                }
            )
    return paired


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


def _results_markdown(
    payload: dict[str, Any],
    *,
    structure_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    comparator_setup_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# External Multi-Objective Comparison Results",
        "",
        "## Repository Structure",
        "",
        *_markdown_table(
            structure_rows,
            ["item", "location", "external_comparison_relevance", "change_needed"],
        ),
        "",
        "## Comparator Protocol",
        "",
        *_markdown_table(
            protocol_rows,
            ["comparator", "target_problem", "implementation", "fairness_contract", "skip_condition", "limitation"],
        ),
        "",
        "## Comparator Setup",
        "",
        *_markdown_table(
            comparator_setup_rows,
            ["comparator", "library", "installed", "version", "executed", "skip_or_failure_reason"],
        ),
        "",
        "## Fairness",
        "",
        f"- Overall fairness status: **{payload['fairness']['status']}**",
        "",
        *_markdown_table(
            payload["fairness"]["issues"],
            ["status", "issue_type", "problem", "algorithm", "message", "severity"],
        ),
        "",
        "## Results",
        "",
        *_markdown_table(
            payload["aggregate_rows"],
            [
                "problem",
                "algorithm",
                "status",
                "seeds",
                "mean_hv",
                "std_hv",
                "median_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_nondominated_count",
                "success_rate",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Paired Summary",
        "",
        *_markdown_table(
            payload["paired_rows"],
            [
                "problem",
                "comparator",
                "metric",
                "internal_nsga2_win",
                "tie",
                "external_win",
                "mean_delta",
                "median_delta",
                "interpretation",
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def _explicit_skip_result(
    *,
    config: GAConfig,
    algorithm_name: str,
    library_name: str,
    seed: int,
    budget: int,
) -> ExternalMOComparatorResult:
    return ExternalMOComparatorResult(
        problem_name=config.problem,
        algorithm_name=algorithm_name,
        library_name=library_name,
        seed=seed,
        requested_budget=budget,
        evaluations=0,
        runtime_seconds=0.0,
        status="skipped",
        success=False,
        error_message="explicitly skipped",
        objective_vectors=[],
        nondominated_objective_vectors=[],
        metadata={},
    )


def _decorate_fairness_row(
    row: dict[str, Any],
    *,
    config: GAConfig,
    reference_point: list[float],
) -> dict[str, Any]:
    metadata = dict(row.get("metadata", {}))
    objective_count = len(config.objective_directions) if config.objective_directions else 2
    bounds = [
        float(config.representation_options.get("low", 0.0)),
        float(config.representation_options.get("high", 1.0)),
    ]
    operator_family = metadata.get("operator_family")
    if operator_family is None:
        algorithm = str(row.get("algorithm"))
        if algorithm == "internal_nsga2":
            operator_family = "internal_nsga2_arithmetic_gaussian"
        elif algorithm == "random_pareto_archive":
            operator_family = "random_pareto_archive"
        else:
            operator_family = "unknown"
    metadata.update(
        {
            "objective_count": objective_count,
            "variable_count": config.genome_length,
            "bounds": bounds,
            "reference_front_source": "analytic_zdt1",
            "hypervolume_reference_point": list(reference_point),
            "metric_postprocessing": METRIC_POSTPROCESSING_ID,
            "operator_family": operator_family,
        }
    )
    row["metadata"] = metadata
    row["problem_objectives"] = objective_count
    row["problem_variables"] = config.genome_length
    row["problem_bounds"] = bounds
    return row


def _report_markdown(
    payload: dict[str, Any],
    *,
    structure_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    comparator_setup_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, str]],
) -> str:
    lines: list[str] = [
        "# External Multi-Objective Comparison Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: internal NSGA-II와 optional external NSGA-II 구현을 같은 ZDT1 설정, 같은 budget, 같은 seed, 같은 후처리 metric으로 비교할 수 있는 protocol을 추가하는 것이다.",
        f"- 비교한 external library: {', '.join(row['library'] for row in comparator_setup_rows)}",
        f"- 실행한 문제: {payload['problem']}",
        "- 핵심 결과: 현재 환경에서는 `pymoo`, `deap`가 모두 미설치라 external comparator는 skip되었고, internal NSGA-II와 random Pareto archive anchor만 실행되었다.",
        "- internal NSGA-II 우세/열세: external 실측 비교는 이번 환경에서는 성립하지 않았고, internal anchor 기준으로만 artifact가 생성되었다.",
        "- skip된 comparator와 이유는 아래 setup 표에 명시했다.",
        "- 아직 결론 내리면 안 되는 영역: external 표준 구현 대비 상대적 우열, ZDT1 밖 benchmark 일반화, 범용 MOEA 수준 판정.",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "Scope:",
        "- ZDT1",
        "- internal NSGA-II vs optional external NSGA-II",
        "- optional external dependency comparison",
        "- 동일 metric 후처리",
        "- seed 반복",
        "- local reproducible comparison",
        "",
        "Non-Scope:",
        "- 대규모 benchmark suite",
        "- 제품화 판단",
        "- constrained multi-objective optimization",
        "- parallel/checkpoint 기능",
        "- algorithm tuning competition",
        "- 논문 수준의 통계 검정",
        "",
        "## 3. External Comparator Setup",
        "",
        *_markdown_table(
            comparator_setup_rows,
            ["comparator", "library", "installed", "version", "executed", "skip_or_failure_reason"],
        ),
        "",
        "설치 안내:",
        "- `pip install .[mo-compare]`",
        "",
        "## 4. Fairness Contract",
        "",
        f"- seed 통제: {payload['seeds']}",
        f"- evaluation budget 통제: requested budget `{payload['budget']}`",
        f"- ZDT1 dimension과 bounds: genome_length `{payload['config_summary']['genome_length']}`, bounds `{payload['config_summary']['bounds']}`",
        f"- objective direction: `{payload['config_summary']['objective_directions']}`",
        f"- reference Pareto front: ZDT1 공식 front {payload['reference_front_point_count']} points",
        f"- hypervolume reference point: `{payload['reference_point']}`",
        "- metric 후처리 통일: internal / baseline / external 후보 모두 `src/ga_lab/experiment/mo_metrics.py`를 사용했다.",
        "- runtime 기록: seed별 runtime_seconds 기록",
        "- paired comparison 방식: 같은 seed끼리 metric 차이를 직접 비교",
        "- budget mismatch 처리: actual_evaluations를 별도 기록하고, mismatch가 있으면 보고서 해석을 보수적으로 유지한다.",
        "",
        "## 5. Comparator Implementations",
        "",
        *_markdown_table(
            protocol_rows,
            ["comparator", "target_problem", "implementation", "fairness_contract", "skip_condition", "limitation"],
        ),
        "",
        "## 6. Experiment Configuration",
        "",
        *_markdown_table(
            payload["config_rows"],
            ["problem", "algorithm", "config", "seeds", "requested_budget", "actual_evaluations", "metric"],
        ),
        "",
        "## 7. Results",
        "",
        *_markdown_table(
            payload["aggregate_rows"],
            [
                "problem",
                "algorithm",
                "status",
                "seeds",
                "mean_hv",
                "std_hv",
                "median_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_nondominated_count",
                "success_rate",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## 8. Paired Summary",
        "",
        *_markdown_table(
            payload["paired_rows"],
            [
                "problem",
                "comparator",
                "metric",
                "internal_nsga2_win",
                "tie",
                "external_win",
                "mean_delta",
                "median_delta",
                "interpretation",
            ],
        ),
        "",
        "## Fairness Summary",
        "",
        f"- Overall fairness status: **{payload['fairness']['status']}**",
        "",
        *_markdown_table(
            payload["fairness"]["issues"],
            ["status", "issue_type", "problem", "algorithm", "message", "severity"],
        ),
        "",
        "## 9. What External Comparison Changed",
        "",
        "- 이번 패스는 external comparator가 없는 환경에서도 skip을 artifact로 남기는 절차를 만들었다.",
        "- 따라서 지금 이 환경에서 바뀐 것은 성능 결론이 아니라, 표준 구현과의 비교를 같은 metric/seed/budget으로 재현할 수 있는 감사 경로다.",
        "- internal NSGA-II가 표준 구현과 비슷한지, 약한지, 우연히 강한지는 external library가 실제로 설치된 환경에서만 판단할 수 있다.",
        "",
        "## 10. Failures, Skips, and Exceptions",
        "",
        *_markdown_table(
            [
                {
                    "comparator": row["comparator"],
                    "library": row["library"],
                    "status": "skipped" if not row["installed"] else "executed",
                    "reason": row["skip_or_failure_reason"],
                }
                for row in comparator_setup_rows
            ],
            ["comparator", "library", "status", "reason"],
        ),
        "",
        "## 11. Regression Check",
        "",
        *_markdown_table(regression_rows, ["command", "result", "note"]),
        "",
        "## 12. Maturity Impact",
        "",
        "- 결론: **Level 판정 유지**",
        "- 판단 기준: external comparator가 실제로 skip된 환경에서는 Level 상향을 주장하지 않는다.",
        "- 이번 패스는 Level 4 근거를 간접 강화할 수 있는 준비 단계이지만, 실측 external comparison 없이 레벨을 올리진 않는다.",
        "",
        "## 13. Recommended Next Work",
        "",
        "1. ZDT2/ZDT3/DTLZ small suite 확장: ZDT1 하나로는 external comparator sanity가 좁다. 난이도 중, 효과 높음.",
        "2. NSGA-II parameter fairness 정교화: pop/generation/evaluation mapping을 internal/external 모두 더 세밀하게 맞춰야 한다. 난이도 중, 효과 높음.",
        "3. external library comparison 반복 자동화: optional dependency가 있는 환경에서 반복 실행 artifact를 자동화해야 한다. 난이도 중, 효과 중.",
        "4. constrained multi-objective contract: 현재는 unconstrained smooth benchmark만 본다. 난이도 상, 효과 높음.",
        "5. checkpoint/resume: 긴 multi-objective 반복 실험의 재현성과 운영 안정성을 높인다. 난이도 중, 효과 중.",
        "6. parallel evaluation: seed 반복과 external comparator 포함 시 throughput 병목을 줄인다. 난이도 중, 효과 중.",
        "7. domain-specific multi-objective benchmark expansion: toy benchmark 이후 실제 적용성 검증이 필요하다. 난이도 상, 효과 높음.",
        "",
        "“이번 external comparison 결과, internal NSGA-II는 동일 metric 후처리와 seed/budget protocol 기준에서 external comparator 대비 직접 실측되지는 않았으며, ZDT1 기반 external sanity를 주장하기 전에는 optional library 설치 환경에서의 반복 검증이 추가로 필요하다.”",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    budget = args.budget or configured_evaluation_budget(config)
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    reference_point = [
        float(value)
        for value in config.algorithm_options.get("hypervolume_reference_point", [1.1, 11.0])
    ]
    reference_front = zdt1_reference_front(201)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = PROJECT_ROOT / args.output_root / f"{timestamp}_{config.problem}_external_compare"
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.artifact_suffix:
        safe_suffix = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in str(args.artifact_suffix).strip()
        ).strip("_")
        if safe_suffix:
            suffix = f"_{safe_suffix}"

    structure_rows = [
        {
            "item": "ZDT1 problem definition",
            "location": "src/ga_lab/problems/zdt1.py",
            "external_comparison_relevance": "same decision dimension, bounds, direction anchor",
            "change_needed": "no",
        },
        {
            "item": "Internal NSGA-II loop",
            "location": "src/ga_lab/algorithms/nsga2.py",
            "external_comparison_relevance": "internal comparator target",
            "change_needed": "no",
        },
        {
            "item": "MO metric layer",
            "location": "src/ga_lab/experiment/mo_metrics.py",
            "external_comparison_relevance": "shared post-processing for internal/external fronts",
            "change_needed": "yes",
        },
        {
            "item": "MO baseline layer",
            "location": "src/ga_lab/experiment/mo_baselines.py",
            "external_comparison_relevance": "random archive anchor reuse",
            "change_needed": "no",
        },
        {
            "item": "Existing MO baseline runner",
            "location": "scripts/compare_mo_baselines.py",
            "external_comparison_relevance": "budget/reference point/reporting template",
            "change_needed": "no",
        },
        {
            "item": "Optional dependency manifest",
            "location": "pyproject.toml",
            "external_comparison_relevance": "optional install path for pymoo/deap",
            "change_needed": "yes",
        },
    ]
    fairness_text = (
        "same ZDT1 definition, same seed list, same requested evaluation budget, "
        "same bounds/direction/reference point, same mo_metrics post-processing"
    )
    protocol_rows = [
        {
            "comparator": "internal_nsga2",
            "target_problem": "zdt1",
            "implementation": "existing internal NSGA-II via ga_lab.api.run_config",
            "fairness_contract": fairness_text,
            "skip_condition": "none",
            "limitation": "internal budget formula differs from standard NSGA-II generation accounting",
        },
        {
            "comparator": "random_pareto_archive",
            "target_problem": "zdt1",
            "implementation": "existing random archive anchor",
            "fairness_contract": fairness_text,
            "skip_condition": "--skip-random-archive",
            "limitation": "not an external standard NSGA-II implementation",
        },
        {
            "comparator": "pymoo_nsga2",
            "target_problem": "zdt1",
            "implementation": "optional pymoo NSGA2 on wrapped internal ZDT1 evaluator",
            "fairness_contract": fairness_text,
            "skip_condition": "pymoo missing or --skip-pymoo",
            "limitation": "operator family differs from internal arithmetic+gaussian setup",
        },
        {
            "comparator": "deap_nsga2",
            "target_problem": "zdt1",
            "implementation": "optional DEAP selNSGA2 comparator on wrapped internal ZDT1 evaluator",
            "fairness_contract": fairness_text,
            "skip_condition": "deap missing or --skip-deap",
            "limitation": "fairness depends on DEAP standard SBX/polynomial operator path",
        },
    ]

    pymoo_status = optional_library_status("pymoo")
    deap_status = optional_library_status("deap")
    comparator_setup_rows = [
        {
            "comparator": "pymoo_nsga2",
            "library": "pymoo",
            "installed": pymoo_status.installed,
            "version": pymoo_status.version,
            "executed": (not args.skip_pymoo) and pymoo_status.installed,
            "skip_or_failure_reason": "explicitly skipped"
            if args.skip_pymoo
            else pymoo_status.reason,
        },
        {
            "comparator": "deap_nsga2",
            "library": "deap",
            "installed": deap_status.installed,
            "version": deap_status.version,
            "executed": (not args.skip_deap) and deap_status.installed,
            "skip_or_failure_reason": "explicitly skipped" if args.skip_deap else deap_status.reason,
        },
    ]

    raw_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        internal_result = run_internal_nsga2(config, seed=seed, output_root=str(output_root))
        internal_row = result_to_front_row(
            internal_result,
            reference_front=reference_front,
            reference_point=reference_point,
        )
        internal_row = _decorate_fairness_row(
            internal_row,
            config=config,
            reference_point=reference_point,
        )
        raw_rows.append(internal_row)

        if not args.skip_random_archive:
            random_result = run_random_archive_anchor(
                run_random_pareto_archive(config, seed=seed, budget=budget)
            )
            raw_rows.append(
                _decorate_fairness_row(
                    result_to_front_row(
                    random_result,
                    reference_front=reference_front,
                    reference_point=reference_point,
                    ),
                    config=config,
                    reference_point=reference_point,
                )
            )

        if args.skip_pymoo:
            pymoo_result = _explicit_skip_result(
                config=config,
                algorithm_name="pymoo_nsga2",
                library_name="pymoo",
                seed=seed,
                budget=budget,
            )
            raw_rows.append(
                _decorate_fairness_row(
                    result_to_front_row(
                    pymoo_result,
                    reference_front=reference_front,
                    reference_point=reference_point,
                    ),
                    config=config,
                    reference_point=reference_point,
                )
            )
        else:
            raw_rows.append(
                _decorate_fairness_row(
                    result_to_front_row(
                    run_pymoo_nsga2(config, seed=seed, budget=budget),
                    reference_front=reference_front,
                    reference_point=reference_point,
                    ),
                    config=config,
                    reference_point=reference_point,
                )
            )

        if args.skip_deap:
            deap_result = _explicit_skip_result(
                config=config,
                algorithm_name="deap_nsga2",
                library_name="deap",
                seed=seed,
                budget=budget,
            )
            raw_rows.append(
                _decorate_fairness_row(
                    result_to_front_row(
                    deap_result,
                    reference_front=reference_front,
                    reference_point=reference_point,
                    ),
                    config=config,
                    reference_point=reference_point,
                )
            )
        else:
            raw_rows.append(
                _decorate_fairness_row(
                    result_to_front_row(
                    run_deap_nsga2(config, seed=seed, budget=budget),
                    reference_front=reference_front,
                    reference_point=reference_point,
                    ),
                    config=config,
                    reference_point=reference_point,
                )
            )

    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[row["algorithm"]].append(row)

    for algorithm, bucket in sorted(grouped.items()):
        actual_evaluations = _summary_stat([row for row in bucket if row.get("success")], "actual_evaluations")[
            "mean"
        ]
        config_rows.append(
            {
                "problem": config.problem,
                "algorithm": algorithm,
                "config": (
                    f"pop={config.population_size}, gen={config.generations}, "
                    f"genome={config.genome_length}, bounds={config.representation_options}"
                ),
                "seeds": len(bucket),
                "requested_budget": budget,
                "actual_evaluations": actual_evaluations,
                "metric": "HV, reference_front_distance, GD, IGD, spacing, nondominated_count",
            }
        )

    aggregate_rows = _aggregate_rows(raw_rows)
    fairness_payload = evaluate_parameter_fairness(
        [row for row in raw_rows if row.get("status") != "skipped"],
        benchmark_rows=[
            {
                "problem": config.problem,
                "source": "internal_problem_registry",
                "objectives": len(config.objective_directions) if config.objective_directions else 2,
                "variables": config.genome_length,
                "bounds": [
                    float(config.representation_options.get("low", 0.0)),
                    float(config.representation_options.get("high", 1.0)),
                ],
                "reference_front": "analytic_zdt1",
                "hv_reference_point": list(reference_point),
                "metric_limitations": "hypervolume_2d only for 2-objective problems",
            }
        ],
    )
    comparator_algorithms = ["pymoo_nsga2", "deap_nsga2"]
    if not args.skip_random_archive:
        comparator_algorithms.append("random_pareto_archive")
    paired_rows = _paired_rows(
        raw_rows,
        internal_algorithm="internal_nsga2",
        comparator_algorithms=comparator_algorithms,
    )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "artifact_suffix": args.artifact_suffix,
        "config_path": str((PROJECT_ROOT / args.config).resolve()),
        "problem": config.problem,
        "seeds": seeds,
        "budget": budget,
        "reference_point": reference_point,
        "reference_front_point_count": len(reference_front),
        "config_summary": {
            "population_size": config.population_size,
            "generations": config.generations,
            "genome_length": config.genome_length,
            "bounds": dict(config.representation_options),
            "objective_directions": list(config.objective_directions),
            "requested_budget_formula": "population_size * (3 * generations + 2)",
        },
        "library_status": {
            "pymoo": pymoo_status.to_dict(),
            "deap": deap_status.to_dict(),
        },
        "structure_rows": structure_rows,
        "protocol_rows": protocol_rows,
        "comparator_setup_rows": comparator_setup_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "config_rows": config_rows,
        "fairness": fairness_payload,
        "runtime_seconds": time.perf_counter() - started,
    }

    results_json_path = artifact_root / f"external_mo_comparison_results{suffix}.json"
    results_csv_path = artifact_root / f"external_mo_comparison_results{suffix}.csv"
    results_md_path = artifact_root / f"external_mo_comparison_results{suffix}.md"
    report_md_path = artifact_root / f"external_mo_comparison_report{suffix}.md"

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
            "std_hv",
            "median_hv",
            "mean_distance",
            "mean_igd",
            "mean_spacing",
            "mean_nondominated_count",
            "success_rate",
            "mean_runtime_seconds",
            "mean_actual_evaluations",
        ],
    )
    results_md_path.write_text(
        _results_markdown(
            payload,
            structure_rows=structure_rows,
            protocol_rows=protocol_rows,
            comparator_setup_rows=comparator_setup_rows,
        ),
        encoding="utf-8",
    )

    regression_rows = [
        {
            "command": "python -m pytest tests/test_mutation_contract.py tests/test_fitness_validation.py -q",
            "result": "pending",
            "note": "fill after regression run",
        },
        {
            "command": "python -m pytest tests/test_nsga2.py tests/test_direction_contracts.py -q",
            "result": "pending",
            "note": "fill after regression run",
        },
        {
            "command": "python -m pytest tests/test_mo_metrics.py tests/test_mo_baselines.py -q",
            "result": "pending",
            "note": "fill after regression run",
        },
        {
            "command": "python -m pytest tests/test_baseline_algorithms.py -q",
            "result": "pending",
            "note": "fill after regression run",
        },
        {
            "command": "python -m pytest tests/test_external_mo_comparators.py -q",
            "result": "pending",
            "note": "fill after regression run",
        },
        {
            "command": "python audit/ga_execution_audit.py",
            "result": "pending",
            "note": "fill after audit smoke",
        },
        {
            "command": "python scripts/check_local_baseline.py",
            "result": "pending",
            "note": "fill after governance smoke",
        },
        {
            "command": "python scripts/compare_mo_baselines.py",
            "result": "pending",
            "note": "fill after MO baseline rerun",
        },
        {
            "command": "python scripts/compare_external_mo.py",
            "result": "PASS",
            "note": "this command generated the current artifact",
        },
    ]
    report_md_path.write_text(
        _report_markdown(
            payload,
            structure_rows=structure_rows,
            protocol_rows=protocol_rows,
            comparator_setup_rows=comparator_setup_rows,
            regression_rows=regression_rows,
        ),
        encoding="utf-8",
    )

    timestamp_root = output_root / "artifact_snapshot"
    timestamp_root.mkdir(parents=True, exist_ok=True)
    _write_json(timestamp_root / "external_mo_comparison_results.json", payload)

    print(results_json_path)
    print(report_md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
