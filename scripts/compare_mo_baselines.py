 # ruff: noqa: E402

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

from ga_lab.api import run_config
from ga_lab.config import GAConfig, load_config
from ga_lab.experiment.budget_baseline_comparison import configured_evaluation_budget
from ga_lab.experiment.mo_baselines import (
    MultiObjectiveBaselineResult,
    run_random_pareto_archive,
    run_weighted_sum_random_archive,
)
from ga_lab.experiment.mo_metrics import (
    coverage_indicator,
    evaluate_front_metrics,
    zdt1_reference_front,
)
from ga_lab.experiment.parameter_fairness import (
    METRIC_POSTPROCESSING_ID,
    evaluate_parameter_fairness,
)


METRIC_SPECS: dict[str, dict[str, Any]] = {
    "hypervolume_2d": {"higher_is_better": True},
    "reference_front_distance": {"higher_is_better": False},
    "spacing": {"higher_is_better": False},
    "nondominated_count": {"higher_is_better": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NSGA-II against simple multi-objective baselines on ZDT1."
    )
    parser.add_argument(
        "--config",
        default="configs/smoke/zdt1_nsga2_smoke.json",
        help="ZDT1 NSGA-II config used as the comparison anchor.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/mo_baseline_comparison",
        help="Directory for timestamped raw comparison outputs.",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Directory for stable MO comparison artifacts.",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of repeated seeds.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=6101,
        help="First seed for the repeated comparison.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Optional explicit evaluation budget override.",
    )
    parser.add_argument(
        "--skip-weighted-sum",
        action="store_true",
        help="Skip the auxiliary weighted-sum random baseline.",
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
        return {
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
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
    successes = [bool(row.get("success")) for row in rows]
    return sum(1 for success in successes if success) / len(successes)


def _reference_point(config: GAConfig) -> list[float]:
    explicit = config.algorithm_options.get("hypervolume_reference_point")
    if isinstance(explicit, list):
        return [float(value) for value in explicit]
    return [1.1, 11.0]


def _coverage_fields(
    nsga_front: list[list[float]],
    baseline_front: list[list[float]],
    directions: list[bool],
) -> dict[str, float]:
    return {
        "coverage_nsga_over_baseline": coverage_indicator(nsga_front, baseline_front, directions),
        "coverage_baseline_over_nsga": coverage_indicator(baseline_front, nsga_front, directions),
    }


def _nsga2_row(
    *,
    config: GAConfig,
    seed: int,
    budget: int,
    reference_front: list[list[float]],
    reference_point: list[float],
    output_root: Path,
) -> dict[str, Any]:
    run_config_payload = GAConfig.from_dict(config.to_dict())
    run_config_payload.seed = seed
    run_config_payload.run_name = f"{config.run_name}_seed{seed}"
    started = time.perf_counter()
    result = run_config(run_config_payload, output_root=output_root)
    runtime_seconds = time.perf_counter() - started
    summary = result.raw_summary
    directions = [bool(value) for value in summary["objective_directions"]]
    front = [list(vector) for vector in summary["pareto_front_vectors"]]
    metrics = evaluate_front_metrics(
        front,
        directions=directions,
        reference_front=reference_front,
        reference_point=reference_point,
    )
    return {
        "problem": config.problem,
        "algorithm": "NSGA-II",
        "seed": seed,
        "budget": budget,
        "evaluations": int(summary["actual_evaluations_used"]),
        "runtime_seconds": float(
            summary["runtime_seconds"] if "runtime_seconds" in summary else runtime_seconds
        ),
        "success": True,
        "error_message": None,
        "objective_vectors": front,
        "nondominated_objective_vectors": front,
        "output_dir": result.output_dir,
        "summary_path": result.summary_path,
        "objective_directions": directions,
        "hypervolume_reference_point": reference_point,
        "metadata": {
            "configured_population_size": config.population_size,
            "configured_generations": config.generations,
            "pareto_ratio": summary.get("pareto_ratio"),
            "reported_hypervolume": summary.get("hypervolume"),
            "reported_spread": summary.get("spread"),
            "reported_pareto_front_size": summary.get("pareto_front_size"),
        },
        **metrics,
    }


def _baseline_row(
    result: MultiObjectiveBaselineResult,
    *,
    config: GAConfig,
    reference_front: list[list[float]],
    reference_point: list[float],
) -> dict[str, Any]:
    directions = [
        bool(value)
        for value in result.metadata.get("objective_directions", config.objective_directions)
    ]
    metrics = evaluate_front_metrics(
        result.objective_vectors,
        directions=directions,
        reference_front=reference_front,
        reference_point=reference_point,
    )
    return {
        "problem": result.problem_name,
        "algorithm": result.algorithm_name,
        "seed": result.seed,
        "budget": result.budget,
        "evaluations": result.evaluations,
        "runtime_seconds": result.runtime_seconds,
        "success": result.success,
        "error_message": result.error_message,
        "objective_vectors": result.objective_vectors,
        "nondominated_objective_vectors": result.nondominated_objective_vectors,
        "objective_directions": directions,
        "hypervolume_reference_point": reference_point,
        "metadata": result.metadata,
        **metrics,
    }


def _decorate_fairness_row(
    row: dict[str, Any],
    *,
    config: GAConfig,
    reference_front_source: str,
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
        if row["algorithm"] == "NSGA-II":
            operator_family = "internal_nsga2_arithmetic_gaussian"
        elif row["algorithm"] == "weighted_sum_random_archive":
            operator_family = "weighted_sum_random_archive"
        else:
            operator_family = "random_pareto_archive"
    metadata.update(
        {
            "objective_count": objective_count,
            "variable_count": config.genome_length,
            "bounds": bounds,
            "reference_front_source": reference_front_source,
            "hypervolume_reference_point": list(reference_point),
            "metric_postprocessing": METRIC_POSTPROCESSING_ID,
            "operator_family": operator_family,
        }
    )
    row["metadata"] = metadata
    row["problem_objectives"] = objective_count
    row["problem_variables"] = config.genome_length
    row["problem_bounds"] = bounds
    if "requested_budget" not in row and "budget" in row:
        row["requested_budget"] = row["budget"]
    if "actual_evaluations" not in row and "evaluations" in row:
        row["actual_evaluations"] = row["evaluations"]
    return row


def _aggregate_algorithm_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    aggregates: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        hv = _summary_stat(bucket, "hypervolume_2d")
        distance = _summary_stat(bucket, "reference_front_distance")
        spacing = _summary_stat(bucket, "spacing")
        nondominated = _summary_stat(bucket, "nondominated_count")
        runtime_summary = _summary_stat(bucket, "runtime_seconds")
        evaluations_summary = _summary_stat(bucket, "evaluations")
        coverage_left = _summary_stat(bucket, "coverage_nsga_over_baseline")
        coverage_right = _summary_stat(bucket, "coverage_baseline_over_nsga")
        aggregates.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "seeds": len(bucket),
                "mean_hv": hv["mean"],
                "std_hv": hv["std"],
                "median_hv": hv["median"],
                "mean_distance": distance["mean"],
                "mean_spacing": spacing["mean"],
                "mean_nondominated_count": nondominated["mean"],
                "success_rate": _success_rate(bucket),
                "mean_runtime_seconds": runtime_summary["mean"],
                "mean_evaluations": evaluations_summary["mean"],
                "mean_nsga_over_baseline_coverage": coverage_left["mean"],
                "mean_baseline_over_nsga_coverage": coverage_right["mean"],
            }
        )
    return aggregates


def _paired_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_algorithm: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    problem = None
    for row in rows:
        by_algorithm[str(row["algorithm"])][int(row["seed"])] = row
        problem = str(row["problem"])

    nsga_rows = by_algorithm.get("NSGA-II", {})
    paired_rows: list[dict[str, Any]] = []
    for algorithm, bucket in sorted(by_algorithm.items()):
        if algorithm == "NSGA-II":
            continue
        common_seeds = sorted(set(nsga_rows) & set(bucket))
        for metric, spec in METRIC_SPECS.items():
            wins = 0
            ties = 0
            losses = 0
            diffs: list[float] = []
            for seed in common_seeds:
                left = nsga_rows[seed].get(metric)
                right = bucket[seed].get(metric)
                if not (
                    isinstance(left, int | float)
                    and math.isfinite(float(left))
                    and isinstance(right, int | float)
                    and math.isfinite(float(right))
                ):
                    continue
                difference = float(left) - float(right)
                diffs.append(difference)
                if spec["higher_is_better"]:
                    if difference > 0:
                        wins += 1
                    elif difference < 0:
                        losses += 1
                    else:
                        ties += 1
                else:
                    if difference < 0:
                        wins += 1
                    elif difference > 0:
                        losses += 1
                    else:
                        ties += 1
            interpretation = "경향 불명"
            if wins > losses:
                interpretation = "NSGA-II 우세 경향"
            elif losses > wins:
                interpretation = "baseline 우세 또는 최소 동급"
            paired_rows.append(
                {
                    "problem": problem,
                    "baseline": algorithm,
                    "metric": metric,
                    "nsga_win": wins,
                    "tie": ties,
                    "nsga_loss": losses,
                    "mean_delta": mean(diffs) if diffs else None,
                    "median_delta": median(diffs) if diffs else None,
                    "interpretation": interpretation,
                }
            )
    return paired_rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        rendered: list[str] = []
        for column in columns:
            value = row.get(column)
            if value is None:
                rendered.append("n/a")
            elif isinstance(value, float):
                rendered.append(f"{value:.4f}" if math.isfinite(value) else "n/a")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def _results_markdown(
    *,
    generated_at: str,
    command: str,
    config_path: Path,
    reference_point: list[float],
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
) -> str:
    lines = [
        "# Multi-Objective Baseline Comparison Results",
        "",
        f"- Generated at: {generated_at}",
        f"- Command: `{command}`",
        f"- Config: `{config_path.as_posix()}`",
        f"- Hypervolume reference point: `{reference_point}`",
        "",
        "## Fairness",
        "",
        f"- Overall fairness status: **{fairness_payload['status']}**",
        "",
        *_markdown_table(
            fairness_payload["issues"],
            ["status", "issue_type", "problem", "algorithm", "message", "severity"],
        ),
        "",
        "## Results",
        "",
        "| 문제 | 알고리즘 | seeds | mean HV | std HV | median HV | mean distance | mean spacing | mean nondominated count | success rate | mean runtime |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        success_rate = row["success_rate"]
        lines.append(
            "| {problem} | {algorithm} | {seeds} | {mean_hv:.4f} | {std_hv:.4f} | {median_hv:.4f} | {mean_distance:.4f} | {mean_spacing:.4f} | {mean_nd:.2f} | {success_rate} | {runtime:.4f} |".format(
                problem=row["problem"],
                algorithm=row["algorithm"],
                seeds=row["seeds"],
                mean_hv=row["mean_hv"] if row["mean_hv"] is not None else float("nan"),
                std_hv=row["std_hv"] if row["std_hv"] is not None else float("nan"),
                median_hv=row["median_hv"] if row["median_hv"] is not None else float("nan"),
                mean_distance=row["mean_distance"] if row["mean_distance"] is not None else float("nan"),
                mean_spacing=row["mean_spacing"] if row["mean_spacing"] is not None else float("nan"),
                mean_nd=row["mean_nondominated_count"] if row["mean_nondominated_count"] is not None else float("nan"),
                success_rate=f"{100.0 * success_rate:.1f}%" if success_rate is not None else "n/a",
                runtime=row["mean_runtime_seconds"] if row["mean_runtime_seconds"] is not None else float("nan"),
            )
        )

    lines.extend(
        [
            "",
            "## Paired Summary",
            "",
            "| 문제 | baseline | metric | NSGA-II win | tie | NSGA-II loss | 평균 차이 | 중앙값 차이 | 해석 |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| {problem} | {baseline} | {metric} | {nsga_win} | {tie} | {nsga_loss} | {mean_delta:.4f} | {median_delta:.4f} | {interpretation} |".format(
                problem=row["problem"],
                baseline=row["baseline"],
                metric=row["metric"],
                nsga_win=row["nsga_win"],
                tie=row["tie"],
                nsga_loss=row["nsga_loss"],
                mean_delta=row["mean_delta"] if row["mean_delta"] is not None else float("nan"),
                median_delta=row["median_delta"] if row["median_delta"] is not None else float("nan"),
                interpretation=row["interpretation"],
            )
        )

    lines.extend(["", "## Failures and Exceptions", ""])
    if failures:
        for failure in failures:
            lines.append(
                f"- `{failure['algorithm']}` / seed `{failure['seed']}`: {failure['error_message']}"
            )
    else:
        lines.append("- 없음")
    lines.append("")
    return "\n".join(lines)


def _report_markdown(
    *,
    generated_at: str,
    command: str,
    config: GAConfig,
    config_path: Path,
    seeds: list[int],
    budget: int,
    reference_point: list[float],
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    regression_results: list[dict[str, str]],
    weighted_sum_enabled: bool,
) -> str:
    lines = [
        "# Multi-Objective Baseline Comparison Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: ZDT1 기준으로 NSGA-II와 단순 multi-objective baseline을 같은 evaluation budget에서 반복 비교하는 프로토콜을 만들고 실행하는 것이다.",
        f"- 구현한 baseline: `random_pareto_archive`{', `weighted_sum_random`' if weighted_sum_enabled else ''}.",
        "- 실행한 문제: `zdt1` 1개, 2-objective continuous minimization.",
        "- 핵심 해석: NSGA-II가 random Pareto archive보다 hypervolume과 참조 front 거리에서 우세한지, 그리고 그 우세가 seed 반복에서 얼마나 일관적인지 본다.",
        "- 아직 결론 내리면 안 되는 영역: ZDT1 밖의 multi-objective benchmark 전반, external library 비교, constrained MOEA.",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "Scope:",
        "- ZDT1",
        "- NSGA-II vs simple multi-objective baseline",
        "- seed 반복",
        "- local reproducible comparison",
        "- front-level metric",
        "",
        "Non-Scope:",
        "- 대규모 multi-objective benchmark suite",
        "- external library comparison",
        "- constrained multi-objective optimization",
        "- 제품화 판단",
        "- parallel/checkpoint 기능",
        "",
        "## 3. Multi-Objective Fairness Contract",
        "",
        f"- seed 통제: `{seeds}`",
        f"- evaluation budget 통제: 모든 알고리즘에 seed당 `{budget}` evaluation",
        f"- objective direction 통일: `{config.objective_directions}` (ZDT1 minimization)",
        f"- hypervolume reference point: `{reference_point}`",
        "- reference Pareto front 생성 방식: ZDT1 공식 front `f2 = 1 - sqrt(f1)`를 201개 point로 균등 샘플링",
        "- 실패 처리: 예외와 실패 seed를 숨기지 않고 결과에 기록",
        "- runtime 기록: seed별 runtime_seconds 포함",
        "- paired comparison 방식: 같은 seed에서 NSGA-II와 baseline metric 차이를 직접 비교",
        "",
        "## 4. Baseline Implementations",
        "",
        "| baseline | 파일 | 지원 문제 | budget 기준 | 한계 |",
        "|---|---|---|---|---|",
        f"| `random_pareto_archive` | `src/ga_lab/experiment/mo_baselines.py` | `zdt1` | random sampling `{budget}` eval | archive quality는 완전 랜덤 샘플 품질에 의존 |",
    ]
    if weighted_sum_enabled:
        lines.append(
            f"| `weighted_sum_random` | `src/ga_lab/experiment/mo_baselines.py` | `zdt1` | total `{budget}` eval을 weight grid에 분할 | non-convex front와 diversity 측면에서 구조적 한계 |"
        )

    lines.extend(
        [
            "",
            "## 5. Metric Definitions",
            "",
            "| metric | 의미 | 높을수록 좋은가 | 계산 위치 | 한계 |",
            "|---|---|---:|---|---|",
            "| `hypervolume_2d` | reference point 대비 dominated area | 예 | `src/ga_lab/experiment/mo_metrics.py` | 2-objective 기준에만 구현 |",
            "| `reference_front_distance` | 얻은 front가 참조 front에 얼마나 가까운지 | 아니오 | `src/ga_lab/experiment/mo_metrics.py` | 참조 front 샘플 밀도에 영향 받음 |",
            "| `spacing` | nondominated front 간격의 균일성 | 아니오 | `src/ga_lab/experiment/mo_metrics.py` | front 크기가 너무 작으면 해석력이 약함 |",
            "| `nondominated_count` | 최종 nondominated point 수 | 예 | `src/ga_lab/experiment/mo_metrics.py` | 수만 많고 품질이 나쁠 수 있음 |",
            "| `coverage_indicator` | 한 front가 다른 front를 지배하는 비율 | 예 | `src/ga_lab/experiment/mo_metrics.py` | seed별 상대 비교 보조 지표 |",
            "",
            "## 6. Experiment Configuration",
            "",
            "| 문제 | 알고리즘 | 설정 | seeds | budget | metric |",
            "|---|---|---|---:|---:|---|",
            f"| `zdt1` | `NSGA-II` | pop `{config.population_size}`, gen `{config.generations}`, genome `{config.genome_length}` | {len(seeds)} | {budget} | HV, reference_front_distance, spacing, nondominated_count |",
            f"| `zdt1` | `random_pareto_archive` | uniform random archive sampling | {len(seeds)} | {budget} | HV, reference_front_distance, spacing, nondominated_count |",
        ]
    )
    if weighted_sum_enabled:
        lines.append(
            f"| `zdt1` | `weighted_sum_random` | random sampling split across weighted sums | {len(seeds)} | {budget} | HV, reference_front_distance, spacing, nondominated_count |"
        )

    lines.extend(
        [
            "",
            "## 7. Results",
            "",
            "| 문제 | 알고리즘 | seeds | mean HV | std HV | median HV | mean distance | mean spacing | mean nondominated count | success rate | mean runtime |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        success_rate = row["success_rate"]
        lines.append(
            "| {problem} | {algorithm} | {seeds} | {mean_hv:.4f} | {std_hv:.4f} | {median_hv:.4f} | {mean_distance:.4f} | {mean_spacing:.4f} | {mean_nd:.2f} | {success_rate} | {runtime:.4f} |".format(
                problem=row["problem"],
                algorithm=row["algorithm"],
                seeds=row["seeds"],
                mean_hv=row["mean_hv"] if row["mean_hv"] is not None else float("nan"),
                std_hv=row["std_hv"] if row["std_hv"] is not None else float("nan"),
                median_hv=row["median_hv"] if row["median_hv"] is not None else float("nan"),
                mean_distance=row["mean_distance"] if row["mean_distance"] is not None else float("nan"),
                mean_spacing=row["mean_spacing"] if row["mean_spacing"] is not None else float("nan"),
                mean_nd=row["mean_nondominated_count"] if row["mean_nondominated_count"] is not None else float("nan"),
                success_rate=f"{100.0 * success_rate:.1f}%" if success_rate is not None else "n/a",
                runtime=row["mean_runtime_seconds"] if row["mean_runtime_seconds"] is not None else float("nan"),
            )
        )

    lines.extend(
        [
            "",
            "## 8. Paired Summary",
            "",
            "| 문제 | baseline | metric | NSGA-II win | tie | NSGA-II loss | 평균 차이 | 중앙값 차이 | 해석 |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| {problem} | {baseline} | {metric} | {nsga_win} | {tie} | {nsga_loss} | {mean_delta:.4f} | {median_delta:.4f} | {interpretation} |".format(
                problem=row["problem"],
                baseline=row["baseline"],
                metric=row["metric"],
                nsga_win=row["nsga_win"],
                tie=row["tie"],
                nsga_loss=row["nsga_loss"],
                mean_delta=row["mean_delta"] if row["mean_delta"] is not None else float("nan"),
                median_delta=row["median_delta"] if row["median_delta"] is not None else float("nan"),
                interpretation=row["interpretation"],
            )
        )

    lines.extend(["", "## 9. Failures and Exceptions", ""])
    if failures:
        for failure in failures:
            lines.append(
                f"- `{failure['algorithm']}` / seed `{failure['seed']}`: {failure['error_message']}"
            )
    else:
        lines.append("- 실패한 seed는 없었다.")

    lines.extend(
        [
            "",
            "## 10. Regression Check",
            "",
            "| 명령 | 결과 | 비고 |",
            "|---|---|---|",
        ]
    )
    for row in regression_results:
        lines.append(f"| `{row['command']}` | {row['result']} | {row['note']} |")

    lines.extend(
        [
            "",
            "## 11. Maturity Impact",
            "",
            "- 결론: **Level 4 근거 강화**",
            "- 이유: ZDT1 기준으로 NSGA-II와 simple MO baseline을 같은 budget/seed에서 반복 비교할 수 있는 프로토콜이 생겼다.",
            "- 보수적 한계: ZDT1 1개 문제와 local simple baselines만으로 범용 multi-objective optimizer나 Level 5를 주장할 수는 없다.",
            "",
            "## 12. Recommended Next Work",
            "",
            "1. external library comparison, 예: pymoo, DEAP",
            "   - 이유: local baseline보다 더 강한 참조선이 필요하다.",
            "   - 난이도: 중간",
            "   - 기대 효과: NSGA-II 구현의 상대적 위치를 더 객관적으로 확인할 수 있다.",
            "2. ZDT2, ZDT3, DTLZ small suite 확장",
            "   - 이유: ZDT1 하나로는 front shape 일반화가 부족하다.",
            "   - 난이도: 중간",
            "   - 기대 효과: non-convex/front discontinuity 대응을 볼 수 있다.",
            "3. constrained multi-objective contract",
            "   - 이유: 현재는 unconstrained smooth benchmark 중심이다.",
            "   - 난이도: 높음",
            "   - 기대 효과: 실제 적용 범위 해석이 더 명확해진다.",
            "4. checkpoint/resume",
            "   - 이유: 더 긴 MO benchmark에서 재현성과 유지보수성이 좋아진다.",
            "   - 난이도: 중간",
            "   - 기대 효과: 긴 seed 반복 실험의 운영 안정성이 올라간다.",
            "5. parallel evaluation",
            "   - 이유: seed 반복과 population evaluation 비용을 줄일 수 있다.",
            "   - 난이도: 중간",
            "   - 기대 효과: benchmark throughput 향상.",
            "6. domain-specific multi-objective benchmark expansion",
            "   - 이유: toy benchmark 이후의 적용 범위를 확인해야 한다.",
            "   - 난이도: 높음",
            "   - 기대 효과: 실제 문제 적합성 판단 근거 강화.",
            "",
            "“이번 multi-objective baseline 패스 결과, 현재 NSGA-II 경로는 ZDT1 같은 smooth 2-objective continuous minimization 문제에서 simple random Pareto archive baseline 대비 우세 여부를 반복 검증할 수 있는 상태에 도달했으며, 더 넓은 multi-objective benchmark 영역으로 확장하기 전에는 external library 비교와 benchmark suite 확장이 추가로 필요하다.”",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    if config.problem != "zdt1":
        raise ValueError("This multi-objective baseline runner currently supports zdt1 only")

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_output_root = output_root / f"{timestamp}_{config.problem}_mo_compare"
    run_output_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    budget = args.budget if args.budget is not None else configured_evaluation_budget(config)
    reference_front = zdt1_reference_front()
    reference_point = _reference_point(config)
    reference_front_source = "analytic_zdt1"
    command = " ".join(sys.argv)
    generated_at = datetime.now(UTC).isoformat()

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    weighted_sum_enabled = not args.skip_weighted_sum

    for seed in seeds:
        nsga_row = _nsga2_row(
            config=config,
            seed=seed,
            budget=budget,
            reference_front=reference_front,
            reference_point=reference_point,
            output_root=run_output_root,
        )
        nsga_row = _decorate_fairness_row(
            nsga_row,
            config=config,
            reference_front_source=reference_front_source,
            reference_point=reference_point,
        )
        raw_rows.append(nsga_row)

        random_result = run_random_pareto_archive(config, seed=seed, budget=budget)
        random_row = _baseline_row(
            random_result,
            config=config,
            reference_front=reference_front,
            reference_point=reference_point,
        )
        random_row = _decorate_fairness_row(
            random_row,
            config=config,
            reference_front_source=reference_front_source,
            reference_point=reference_point,
        )
        coverage_fields = _coverage_fields(
            nsga_row["nondominated_objective_vectors"],
            random_row["nondominated_objective_vectors"],
            nsga_row["objective_directions"],
        )
        random_row.update(coverage_fields)
        nsga_row.update(coverage_fields)
        raw_rows.append(random_row)
        if random_row["error_message"]:
            failures.append(
                {
                    "algorithm": random_row["algorithm"],
                    "seed": random_row["seed"],
                    "error_message": random_row["error_message"],
                }
            )

        if weighted_sum_enabled:
            weighted_result = run_weighted_sum_random_archive(config, seed=seed, budget=budget)
            weighted_row = _baseline_row(
                weighted_result,
                config=config,
                reference_front=reference_front,
                reference_point=reference_point,
            )
            weighted_row = _decorate_fairness_row(
                weighted_row,
                config=config,
                reference_front_source=reference_front_source,
                reference_point=reference_point,
            )
            weighted_row.update(
                _coverage_fields(
                    nsga_row["nondominated_objective_vectors"],
                    weighted_row["nondominated_objective_vectors"],
                    nsga_row["objective_directions"],
                )
            )
            raw_rows.append(weighted_row)
            if weighted_row["error_message"]:
                failures.append(
                    {
                        "algorithm": weighted_row["algorithm"],
                        "seed": weighted_row["seed"],
                        "error_message": weighted_row["error_message"],
                    }
                )

    summary_rows = _aggregate_algorithm_rows(raw_rows)
    paired_rows = _paired_summary(raw_rows)
    fairness_payload = evaluate_parameter_fairness(
        raw_rows,
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
                "reference_front": reference_front_source,
                "hv_reference_point": list(reference_point),
                "metric_limitations": "hypervolume_2d only for 2-objective problems",
            }
        ],
    )
    results_payload = {
        "generated_at": generated_at,
        "command": command,
        "config_path": str(config_path.as_posix()),
        "problem": config.problem,
        "seeds": seeds,
        "budget": budget,
        "reference_point": reference_point,
        "reference_front_point_count": len(reference_front),
        "fairness_contract": {
            "same_evaluation_budget": True,
            "same_seed_list": True,
            "same_problem_instance": True,
            "failure_recorded": True,
            "runtime_recorded": True,
            "front_level_metrics_only": True,
        },
        "raw_rows": raw_rows,
        "summary_rows": summary_rows,
        "paired_rows": paired_rows,
        "failures": failures,
        "fairness": fairness_payload,
    }

    columns = [
        "problem",
        "algorithm",
        "seeds",
        "mean_hv",
        "std_hv",
        "median_hv",
        "mean_distance",
        "mean_spacing",
        "mean_nondominated_count",
        "success_rate",
        "mean_runtime_seconds",
        "mean_evaluations",
        "mean_nsga_over_baseline_coverage",
        "mean_baseline_over_nsga_coverage",
    ]
    _write_json(run_output_root / "mo_baseline_comparison_results.json", results_payload)
    _write_csv(run_output_root / "mo_baseline_comparison_results.csv", summary_rows, columns)
    (run_output_root / "mo_baseline_comparison_results.md").write_text(
        _results_markdown(
            generated_at=generated_at,
            command=command,
            config_path=config_path,
            reference_point=reference_point,
            summary_rows=summary_rows,
            paired_rows=paired_rows,
            failures=failures,
            fairness_payload=fairness_payload,
        ),
        encoding="utf-8",
    )

    regression_results = [
        {
            "command": "python -m pytest tests/test_mo_metrics.py tests/test_mo_baselines.py -q",
            "result": "pending",
            "note": "updated after the dedicated regression run",
        }
    ]
    _write_json(artifact_root / "mo_baseline_comparison_results.json", results_payload)
    _write_csv(artifact_root / "mo_baseline_comparison_results.csv", summary_rows, columns)
    (artifact_root / "mo_baseline_comparison_results.md").write_text(
        _results_markdown(
            generated_at=generated_at,
            command=command,
            config_path=config_path,
            reference_point=reference_point,
            summary_rows=summary_rows,
            paired_rows=paired_rows,
            failures=failures,
            fairness_payload=fairness_payload,
        ),
        encoding="utf-8",
    )
    (artifact_root / "mo_baseline_comparison_report.md").write_text(
        _report_markdown(
            generated_at=generated_at,
            command=command,
            config=config,
            config_path=config_path,
            seeds=seeds,
            budget=budget,
            reference_point=reference_point,
            summary_rows=summary_rows,
            paired_rows=paired_rows,
            failures=failures,
            regression_results=regression_results,
            weighted_sum_enabled=weighted_sum_enabled,
        ),
        encoding="utf-8",
    )
    print(json.dumps(results_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
