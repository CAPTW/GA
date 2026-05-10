# ruff: noqa: E501

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from ga_lab.config import GAConfig
from ga_lab.experiment import budget_baseline_comparison as basecmp
from ga_lab.experiment import hybrid_comparison as hybridcmp
from ga_lab.governance.run_metadata import build_run_metadata, write_run_metadata

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency
    scipy_stats = None


PROJECT_ROOT = basecmp.PROJECT_ROOT

BOOTSTRAP_ITERATIONS = 2000


RUN_COLUMNS = (
    "suite_name",
    "suite_kind",
    "entry_id",
    "problem",
    "tier",
    "size",
    "size_key",
    "label",
    "family",
    "seed",
    "preset_path",
    "configured_budget",
    "configured_evaluation_budget",
    "actual_evaluations_used",
    "actual_evaluations",
    "extra_evaluations_from_hybrid",
    "hybrid_extra_evaluations",
    "hybrid_local_search_applications",
    "hybrid_local_search_improvements",
    "runtime_seconds",
    "early_stop_triggered",
    "success_to_target",
    "evaluations_to_target",
    "generations_to_target",
    "final_best_fitness",
    "final_best_distance",
    "best_feasible_fitness",
    "feasible_rate",
    "mean_violation",
    "hypervolume",
    "pareto_ratio",
    "spread",
    "pareto_front_size",
)


AGGREGATE_COLUMNS = (
    "entry_id",
    "problem",
    "tier",
    "size",
    "label",
    "family",
    "preset_path",
    "configured_budget",
    "seeds",
    "actual_evaluations_used",
    "extra_evaluations_from_hybrid",
    "runtime_seconds",
    "early_stop_triggered",
    "success_rate",
    "success_ci_low",
    "success_ci_high",
    "mean_evaluations_to_target",
    "evaluations_to_target_std",
    "evaluations_to_target_median",
    "evaluations_to_target_ci_low",
    "evaluations_to_target_ci_high",
    "mean_final_best_fitness",
    "final_best_fitness_std",
    "final_best_fitness_median",
    "final_best_fitness_ci_low",
    "final_best_fitness_ci_high",
    "mean_final_best_distance",
    "final_best_distance_std",
    "final_best_distance_median",
    "final_best_distance_ci_low",
    "final_best_distance_ci_high",
    "mean_best_feasible_fitness",
    "best_feasible_fitness_std",
    "best_feasible_fitness_median",
    "best_feasible_fitness_ci_low",
    "best_feasible_fitness_ci_high",
    "mean_feasible_rate",
    "feasible_rate_std",
    "mean_violation",
    "mean_hypervolume",
    "hypervolume_std",
    "hypervolume_median",
    "hypervolume_ci_low",
    "hypervolume_ci_high",
    "mean_pareto_ratio",
    "pareto_ratio_std",
    "pareto_ratio_median",
    "pareto_ratio_ci_low",
    "pareto_ratio_ci_high",
    "mean_spread",
    "spread_std",
    "spread_median",
    "spread_ci_low",
    "spread_ci_high",
    "mean_pareto_front_size",
    "pareto_front_size_std",
)


COMPARISON_COLUMNS = (
    "entry_id",
    "comparison_id",
    "problem",
    "tier",
    "size",
    "metric",
    "objective",
    "left",
    "right",
    "seeds",
    "configured_budget",
    "left_mean",
    "left_std",
    "left_median",
    "left_ci_low",
    "left_ci_high",
    "right_mean",
    "right_std",
    "right_median",
    "right_ci_low",
    "right_ci_high",
    "oriented_mean_diff",
    "oriented_diff_ci_low",
    "oriented_diff_ci_high",
    "relative_advantage",
    "paired_effect_size",
    "paired_significance_test",
    "p_value",
    "runtime_seconds_left",
    "runtime_seconds_right",
    "actual_evaluations_used_left",
    "actual_evaluations_used_right",
    "extra_evaluations_from_hybrid_left",
    "extra_evaluations_from_hybrid_right",
    "early_stop_triggered_left",
    "early_stop_triggered_right",
    "practical_significance_note",
)


INVENTORY_COLUMNS = (
    "item",
    "current_status",
    "evidence_source",
    "current_weakness",
    "ablation_needed",
    "statistical_hardening_needed",
)


CLASSIFICATION_COLUMNS = (
    "item",
    "status",
    "scope",
    "reason",
    "evidence",
)


@dataclass(slots=True)
class AblationMethod:
    label: str
    kind: str
    family: str | None = None
    overrides: dict[str, Any] | None = None
    preset_path: Path | None = None


@dataclass(slots=True)
class MetricComparison:
    comparison_id: str
    left: str
    right: str
    metric: str
    objective: str
    note: str | None = None


@dataclass(slots=True)
class AblationEntry:
    suite_name: str
    suite_kind: str
    entry_id: str
    problem: str
    size: int
    preset_path: Path
    methods: tuple[AblationMethod, ...]
    comparisons: tuple[MetricComparison, ...]
    seeds: int
    seed_start: int


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = json.loads(json.dumps(value))
    return merged


def _resolve_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    candidates = [
        (PROJECT_ROOT / path).resolve(),
        (manifest_path.parent / path).resolve(),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _inventory_rows() -> list[dict[str, str]]:
    return [
        {
            "item": "onemax practical default = hill_climb",
            "current_status": "official",
            "evidence_source": "README, docs/benchmark_vs_baselines.md, outputs/benchmark_summary/baseline_comparison_summary.json",
            "current_weakness": "paired statistics were still light",
            "ablation_needed": "no",
            "statistical_hardening_needed": "yes",
        },
        {
            "item": "onemax GA-family variants as practical solvers",
            "current_status": "note only",
            "evidence_source": "README, docs/solver_choice_guide.md",
            "current_weakness": "mutation-only or seeded variants were not paired against hill climb",
            "ablation_needed": "yes",
            "statistical_hardening_needed": "yes",
        },
        {
            "item": "knapsack greedy_local_search practical default",
            "current_status": "official",
            "evidence_source": "README, docs/benchmark_vs_baselines.md, outputs/benchmark_summary/hybrid_comparison_summary.json",
            "current_weakness": "hybrid component value was not isolated",
            "ablation_needed": "yes",
            "statistical_hardening_needed": "yes",
        },
        {
            "item": "knapsack seeded-repair hybrid recipe",
            "current_status": "experimental",
            "evidence_source": "docs/hybrid_vs_baselines.md, outputs/benchmark_summary/hybrid_comparison_summary.json",
            "current_weakness": "full hybrid recipe was not decomposed into seed / repair / local-search effects",
            "ablation_needed": "yes",
            "statistical_hardening_needed": "yes",
        },
        {
            "item": "tsp_medium_hybrid promoted preset",
            "current_status": "official",
            "evidence_source": "README, docs/hybrid_vs_baselines.md, configs/presets/tsp_medium_hybrid.json",
            "current_weakness": "official promotion depended on mean comparison without component ablation",
            "ablation_needed": "yes",
            "statistical_hardening_needed": "yes",
        },
        {
            "item": "tsp large hybrid gap-closing path",
            "current_status": "experimental",
            "evidence_source": "docs/hybrid_vs_baselines.md, outputs/benchmark_summary/hybrid_comparison_summary.json",
            "current_weakness": "baseline gap remained and component-level evidence was missing",
            "ablation_needed": "yes",
            "statistical_hardening_needed": "yes",
        },
        {
            "item": "zdt1 NSGA-II default path",
            "current_status": "official",
            "evidence_source": "README, docs/benchmark_vs_baselines.md, docs/solver_choice_guide.md",
            "current_weakness": "large-size metric tradeoff needed stronger paired interpretation",
            "ablation_needed": "limited",
            "statistical_hardening_needed": "yes",
        },
        {
            "item": "zdt1 mutation archive as a cheap alternative",
            "current_status": "note only",
            "evidence_source": "README, docs/solver_choice_guide.md, outputs/benchmark_summary/baseline_comparison_summary.json",
            "current_weakness": "HV-first vs diversity tradeoff was not quantified with paired confidence intervals",
            "ablation_needed": "limited",
            "statistical_hardening_needed": "yes",
        },
    ]


def _load_methods(manifest_path: Path, raw_entry: dict[str, Any]) -> tuple[AblationMethod, ...]:
    raw_methods = raw_entry.get("methods")
    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError("Ablation entry requires a non-empty methods list")
    methods: list[AblationMethod] = []
    seen: set[str] = set()
    for raw_method in raw_methods:
        if not isinstance(raw_method, dict):
            raise ValueError("Ablation methods must be JSON objects")
        label = str(raw_method["label"])
        if label in seen:
            raise ValueError(f"Duplicate ablation method label: {label}")
        seen.add(label)
        kind = str(raw_method["kind"])
        family = str(raw_method["family"]) if "family" in raw_method else None
        overrides = dict(raw_method.get("overrides", {}))
        preset_path = None
        if "preset" in raw_method:
            preset_path = _resolve_path(manifest_path, str(raw_method["preset"]))
        methods.append(
            AblationMethod(
                label=label,
                kind=kind,
                family=family,
                overrides=overrides,
                preset_path=preset_path,
            )
        )
    return tuple(methods)


def _load_comparisons(raw_entry: dict[str, Any], labels: set[str]) -> tuple[MetricComparison, ...]:
    raw_comparisons = raw_entry.get("comparisons")
    if not isinstance(raw_comparisons, list) or not raw_comparisons:
        raise ValueError("Ablation entry requires a non-empty comparisons list")
    comparisons: list[MetricComparison] = []
    seen: set[str] = set()
    for raw_comparison in raw_comparisons:
        if not isinstance(raw_comparison, dict):
            raise ValueError("Ablation comparisons must be JSON objects")
        comparison_id = str(raw_comparison["comparison_id"])
        if comparison_id in seen:
            raise ValueError(f"Duplicate ablation comparison id: {comparison_id}")
        seen.add(comparison_id)
        left = str(raw_comparison["left"])
        right = str(raw_comparison["right"])
        if left not in labels or right not in labels:
            raise ValueError(
                f"Ablation comparison references unknown labels: {left}, {right}"
            )
        objective = str(raw_comparison["objective"])
        if objective not in {"max", "min"}:
            raise ValueError("Comparison objective must be 'max' or 'min'")
        comparisons.append(
            MetricComparison(
                comparison_id=comparison_id,
                left=left,
                right=right,
                metric=str(raw_comparison["metric"]),
                objective=objective,
                note=str(raw_comparison["note"]) if "note" in raw_comparison else None,
            )
        )
    return tuple(comparisons)


def load_ablation_manifest(path: str | Path) -> tuple[dict[str, Any], list[AblationEntry]]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Ablation manifest must be a JSON object")
    suite_name = str(manifest.get("suite_name", manifest_path.stem))
    suite_kind = str(manifest.get("suite_kind", "comparison"))
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Ablation manifest requires a non-empty entries list")

    entries: list[AblationEntry] = []
    seen_entry_ids: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Each ablation manifest entry must be an object")
        entry_id = str(raw_entry["entry_id"])
        if entry_id in seen_entry_ids:
            raise ValueError(f"Duplicate ablation entry id: {entry_id}")
        seen_entry_ids.add(entry_id)
        problem = str(raw_entry["problem"])
        size = int(raw_entry["size"])
        basecmp._validated_entry(problem, size, ())
        preset_path = _resolve_path(manifest_path, str(raw_entry["preset"]))
        methods = _load_methods(manifest_path, raw_entry)
        labels = {method.label for method in methods}
        comparisons = _load_comparisons(raw_entry, labels)
        entries.append(
            AblationEntry(
                suite_name=suite_name,
                suite_kind=suite_kind,
                entry_id=entry_id,
                problem=problem,
                size=size,
                preset_path=preset_path,
                methods=methods,
                comparisons=comparisons,
                seeds=int(raw_entry.get("seeds", manifest.get("default_seeds", 5))),
                seed_start=int(raw_entry.get("seed_start", 0)),
            )
        )
    return manifest, entries


def _preset_reference(base_preset_path: Path, method: AblationMethod) -> str:
    if method.preset_path is not None:
        return str(method.preset_path.as_posix())
    if method.kind == "base_preset":
        return str(base_preset_path.as_posix())
    return f"{base_preset_path.as_posix()}#ablation:{method.label}"


def _to_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        scalar = _to_scalar(row.get(key))
        if scalar is not None:
            values.append(scalar)
    return values


def _mean_series(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _series(rows, key)
    if not values:
        return None
    return mean(values)


def _bootstrap_ci(values: list[float], seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    sample_count = len(values)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        resample = [values[rng.randrange(sample_count)] for _ in range(sample_count)]
        estimates.append(mean(resample))
    estimates.sort()
    lower_idx = int(0.025 * (BOOTSTRAP_ITERATIONS - 1))
    upper_idx = int(0.975 * (BOOTSTRAP_ITERATIONS - 1))
    return estimates[lower_idx], estimates[upper_idx]


def _summary_stats(values: list[float], seed: int) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "ci_low": None,
            "ci_high": None,
        }
    ci_low, ci_high = _bootstrap_ci(values, seed)
    return {
        "mean": mean(values),
        "std": 0.0 if len(values) == 1 else stdev(values),
        "median": median(values),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _rank_biserial(differences: list[float]) -> float:
    non_zero = [value for value in differences if value != 0.0]
    if not non_zero:
        return 0.0

    indexed = sorted((abs(value), idx) for idx, value in enumerate(non_zero))
    ranks = [0.0] * len(non_zero)
    position = 1
    cursor = 0
    while cursor < len(indexed):
        tie_end = cursor
        while tie_end < len(indexed) and indexed[tie_end][0] == indexed[cursor][0]:
            tie_end += 1
        average_rank = (position + (position + (tie_end - cursor) - 1)) / 2.0
        for _, original_idx in indexed[cursor:tie_end]:
            ranks[original_idx] = average_rank
        position += tie_end - cursor
        cursor = tie_end

    positive = sum(rank for rank, value in zip(ranks, non_zero, strict=True) if value > 0.0)
    negative = sum(rank for rank, value in zip(ranks, non_zero, strict=True) if value < 0.0)
    total = positive + negative
    if total == 0.0:
        return 0.0
    return (positive - negative) / total


def _sign_test_pvalue(differences: list[float]) -> float:
    non_zero = [value for value in differences if value != 0.0]
    trials = len(non_zero)
    if trials == 0:
        return 1.0
    positives = sum(1 for value in non_zero if value > 0.0)
    negatives = trials - positives
    threshold = min(positives, negatives)
    numerator = sum(math.comb(trials, idx) for idx in range(threshold + 1))
    return min(1.0, 2.0 * numerator / (2**trials))


def _paired_test(differences: list[float]) -> tuple[str, float]:
    non_zero = [value for value in differences if value != 0.0]
    if not non_zero:
        return "all_zero", 1.0
    if scipy_stats is not None:
        try:
            result = scipy_stats.wilcoxon(non_zero, zero_method="wilcox", alternative="two-sided")
            return "wilcoxon_signed_rank", float(result.pvalue)
        except ValueError:
            pass
    return "sign_test", _sign_test_pvalue(non_zero)


def _relative_advantage(left_mean: float | None, right_mean: float | None, objective: str) -> float | None:
    if left_mean is None or right_mean is None:
        return None
    baseline = abs(right_mean)
    if baseline == 0.0:
        return None
    if objective == "max":
        return (left_mean - right_mean) / baseline
    return (right_mean - left_mean) / baseline


def _practical_note(
    *,
    p_value: float,
    ci_low: float | None,
    ci_high: float | None,
    relative_advantage: float | None,
    runtime_left: float | None,
    runtime_right: float | None,
    extra_left: float | None,
    extra_right: float | None,
) -> str:
    if ci_low is None or ci_high is None:
        return "paired data가 부족해서 정식 판정 대신 참고용 trend만 남겼다."

    statistically_clear = p_value < 0.05 and not (ci_low <= 0.0 <= ci_high)
    if statistically_clear:
        if relative_advantage is not None and abs(relative_advantage) < 0.01:
            headline = "통계적으로는 차이가 보이지만 실질 효과는 작다."
        elif ci_low > 0.0:
            headline = "left method가 실질적으로도 유리한 방향이다."
        else:
            headline = "left method가 불리한 방향으로 차이가 난다."
    else:
        headline = "통계적으로 분명한 우위라기보다 trend 수준이다."

    runtime_note = ""
    if runtime_left is not None and runtime_right not in {None, 0.0}:
        runtime_ratio = runtime_left / runtime_right
        if runtime_ratio >= 1.2:
            runtime_note = f" runtime은 left가 약 {runtime_ratio:.2f}배 더 크다."
        elif runtime_ratio <= 0.8:
            runtime_note = f" runtime은 left가 약 {1 / max(runtime_ratio, 1e-9):.2f}배 더 작다."

    extra_note = ""
    left_extra = extra_left or 0.0
    right_extra = extra_right or 0.0
    if left_extra > right_extra:
        extra_note = f" 하이브리드 추가 평가는 left가 평균 {left_extra:.1f}회 더 쓴다."
    elif right_extra > left_extra:
        extra_note = f" 하이브리드 추가 평가는 right가 평균 {right_extra:.1f}회 더 쓴다."

    return headline + runtime_note + extra_note


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["entry_id"],
            row["problem"],
            row["tier"],
            row["size"],
            row["label"],
            row["family"],
            row["preset_path"],
            row["configured_budget"],
        )
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(grouped)):
        bucket = grouped[key]
        sample = bucket[0]
        summary_seed = 1000 + index * 10
        success_values = _series(bucket, "success_to_target")
        evals_to_target = _series(bucket, "evaluations_to_target")
        final_best_fitness = _series(bucket, "final_best_fitness")
        final_best_distance = _series(bucket, "final_best_distance")
        best_feasible_fitness = _series(bucket, "best_feasible_fitness")
        feasible_rate = _series(bucket, "feasible_rate")
        mean_violation = _series(bucket, "mean_violation")
        hypervolume = _series(bucket, "hypervolume")
        pareto_ratio = _series(bucket, "pareto_ratio")
        spread = _series(bucket, "spread")
        pareto_front_size = _series(bucket, "pareto_front_size")
        actual_evaluations = _series(bucket, "actual_evaluations_used")
        extra_evaluations = _series(bucket, "extra_evaluations_from_hybrid")
        runtime = _series(bucket, "runtime_seconds")
        early_stop = _series(bucket, "early_stop_triggered")

        eval_stats = _summary_stats(evals_to_target, summary_seed + 1)
        fit_stats = _summary_stats(final_best_fitness, summary_seed + 2)
        distance_stats = _summary_stats(final_best_distance, summary_seed + 3)
        feasible_stats = _summary_stats(best_feasible_fitness, summary_seed + 4)
        hv_stats = _summary_stats(hypervolume, summary_seed + 5)
        pareto_stats = _summary_stats(pareto_ratio, summary_seed + 6)
        spread_stats = _summary_stats(spread, summary_seed + 7)
        success_ci_low, success_ci_high = _bootstrap_ci(success_values, summary_seed + 8)

        aggregates.append(
            {
                "entry_id": sample["entry_id"],
                "problem": sample["problem"],
                "tier": sample["tier"],
                "size": sample["size"],
                "label": sample["label"],
                "family": sample["family"],
                "preset_path": sample["preset_path"],
                "configured_budget": sample["configured_budget"],
                "seeds": len(bucket),
                "actual_evaluations_used": mean(actual_evaluations) if actual_evaluations else None,
                "extra_evaluations_from_hybrid": mean(extra_evaluations) if extra_evaluations else 0.0,
                "runtime_seconds": mean(runtime) if runtime else None,
                "early_stop_triggered": mean(early_stop) if early_stop else 0.0,
                "success_rate": mean(success_values) if success_values else None,
                "success_ci_low": success_ci_low,
                "success_ci_high": success_ci_high,
                "mean_evaluations_to_target": eval_stats["mean"],
                "evaluations_to_target_std": eval_stats["std"],
                "evaluations_to_target_median": eval_stats["median"],
                "evaluations_to_target_ci_low": eval_stats["ci_low"],
                "evaluations_to_target_ci_high": eval_stats["ci_high"],
                "mean_final_best_fitness": fit_stats["mean"],
                "final_best_fitness_std": fit_stats["std"],
                "final_best_fitness_median": fit_stats["median"],
                "final_best_fitness_ci_low": fit_stats["ci_low"],
                "final_best_fitness_ci_high": fit_stats["ci_high"],
                "mean_final_best_distance": distance_stats["mean"],
                "final_best_distance_std": distance_stats["std"],
                "final_best_distance_median": distance_stats["median"],
                "final_best_distance_ci_low": distance_stats["ci_low"],
                "final_best_distance_ci_high": distance_stats["ci_high"],
                "mean_best_feasible_fitness": feasible_stats["mean"],
                "best_feasible_fitness_std": feasible_stats["std"],
                "best_feasible_fitness_median": feasible_stats["median"],
                "best_feasible_fitness_ci_low": feasible_stats["ci_low"],
                "best_feasible_fitness_ci_high": feasible_stats["ci_high"],
                "mean_feasible_rate": mean(feasible_rate) if feasible_rate else None,
                "feasible_rate_std": (
                    0.0 if len(feasible_rate) == 1 else stdev(feasible_rate) if feasible_rate else None
                ),
                "mean_violation": mean(mean_violation) if mean_violation else None,
                "mean_hypervolume": hv_stats["mean"],
                "hypervolume_std": hv_stats["std"],
                "hypervolume_median": hv_stats["median"],
                "hypervolume_ci_low": hv_stats["ci_low"],
                "hypervolume_ci_high": hv_stats["ci_high"],
                "mean_pareto_ratio": pareto_stats["mean"],
                "pareto_ratio_std": pareto_stats["std"],
                "pareto_ratio_median": pareto_stats["median"],
                "pareto_ratio_ci_low": pareto_stats["ci_low"],
                "pareto_ratio_ci_high": pareto_stats["ci_high"],
                "mean_spread": spread_stats["mean"],
                "spread_std": spread_stats["std"],
                "spread_median": spread_stats["median"],
                "spread_ci_low": spread_stats["ci_low"],
                "spread_ci_high": spread_stats["ci_high"],
                "mean_pareto_front_size": mean(pareto_front_size) if pareto_front_size else None,
                "pareto_front_size_std": (
                    0.0
                    if len(pareto_front_size) == 1
                    else stdev(pareto_front_size)
                    if pareto_front_size
                    else None
                ),
            }
        )
    return aggregates


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    return _to_scalar(row.get(metric))


def _comparison_rows(
    entries: dict[str, AblationEntry],
    run_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in run_rows:
        index[(row["entry_id"], row["label"], int(row["seed"]))] = row

    comparisons: list[dict[str, Any]] = []
    for entry_id, entry in sorted(entries.items()):
        for comparison in entry.comparisons:
            pairs: list[tuple[int, dict[str, Any], dict[str, Any], float, float]] = []
            for row in run_rows:
                if row["entry_id"] != entry_id or row["label"] != comparison.left:
                    continue
                right_row = index.get((entry_id, comparison.right, int(row["seed"])))
                if right_row is None:
                    continue
                left_value = _metric_value(row, comparison.metric)
                right_value = _metric_value(right_row, comparison.metric)
                if left_value is None or right_value is None:
                    continue
                pairs.append((int(row["seed"]), row, right_row, left_value, right_value))

            left_values = [left_value for _, _, _, left_value, _ in pairs]
            right_values = [right_value for _, _, _, _, right_value in pairs]
            if comparison.objective == "max":
                oriented_differences = [
                    left - right for left, right in zip(left_values, right_values, strict=True)
                ]
            else:
                oriented_differences = [
                    right - left for left, right in zip(left_values, right_values, strict=True)
                ]

            left_stats = _summary_stats(left_values, 2000 + len(comparisons) * 10 + 1)
            right_stats = _summary_stats(right_values, 2000 + len(comparisons) * 10 + 2)
            diff_ci_low, diff_ci_high = _bootstrap_ci(
                oriented_differences,
                2000 + len(comparisons) * 10 + 3,
            )
            test_name, p_value = _paired_test(oriented_differences)
            effect_size = _rank_biserial(oriented_differences)
            left_rows = [left_row for _, left_row, _, _, _ in pairs]
            right_rows = [right_row for _, _, right_row, _, _ in pairs]
            runtime_left = _mean_series(left_rows, "runtime_seconds")
            runtime_right = _mean_series(right_rows, "runtime_seconds")
            eval_left = _mean_series(left_rows, "actual_evaluations_used")
            eval_right = _mean_series(right_rows, "actual_evaluations_used")
            extra_left = _mean_series(left_rows, "extra_evaluations_from_hybrid") or 0.0
            extra_right = _mean_series(right_rows, "extra_evaluations_from_hybrid") or 0.0
            early_left = _mean_series(left_rows, "early_stop_triggered") or 0.0
            early_right = _mean_series(right_rows, "early_stop_triggered") or 0.0
            relative = _relative_advantage(left_stats["mean"], right_stats["mean"], comparison.objective)
            comparisons.append(
                {
                    "entry_id": entry_id,
                    "comparison_id": comparison.comparison_id,
                    "problem": entry.problem,
                    "tier": basecmp._tier_name(entry.problem, entry.size),
                    "size": entry.size,
                    "metric": comparison.metric,
                    "objective": comparison.objective,
                    "left": comparison.left,
                    "right": comparison.right,
                    "seeds": len(pairs),
                    "configured_budget": pairs[0][1]["configured_budget"] if pairs else None,
                    "left_mean": left_stats["mean"],
                    "left_std": left_stats["std"],
                    "left_median": left_stats["median"],
                    "left_ci_low": left_stats["ci_low"],
                    "left_ci_high": left_stats["ci_high"],
                    "right_mean": right_stats["mean"],
                    "right_std": right_stats["std"],
                    "right_median": right_stats["median"],
                    "right_ci_low": right_stats["ci_low"],
                    "right_ci_high": right_stats["ci_high"],
                    "oriented_mean_diff": mean(oriented_differences) if oriented_differences else None,
                    "oriented_diff_ci_low": diff_ci_low,
                    "oriented_diff_ci_high": diff_ci_high,
                    "relative_advantage": relative,
                    "paired_effect_size": effect_size,
                    "paired_significance_test": test_name,
                    "p_value": p_value,
                    "runtime_seconds_left": runtime_left,
                    "runtime_seconds_right": runtime_right,
                    "actual_evaluations_used_left": eval_left,
                    "actual_evaluations_used_right": eval_right,
                    "extra_evaluations_from_hybrid_left": extra_left,
                    "extra_evaluations_from_hybrid_right": extra_right,
                    "early_stop_triggered_left": early_left,
                    "early_stop_triggered_right": early_right,
                    "practical_significance_note": _practical_note(
                        p_value=p_value,
                        ci_low=diff_ci_low,
                        ci_high=diff_ci_high,
                        relative_advantage=relative,
                        runtime_left=runtime_left,
                        runtime_right=runtime_right,
                        extra_left=extra_left,
                        extra_right=extra_right,
                    ),
                }
            )
    return comparisons


def _comparison_lookup(rows: list[dict[str, Any]], comparison_id: str) -> dict[str, Any] | None:
    for row in rows:
        if row["comparison_id"] == comparison_id:
            return row
    return None


def _format_relative(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _classification_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    tsp_medium_vs_baseline = _comparison_lookup(
        comparison_rows,
        "tsp20_official_vs_baseline_distance",
    )
    tsp_medium_vs_ga = _comparison_lookup(
        comparison_rows,
        "tsp20_official_vs_pure_ga_distance",
    )
    tsp_medium_vs_seeded = _comparison_lookup(
        comparison_rows,
        "tsp20_official_vs_seeded_only_distance",
    )
    if tsp_medium_vs_baseline and tsp_medium_vs_ga:
        if (
            (tsp_medium_vs_baseline["oriented_diff_ci_low"] or -1.0) > 0.0
            and (tsp_medium_vs_ga["oriented_diff_ci_low"] or -1.0) > 0.0
            and tsp_medium_vs_baseline["p_value"] < 0.05
            and tsp_medium_vs_ga["p_value"] < 0.05
        ):
            tsp_medium_status = "Official"
            seeded_note = (
                " seeded-only와의 차이도 통계적으로 분명하지 않았다."
                if tsp_medium_vs_seeded and tsp_medium_vs_seeded["p_value"] >= 0.05
                else ""
            )
            tsp_medium_reason = (
                "validated medium(20) quality-first path로는 유지 가능하다. 다만 baseline 대비 "
                "이득은 작고, ablation상 nearest-neighbor seeding이 핵심이며 2-opt 추가 단계는 "
                "seeded-only보다 분명히 낫다고 말할 정도는 아니었다." + seeded_note
            )
        else:
            tsp_medium_status = "Experimental"
            tsp_medium_reason = (
                "validated medium(20)에서 hybrid 이득 신호는 있지만 strongest cheap baseline 대비 "
                "공식 승격을 유지할 만큼 충분히 단단하지 않았다."
            )
    else:
        tsp_medium_status = "Experimental"
        tsp_medium_reason = "medium TSP hybrid 비교 데이터가 부족해서 공식 승격을 유지하기 어렵다."
    rows.append(
        {
            "item": "tsp_medium_hybrid",
            "status": tsp_medium_status,
            "scope": "tsp validated medium (20 cities)",
            "reason": tsp_medium_reason,
            "evidence": "tsp20_official_vs_pure_ga_distance / tsp20_official_vs_baseline_distance",
        }
    )

    knapsack_seed_repair = _comparison_lookup(comparison_rows, "knapsack20_seed_repair_vs_greedy")
    knapsack_seed_repair_large = _comparison_lookup(comparison_rows, "knapsack80_seed_repair_vs_greedy")
    if knapsack_seed_repair and knapsack_seed_repair_large:
        if (
            knapsack_seed_repair["oriented_diff_ci_low"] is not None
            and knapsack_seed_repair_large["oriented_diff_ci_low"] is not None
            and knapsack_seed_repair["oriented_diff_ci_low"] > 0.0
            and knapsack_seed_repair_large["oriented_diff_ci_low"] > 0.0
        ):
            knapsack_status = "Official"
            knapsack_reason = "seed+repair hybrid가 cheap greedy baseline보다도 일관적으로 우세했다."
        else:
            knapsack_status = "Experimental"
            knapsack_reason = (
                "seed+repair hybrid는 pure GA보다 낫지만 greedy baseline을 안정적으로 넘지는 못했다."
            )
    else:
        knapsack_status = "Experimental"
        knapsack_reason = "knapsack hybrid evidence가 부분적이라 실험적 경로로만 남긴다."
    rows.append(
        {
            "item": "knapsack_seeded_repair_hybrid",
            "status": knapsack_status,
            "scope": "knapsack validated sizes",
            "reason": knapsack_reason,
            "evidence": "knapsack20_seed_repair_vs_greedy / knapsack80_seed_repair_vs_greedy",
        }
    )

    knapsack_full_vs_seed = _comparison_lookup(comparison_rows, "knapsack20_full_hybrid_vs_seed_repair")
    if knapsack_full_vs_seed and (
        knapsack_full_vs_seed["oriented_diff_ci_high"] is None
        or knapsack_full_vs_seed["oriented_diff_ci_high"] <= 0.0
        or (
            knapsack_full_vs_seed["relative_advantage"] is not None
            and knapsack_full_vs_seed["relative_advantage"] < 0.01
        )
    ):
        full_knapsack_status = "Not recommended"
        full_knapsack_reason = "local improvement가 추가 복잡도와 평가량 대비 실익을 거의 만들지 못했다."
    else:
        full_knapsack_status = "Experimental"
        full_knapsack_reason = "local improvement의 추가 효과가 약하지만 완전히 배제할 만큼은 아니다."
    rows.append(
        {
            "item": "knapsack_full_memetic_hybrid",
            "status": full_knapsack_status,
            "scope": "knapsack validated sizes",
            "reason": full_knapsack_reason,
            "evidence": "knapsack20_full_hybrid_vs_seed_repair",
        }
    )

    onemax_hill_vs_ga = _comparison_lookup(comparison_rows, "onemax128_hill_vs_pure_ga_evals")
    rows.append(
        {
            "item": "onemax_hill_climb_default",
            "status": "Official",
            "scope": "onemax validated sizes",
            "reason": (
                "hill climb가 validated onemax practical default라는 기존 해석을 유지한다. "
                f"large paired run에서도 mean evaluation advantage는 {_format_relative(onemax_hill_vs_ga.get('relative_advantage') if onemax_hill_vs_ga else None)}."
            ),
            "evidence": "baseline_comparison_summary + onemax128_hill_vs_pure_ga_evals",
        }
    )
    rows.append(
        {
            "item": "onemax_ga_family_as_practical_solver",
            "status": "Not recommended",
            "scope": "onemax validated sizes",
            "reason": (
                "mutation-only EA를 포함한 GA-family 변형도 practical default를 hill climb에서 "
                "되돌릴 근거를 만들지 못했다."
            ),
            "evidence": "onemax128_mutation_only_vs_pure_ga_evals / onemax128_hill_vs_pure_ga_evals",
        }
    )

    rows.append(
        {
            "item": "knapsack_greedy_local_search_default",
            "status": "Official",
            "scope": "knapsack validated sizes",
            "reason": "hybrid gap-closing evidence가 있어도 cheap greedy baseline practical default는 유지된다.",
            "evidence": "baseline_comparison_summary + knapsack ablation comparisons",
        }
    )

    tsp_large_vs_baseline = _comparison_lookup(comparison_rows, "tsp50_full_hybrid_vs_baseline_distance")
    rows.append(
        {
            "item": "tsp_large_hybrid_path",
            "status": (
                "Experimental"
                if tsp_large_vs_baseline
                and tsp_large_vs_baseline["oriented_mean_diff"] is not None
                and tsp_large_vs_baseline["oriented_mean_diff"] > 0.0
                else "Not recommended"
            ),
            "scope": "tsp validated large (50 cities)",
            "reason": (
                "large hybrid는 pure GA gap-closing에는 의미가 있지만 strongest cheap baseline을 "
                "넘지 못하면 default로 올리기 어렵다."
            ),
            "evidence": "tsp50_full_hybrid_vs_baseline_distance / tsp50_full_hybrid_vs_pure_ga_distance",
        }
    )

    rows.append(
        {
            "item": "zdt1_nsga2_default",
            "status": "Official",
            "scope": "zdt1 validated sizes",
            "reason": (
                "NSGA-II preset은 validated zdt1 범위에서 기본 multi-metric 경로로 남긴다. "
                "large는 scalar winner가 아니라 metric-priority tradeoff로 문서화한다."
            ),
            "evidence": "zdt1 HV / pareto_ratio / spread paired comparisons",
        }
    )
    rows.append(
        {
            "item": "zdt1_mutation_archive_hv_only",
            "status": "Experimental",
            "scope": "zdt1 validated large (50)",
            "reason": "large에서 HV-first cheap alternative로는 의미가 있지만 multi-metric default는 아니다.",
            "evidence": "zdt150_nsga2_vs_mutation_hv / zdt150_nsga2_vs_mutation_spread",
        }
    )
    return rows


def _run_method_trial(entry: AblationEntry, base_config: GAConfig, method: AblationMethod, seed: int) -> dict[str, Any]:
    if method.kind == "baseline":
        baseline_data = base_config.to_dict()
        baseline_data["seed"] = seed
        baseline_config = GAConfig.from_dict(baseline_data)
        baseline_row = basecmp._run_baseline_trial(entry, baseline_config, str(method.family), seed)
        baseline_row.update(
            {
                "entry_id": entry.entry_id,
                "configured_budget": baseline_row["configured_evaluation_budget"],
                "actual_evaluations_used": baseline_row["actual_evaluations"],
                "extra_evaluations_from_hybrid": None,
                "hybrid_extra_evaluations": None,
                "hybrid_local_search_applications": None,
                "hybrid_local_search_improvements": None,
                "early_stop_triggered": baseline_row["actual_evaluations"] < baseline_row["configured_evaluation_budget"],
            }
        )
        return baseline_row

    if method.kind == "base_preset":
        method_data = base_config.to_dict()
    elif method.kind == "preset":
        if method.preset_path is None:
            raise ValueError(f"Preset path is required for ablation method {method.label}")
        method_data = json.loads(method.preset_path.read_text(encoding="utf-8"))
    elif method.kind == "override":
        method_data = _deep_merge(base_config.to_dict(), method.overrides or {})
    else:
        raise ValueError(f"Unsupported ablation method kind: {method.kind}")

    method_data["seed"] = seed
    config = GAConfig.from_dict(method_data)
    row = hybridcmp._run_optimizer_trial(
        entry,
        config,
        label=method.label,
        family=method.family or config.algorithm,
        preset_reference=_preset_reference(entry.preset_path, method),
        seed=seed,
    )
    row.update(
        {
            "entry_id": entry.entry_id,
            "configured_budget": row["configured_evaluation_budget"],
            "actual_evaluations_used": row["actual_evaluations"],
            "extra_evaluations_from_hybrid": row.get("hybrid_extra_evaluations"),
            "early_stop_triggered": row["actual_evaluations"] < row["configured_evaluation_budget"],
        }
    )
    return row


def _run_manifest(
    manifest_path: str | Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, dict[str, AblationEntry]]:
    manifest, entries = load_ablation_manifest(manifest_path)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suite_dir = output_root / f"{timestamp}_{manifest['suite_name']}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    entry_registry: dict[str, AblationEntry] = {}
    for entry in entries:
        entry_registry[entry.entry_id] = entry
        base_config = GAConfig.from_dict(json.loads(entry.preset_path.read_text(encoding="utf-8")))
        if base_config.problem != entry.problem:
            raise ValueError(
                f"Preset problem mismatch for {entry.preset_path}: {base_config.problem} != {entry.problem}"
            )
        if basecmp._size_value(base_config) != entry.size:
            raise ValueError(
                f"Preset size mismatch for {entry.preset_path}: {basecmp._size_value(base_config)} != {entry.size}"
            )

        for offset in range(entry.seeds):
            seed = entry.seed_start + offset
            for method in entry.methods:
                run_rows.append(_run_method_trial(entry, base_config, method, seed))

    aggregate_rows = _aggregate_rows(run_rows)
    comparison_rows = _comparison_rows(entry_registry, run_rows)
    (suite_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (suite_dir / "run_rows.json").write_text(
        json.dumps(run_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (suite_dir / "aggregate_rows.json").write_text(
        json.dumps(aggregate_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (suite_dir / "comparison_rows.json").write_text(
        json.dumps(comparison_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest, run_rows, suite_dir, entry_registry


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Ablation summary",
        "",
        "## Scope",
        "",
        "공식 비교 범위는 validated size range 안으로만 제한했다.",
        "",
        "| Problem | Size key | Validated sizes |",
        "| --- | --- | --- |",
    ]
    for problem, scope in summary["validated_scope"].items():
        lines.append(
            f"| {problem} | `{scope['size_key']}` | `{' / '.join(str(size) for size in scope['sizes'])}` |"
        )

    lines.extend(
        [
            "",
            "## Current official claims inventory",
            "",
            "| Item | Current status | Evidence source | Current weakness | Ablation needed? | Statistical hardening needed? |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["official_claim_inventory"]:
        lines.append(
            f"| {row['item']} | `{row['current_status']}` | {row['evidence_source']} | "
            f"{row['current_weakness']} | `{row['ablation_needed']}` | `{row['statistical_hardening_needed']}` |"
        )

    lines.extend(
        [
            "",
            "## Statistical hardening method",
            "",
            "- Paired seed design: 같은 `entry_id` 안에서 method 간에 seed를 공유했다.",
            "- Primary fairness basis: matched function evaluation budget.",
            "- Reported per comparison: `n`, mean, std, median, 95% bootstrap CI, paired rank-biserial effect size, paired significance test, practical note.",
            "- Test policy: Wilcoxon signed-rank when available, otherwise sign test fallback.",
            "- zdt1는 metric별로 따로 해석했다. single scalar winner로 합치지 않았다.",
            "",
            "## Method summary",
            "",
            "| Entry | Label | Seeds | Budget | Actual evals | Extra hybrid evals | Runtime | Key metric |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in summary["aggregate_rows"]:
        key_metric = "-"
        if row["success_rate"] is not None:
            key_metric = f"success `{row['success_rate']:.2f}`"
        elif row["mean_best_feasible_fitness"] is not None:
            key_metric = f"feasible `{row['mean_best_feasible_fitness']:.2f}`"
        elif row["mean_final_best_distance"] is not None:
            key_metric = f"distance `{row['mean_final_best_distance']:.2f}`"
        elif row["mean_hypervolume"] is not None:
            key_metric = f"HV `{row['mean_hypervolume']:.4f}`"
        lines.append(
            f"| `{row['entry_id']}` | `{row['label']}` | `{row['seeds']}` | `{row['configured_budget']}` | "
            f"`{(row['actual_evaluations_used'] or 0.0):.1f}` | `{(row['extra_evaluations_from_hybrid'] or 0.0):.1f}` | "
            f"`{(row['runtime_seconds'] or 0.0):.3f}s` | {key_metric} |"
        )

    lines.extend(
        [
            "",
            "## Paired comparison snapshot",
            "",
            "| Comparison | Metric | Seeds | Left | Right | Relative advantage | p-value | Practical note |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["comparison_rows"]:
        lines.append(
            f"| `{row['comparison_id']}` | `{row['metric']}` | `{row['seeds']}` | "
            f"`{row['left_mean']:.4f}` | `{row['right_mean']:.4f}` | "
            f"`{_format_relative(row['relative_advantage'])}` | `{row['p_value']:.4g}` | {row['practical_significance_note']} |"
        )

    lines.extend(
        [
            "",
            "## Classification",
            "",
            "| Item | Status | Scope | Reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in summary["classification_rows"]:
        lines.append(
            f"| {row['item']} | `{row['status']}` | {row['scope']} | {row['reason']} |"
        )
    return "\n".join(lines) + "\n"


def run_manifests(
    manifest_paths: list[str | Path],
    *,
    output_root: str | Path,
    summary_stem: str,
) -> dict[str, Any]:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    all_run_rows: list[dict[str, Any]] = []
    entry_registry: dict[str, AblationEntry] = {}
    suite_inputs: dict[str, str] = {}
    manifests_payload: dict[str, Any] = {}

    for manifest_path in manifest_paths:
        manifest, run_rows, suite_dir, suite_entries = _run_manifest(manifest_path, output_root_path)
        suite_name = str(manifest["suite_name"])
        suite_inputs[suite_name] = str(suite_dir.as_posix())
        manifests_payload[suite_name] = manifest
        all_run_rows.extend(run_rows)
        entry_registry.update(suite_entries)

    aggregate_rows = _aggregate_rows(all_run_rows)
    comparison_rows = _comparison_rows(entry_registry, all_run_rows)
    classification_rows = _classification_rows(comparison_rows)
    run_metadata = build_run_metadata(
        project_root=PROJECT_ROOT,
        summary_stem=summary_stem,
        output_root=output_root_path,
        manifest_paths=manifest_paths,
        extra={
            "suite_kind": "ablation",
            "suite_names": sorted(suite_inputs),
            "run_row_count": len(all_run_rows),
        },
    )
    summary = {
        "summary_version": 1,
        "summary_schema_version": 2,
        "validated_scope": {
            problem: {
                "size_key": scope["size_key"],
                "sizes": list(scope["sizes"]),
            }
            for problem, scope in basecmp.VALIDATED_SCOPE.items()
        },
        "suite_inputs": suite_inputs,
        "manifests": manifests_payload,
        "fairness_policy": {
            "primary_basis": "matched_function_evaluation_budget",
            "initial_population_evaluations_included": True,
            "final_post_loop_population_re_evaluation_included": True,
            "hybrid_extra_evaluations_counted_in_budget": True,
            "repair_without_objective_calls_counted_only_as_note": True,
            "runtime_is_secondary": True,
        },
        "budget_definition": {
            "ga": basecmp.budget_formula_text("ga"),
            "hybrid_ga": basecmp.budget_formula_text("hybrid_ga"),
            "nsga2": basecmp.budget_formula_text("nsga2"),
        },
        "official_claim_inventory": _inventory_rows(),
        "run_rows": all_run_rows,
        "aggregate_rows": aggregate_rows,
        "comparison_rows": comparison_rows,
        "classification_rows": classification_rows,
        "run_metadata": run_metadata,
    }

    (output_root_path / f"{summary_stem}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_run_metadata(output_root_path / f"{summary_stem}_run_metadata.json", run_metadata)
    if comparison_rows:
        basecmp._write_csv(
            output_root_path / f"{summary_stem}.csv",
            comparison_rows,
            COMPARISON_COLUMNS,
        )
    (output_root_path / f"{summary_stem}.md").write_text(
        _markdown_summary(summary),
        encoding="utf-8",
    )
    return summary
