# ruff: noqa: E501

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ga_lab.benchmarks.external import (
    BENCHMARK_SOURCES,
    benchmark_inventory_rows,
    load_benchmark_problem_overrides,
)
from ga_lab.config import GAConfig
from ga_lab.core.representation import build_representation_adapter
from ga_lab.experiment import ablation_study as abl
from ga_lab.experiment import budget_baseline_comparison as basecmp
from ga_lab.factory import build_runtime_context
from ga_lab.governance.run_metadata import build_run_metadata, write_run_metadata
from ga_lab.utils.seed import make_rng

PROJECT_ROOT = basecmp.PROJECT_ROOT

RUN_COLUMNS = (
    "entry_id",
    "problem",
    "instance_or_family",
    "benchmark_source",
    "validated_internal_range",
    "size",
    "solver_family",
    "algorithm",
    "label",
    "seed",
    "preset_path",
    "configured_budget",
    "actual_evaluations_used",
    "extra_evaluations_from_hybrid",
    "runtime_seconds",
    "early_stop_triggered",
    "success_to_target",
    "evaluations_to_target",
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
    "instance_or_family",
    "benchmark_source",
    "validated_internal_range",
    "size",
    "solver_family",
    "algorithm",
    "label",
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
    "comparison_id",
    "entry_id",
    "problem",
    "instance_or_family",
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
    "practical_significance_note",
)

CLAIM_COLUMNS = (
    "item",
    "internal_status",
    "external_status",
    "evidence",
    "note",
)

FREEZE_COLUMNS = (
    "problem_family",
    "current_internal_official_claim",
    "current_external_supported_claim",
    "current_strongest_comparator",
    "remaining_gap",
    "should_this_become_family_conditional",
    "evidence_still_missing",
)


@dataclass(slots=True)
class ExternalMethod:
    label: str
    kind: str
    solver_family: str
    family: str | None = None
    overrides: dict[str, Any] | None = None
    preset_path: Path | None = None


@dataclass(slots=True)
class MetricComparison:
    left: str
    right: str
    metric: str
    objective: str
    comparison_id: str


@dataclass(slots=True)
class ExternalEntry:
    suite_name: str
    entry_id: str
    problem: str
    size: int
    preset_path: Path
    instance_or_family: str
    benchmark_source: str
    validated_internal_range: str
    benchmark_instance_id: str | None
    synthetic_family: str | None
    synthetic_options: dict[str, Any]
    base_overrides: dict[str, Any]
    methods: tuple[ExternalMethod, ...]
    comparisons: tuple[MetricComparison, ...]
    seeds: int
    seed_start: int


def _resolve_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    candidates = [
        (PROJECT_ROOT / path).resolve(),
        (manifest_path.parent / path).resolve(),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = json.loads(json.dumps(value))
    return merged


def _load_methods(manifest_path: Path, raw_entry: dict[str, Any]) -> tuple[ExternalMethod, ...]:
    raw_methods = raw_entry.get("methods")
    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError("External benchmark entry requires a non-empty methods list")
    methods: list[ExternalMethod] = []
    seen: set[str] = set()
    for raw_method in raw_methods:
        if not isinstance(raw_method, dict):
            raise ValueError("External benchmark methods must be objects")
        label = str(raw_method["label"])
        if label in seen:
            raise ValueError(f"Duplicate external benchmark method label: {label}")
        seen.add(label)
        preset_path = None
        if "preset" in raw_method:
            preset_path = _resolve_path(manifest_path, str(raw_method["preset"]))
        methods.append(
            ExternalMethod(
                label=label,
                kind=str(raw_method["kind"]),
                solver_family=str(raw_method.get("solver_family", "pure-ga")),
                family=str(raw_method["family"]) if "family" in raw_method else None,
                overrides=dict(raw_method.get("overrides", {})),
                preset_path=preset_path,
            )
        )
    return tuple(methods)


def _load_comparisons(raw_entry: dict[str, Any], labels: set[str]) -> tuple[MetricComparison, ...]:
    raw_comparisons = raw_entry.get("comparisons")
    if not isinstance(raw_comparisons, list) or not raw_comparisons:
        raise ValueError("External benchmark entry requires a non-empty comparisons list")
    comparisons: list[MetricComparison] = []
    seen: set[str] = set()
    for raw_comparison in raw_comparisons:
        if not isinstance(raw_comparison, dict):
            raise ValueError("External benchmark comparisons must be objects")
        comparison_id = str(raw_comparison["comparison_id"])
        if comparison_id in seen:
            raise ValueError(f"Duplicate external benchmark comparison id: {comparison_id}")
        seen.add(comparison_id)
        left = str(raw_comparison["left"])
        right = str(raw_comparison["right"])
        if left not in labels or right not in labels:
            raise ValueError(f"Comparison references unknown labels: {left}, {right}")
        objective = str(raw_comparison["objective"])
        if objective not in {"max", "min"}:
            raise ValueError("Comparison objective must be 'max' or 'min'")
        comparisons.append(
            MetricComparison(
                left=left,
                right=right,
                metric=str(raw_comparison["metric"]),
                objective=objective,
                comparison_id=comparison_id,
            )
        )
    return tuple(comparisons)


def load_manifest(path: str | Path) -> tuple[dict[str, Any], list[ExternalEntry]]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("External benchmark manifest must be a JSON object")
    suite_name = str(manifest.get("suite_name", manifest_path.stem))
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("External benchmark manifest requires a non-empty entries list")

    entries: list[ExternalEntry] = []
    seen_entry_ids: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Each external benchmark entry must be an object")
        entry_id = str(raw_entry["entry_id"])
        if entry_id in seen_entry_ids:
            raise ValueError(f"Duplicate external benchmark entry id: {entry_id}")
        seen_entry_ids.add(entry_id)
        methods = _load_methods(manifest_path, raw_entry)
        labels = {method.label for method in methods}
        benchmark = raw_entry.get("benchmark", {})
        if not isinstance(benchmark, dict):
            raise ValueError("benchmark must be a JSON object")
        entries.append(
            ExternalEntry(
                suite_name=suite_name,
                entry_id=entry_id,
                problem=str(raw_entry["problem"]),
                size=int(raw_entry["size"]),
                preset_path=_resolve_path(manifest_path, str(raw_entry["preset"])),
                instance_or_family=str(raw_entry["instance_or_family"]),
                benchmark_source=str(raw_entry["benchmark_source"]),
                validated_internal_range=str(raw_entry["validated_internal_range"]),
                benchmark_instance_id=(
                    str(benchmark["instance_id"]) if "instance_id" in benchmark else None
                ),
                synthetic_family=(
                    str(benchmark["synthetic_family"]) if "synthetic_family" in benchmark else None
                ),
                synthetic_options=dict(benchmark.get("synthetic_options", {})),
                base_overrides=dict(raw_entry.get("base_overrides", {})),
                methods=methods,
                comparisons=_load_comparisons(raw_entry, labels),
                seeds=int(raw_entry.get("seeds", manifest.get("default_seeds", 10))),
                seed_start=int(raw_entry.get("seed_start", 0)),
            )
        )
    return manifest, entries


def _benchmark_overrides(entry: ExternalEntry, cache_root: Path) -> dict[str, Any]:
    if entry.benchmark_instance_id is not None:
        return load_benchmark_problem_overrides(entry.benchmark_instance_id, cache_root=cache_root)
    if entry.synthetic_family is None:
        return {}
    if entry.problem == "onemax":
        return {
            "genome_length": entry.size,
            "target_fitness": entry.size,
            "problem_options": {
                "family": entry.synthetic_family,
                **entry.synthetic_options,
            },
        }
    if entry.problem == "zdt1":
        return {
            "genome_length": entry.size,
            "problem_options": {
                "family": entry.synthetic_family,
                **entry.synthetic_options,
            },
        }
    raise ValueError(f"Unsupported synthetic benchmark problem: {entry.problem}")


def _build_config_data(
    entry: ExternalEntry,
    method: ExternalMethod | None,
    *,
    seed: int,
    cache_root: Path,
) -> dict[str, Any]:
    base_path = (
        method.preset_path
        if method is not None and method.preset_path is not None
        else entry.preset_path
    )
    data = json.loads(base_path.read_text(encoding="utf-8"))
    data = _deep_merge(data, entry.base_overrides)
    data = _deep_merge(data, _benchmark_overrides(entry, cache_root))
    if method is not None and method.overrides:
        data = _deep_merge(data, method.overrides)
    data["seed"] = seed
    data["run_name"] = f"{entry.entry_id}_{method.label if method else 'base'}_{seed}"
    return data


def _run_optimizer_trial(
    entry: ExternalEntry,
    config: GAConfig,
    *,
    label: str,
    solver_family: str,
    preset_reference: str,
    seed: int,
) -> dict[str, Any]:
    runtime = build_runtime_context(config)
    tracked_problem = basecmp.TrackedProblem(runtime.problem, config)
    rng = make_rng(seed)
    started = time.perf_counter()
    algorithm_summary, _history = runtime.algorithm_fn(
        config=config,
        problem=tracked_problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=rng,
    )
    elapsed = time.perf_counter() - started
    tracked_metrics = tracked_problem.tracked_metrics()
    configured_budget = basecmp.configured_evaluation_budget(config)
    return {
        "entry_id": entry.entry_id,
        "problem": entry.problem,
        "instance_or_family": entry.instance_or_family,
        "benchmark_source": entry.benchmark_source,
        "validated_internal_range": entry.validated_internal_range,
        "size": entry.size,
        "solver_family": solver_family,
        "algorithm": config.algorithm,
        "label": label,
        "seed": seed,
        "preset_path": preset_reference,
        "configured_budget": configured_budget,
        "actual_evaluations_used": tracked_metrics["actual_evaluations"],
        "extra_evaluations_from_hybrid": (
            algorithm_summary.get("hybrid_extra_evaluations", 0.0) or 0.0
        ),
        "runtime_seconds": elapsed,
        "early_stop_triggered": algorithm_summary.get("stop_reason") != "max_generations",
        "success_to_target": (
            algorithm_summary.get("stop_reason") == "target_fitness_reached"
            if config.target_fitness is not None
            else None
        ),
        "evaluations_to_target": tracked_metrics.get("evaluations_to_target"),
        "final_best_fitness": algorithm_summary.get("best_fitness"),
        "final_best_distance": algorithm_summary.get("best_route_distance"),
        "best_feasible_fitness": tracked_metrics.get(
            "best_feasible_fitness",
            algorithm_summary.get("best_total_value")
            if algorithm_summary.get("best_is_feasible")
            else None,
        ),
        "feasible_rate": tracked_metrics.get("feasible_rate"),
        "mean_violation": tracked_metrics.get("mean_violation"),
        "hypervolume": algorithm_summary.get("hypervolume"),
        "pareto_ratio": algorithm_summary.get("pareto_ratio"),
        "spread": algorithm_summary.get("spread"),
        "pareto_front_size": algorithm_summary.get("pareto_front_size"),
    }


def _run_method(
    entry: ExternalEntry,
    method: ExternalMethod,
    *,
    seed: int,
    cache_root: Path,
) -> dict[str, Any]:
    config_data = _build_config_data(entry, method, seed=seed, cache_root=cache_root)
    config = GAConfig.from_dict(config_data)
    preset_reference = (
        str(method.preset_path.as_posix())
        if method.preset_path is not None
        else str(entry.preset_path.as_posix())
    )
    if method.kind == "baseline":
        adapter = build_representation_adapter(config)
        problem = build_runtime_context(config).problem
        tracked_problem = basecmp.TrackedProblem(problem, config)
        rng = make_rng(seed)
        budget = basecmp.configured_evaluation_budget(config)
        runners = {
            ("onemax", "random_search"): basecmp._run_onemax_random_search,
            ("onemax", "hill_climb"): basecmp._run_onemax_hill_climb,
            ("knapsack", "random_sampling"): basecmp._run_knapsack_random_sampling,
            ("knapsack", "greedy_local_search"): basecmp._run_knapsack_greedy_local_search,
            ("tsp", "random_tours"): basecmp._run_tsp_random_tours,
            ("tsp", "nearest_neighbor_2opt"): basecmp._run_tsp_nearest_neighbor_2opt,
            ("zdt1", "random_archive"): basecmp._run_zdt1_random_archive,
            ("zdt1", "mutation_archive"): basecmp._run_zdt1_mutation_archive,
        }
        runner = runners[(config.problem, method.family or "")]
        started = time.perf_counter()
        payload = runner(config, tracked_problem, adapter, rng, budget)
        elapsed = time.perf_counter() - started
        tracked_metrics = tracked_problem.tracked_metrics()
        return {
            "entry_id": entry.entry_id,
            "problem": entry.problem,
            "instance_or_family": entry.instance_or_family,
            "benchmark_source": entry.benchmark_source,
            "validated_internal_range": entry.validated_internal_range,
            "size": entry.size,
            "solver_family": method.solver_family,
            "algorithm": "baseline",
            "label": method.label,
            "seed": seed,
            "preset_path": preset_reference,
            "configured_budget": budget,
            "actual_evaluations_used": tracked_metrics["actual_evaluations"],
            "extra_evaluations_from_hybrid": 0.0,
            "runtime_seconds": elapsed,
            "early_stop_triggered": False,
            "success_to_target": payload.get("success_to_target"),
            "evaluations_to_target": tracked_metrics.get("evaluations_to_target"),
            "final_best_fitness": payload.get("final_best_fitness"),
            "final_best_distance": payload.get("final_best_distance"),
            "best_feasible_fitness": payload.get(
                "best_feasible_fitness",
                tracked_metrics.get("best_feasible_fitness"),
            ),
            "feasible_rate": payload.get("feasible_rate", tracked_metrics.get("feasible_rate")),
            "mean_violation": payload.get("mean_violation", tracked_metrics.get("mean_violation")),
            "hypervolume": payload.get("hypervolume"),
            "pareto_ratio": payload.get("pareto_ratio"),
            "spread": payload.get("spread"),
            "pareto_front_size": payload.get("pareto_front_size"),
        }
    if method.kind not in {"base_preset", "override", "preset"}:
        raise ValueError(f"Unsupported external benchmark method kind: {method.kind}")
    return _run_optimizer_trial(
        entry,
        config,
        label=method.label,
        solver_family=method.solver_family,
        preset_reference=preset_reference,
        seed=seed,
    )


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["entry_id"],
            row["problem"],
            row["instance_or_family"],
            row["label"],
            row["solver_family"],
            row["algorithm"],
            row["preset_path"],
            row["configured_budget"],
        )
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(grouped)):
        bucket = grouped[key]
        sample = bucket[0]
        summary_seed = 4000 + index * 10
        success_values = abl._series(bucket, "success_to_target")
        eval_values = abl._series(bucket, "evaluations_to_target")
        fitness_values = abl._series(bucket, "final_best_fitness")
        distance_values = abl._series(bucket, "final_best_distance")
        feasible_values = abl._series(bucket, "best_feasible_fitness")
        feasible_rate_values = abl._series(bucket, "feasible_rate")
        violation_values = abl._series(bucket, "mean_violation")
        hypervolume_values = abl._series(bucket, "hypervolume")
        pareto_ratio_values = abl._series(bucket, "pareto_ratio")
        spread_values = abl._series(bucket, "spread")
        front_size_values = abl._series(bucket, "pareto_front_size")
        actual_eval_values = abl._series(bucket, "actual_evaluations_used")
        extra_eval_values = abl._series(bucket, "extra_evaluations_from_hybrid")
        runtime_values = abl._series(bucket, "runtime_seconds")
        early_stop_values = abl._series(bucket, "early_stop_triggered")

        eval_stats = abl._summary_stats(eval_values, summary_seed + 1)
        fitness_stats = abl._summary_stats(fitness_values, summary_seed + 2)
        distance_stats = abl._summary_stats(distance_values, summary_seed + 3)
        feasible_stats = abl._summary_stats(feasible_values, summary_seed + 4)
        hv_stats = abl._summary_stats(hypervolume_values, summary_seed + 5)
        pareto_stats = abl._summary_stats(pareto_ratio_values, summary_seed + 6)
        spread_stats = abl._summary_stats(spread_values, summary_seed + 7)
        success_ci_low, success_ci_high = abl._bootstrap_ci(success_values, summary_seed + 8)

        aggregates.append(
            {
                "entry_id": sample["entry_id"],
                "problem": sample["problem"],
                "instance_or_family": sample["instance_or_family"],
                "benchmark_source": sample["benchmark_source"],
                "validated_internal_range": sample["validated_internal_range"],
                "size": sample["size"],
                "solver_family": sample["solver_family"],
                "algorithm": sample["algorithm"],
                "label": sample["label"],
                "preset_path": sample["preset_path"],
                "configured_budget": sample["configured_budget"],
                "seeds": len(bucket),
                "actual_evaluations_used": mean(actual_eval_values) if actual_eval_values else None,
                "extra_evaluations_from_hybrid": mean(extra_eval_values) if extra_eval_values else 0.0,
                "runtime_seconds": mean(runtime_values) if runtime_values else None,
                "early_stop_triggered": mean(early_stop_values) if early_stop_values else 0.0,
                "success_rate": mean(success_values) if success_values else None,
                "success_ci_low": success_ci_low,
                "success_ci_high": success_ci_high,
                "mean_evaluations_to_target": eval_stats["mean"],
                "evaluations_to_target_std": eval_stats["std"],
                "evaluations_to_target_median": eval_stats["median"],
                "evaluations_to_target_ci_low": eval_stats["ci_low"],
                "evaluations_to_target_ci_high": eval_stats["ci_high"],
                "mean_final_best_fitness": fitness_stats["mean"],
                "final_best_fitness_std": fitness_stats["std"],
                "final_best_fitness_median": fitness_stats["median"],
                "final_best_fitness_ci_low": fitness_stats["ci_low"],
                "final_best_fitness_ci_high": fitness_stats["ci_high"],
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
                "mean_feasible_rate": mean(feasible_rate_values) if feasible_rate_values else None,
                "feasible_rate_std": (
                    0.0
                    if len(feasible_rate_values) == 1
                    else stdev(feasible_rate_values)
                    if feasible_rate_values
                    else None
                ),
                "mean_violation": mean(violation_values) if violation_values else None,
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
                "mean_pareto_front_size": mean(front_size_values) if front_size_values else None,
                "pareto_front_size_std": (
                    0.0
                    if len(front_size_values) == 1
                    else stdev(front_size_values)
                    if front_size_values
                    else None
                ),
            }
        )
    return aggregates


def _comparison_rows(entries: dict[str, ExternalEntry], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        index[(row["entry_id"], row["label"], int(row["seed"]))] = row

    comparisons: list[dict[str, Any]] = []
    for entry_id, entry in sorted(entries.items()):
        for comparison in entry.comparisons:
            pairs: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
            for row in rows:
                if row["entry_id"] != entry_id or row["label"] != comparison.left:
                    continue
                right_row = index.get((entry_id, comparison.right, int(row["seed"])))
                if right_row is None:
                    continue
                left_value = abl._to_scalar(row.get(comparison.metric))
                right_value = abl._to_scalar(right_row.get(comparison.metric))
                if left_value is None or right_value is None:
                    continue
                pairs.append((row, right_row, left_value, right_value))

            left_values = [left for _, _, left, _ in pairs]
            right_values = [right for _, _, _, right in pairs]
            oriented_diffs = [
                left - right if comparison.objective == "max" else right - left
                for left, right in zip(left_values, right_values, strict=True)
            ]
            left_stats = abl._summary_stats(left_values, 7000 + len(comparisons) * 10 + 1)
            right_stats = abl._summary_stats(right_values, 7000 + len(comparisons) * 10 + 2)
            diff_ci_low, diff_ci_high = abl._bootstrap_ci(
                oriented_diffs,
                7000 + len(comparisons) * 10 + 3,
            )
            test_name, p_value = abl._paired_test(oriented_diffs)
            effect_size = abl._rank_biserial(oriented_diffs)
            left_rows = [left_row for left_row, _, _, _ in pairs]
            right_rows = [right_row for _, right_row, _, _ in pairs]
            runtime_left = abl._mean_series(left_rows, "runtime_seconds")
            runtime_right = abl._mean_series(right_rows, "runtime_seconds")
            eval_left = abl._mean_series(left_rows, "actual_evaluations_used")
            eval_right = abl._mean_series(right_rows, "actual_evaluations_used")
            extra_left = abl._mean_series(left_rows, "extra_evaluations_from_hybrid") or 0.0
            extra_right = abl._mean_series(right_rows, "extra_evaluations_from_hybrid") or 0.0
            relative_advantage = abl._relative_advantage(
                left_stats["mean"],
                right_stats["mean"],
                comparison.objective,
            )
            comparisons.append(
                {
                    "comparison_id": comparison.comparison_id,
                    "entry_id": entry_id,
                    "problem": entry.problem,
                    "instance_or_family": entry.instance_or_family,
                    "metric": comparison.metric,
                    "objective": comparison.objective,
                    "left": comparison.left,
                    "right": comparison.right,
                    "seeds": len(pairs),
                    "configured_budget": pairs[0][0]["configured_budget"] if pairs else None,
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
                    "oriented_mean_diff": mean(oriented_diffs) if oriented_diffs else None,
                    "oriented_diff_ci_low": diff_ci_low,
                    "oriented_diff_ci_high": diff_ci_high,
                    "relative_advantage": relative_advantage,
                    "paired_effect_size": effect_size,
                    "paired_significance_test": test_name,
                    "p_value": p_value,
                    "runtime_seconds_left": runtime_left,
                    "runtime_seconds_right": runtime_right,
                    "actual_evaluations_used_left": eval_left,
                    "actual_evaluations_used_right": eval_right,
                    "extra_evaluations_from_hybrid_left": extra_left,
                    "extra_evaluations_from_hybrid_right": extra_right,
                    "practical_significance_note": abl._practical_note(
                        p_value=p_value,
                        ci_low=diff_ci_low,
                        ci_high=diff_ci_high,
                        relative_advantage=relative_advantage,
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


def _claim_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    k50 = _comparison_lookup(comparison_rows, "kplib50_greedy_vs_pure_ga")
    k50_h = _comparison_lookup(comparison_rows, "kplib50_greedy_vs_seedrepair")
    k100 = _comparison_lookup(comparison_rows, "kplib100_greedy_vs_pure_ga")
    k100_h = _comparison_lookup(comparison_rows, "kplib100_greedy_vs_seedrepair")
    tsp_medium_h = _comparison_lookup(comparison_rows, "tsplib22_hybrid_vs_baseline")
    tsp_large_p = _comparison_lookup(comparison_rows, "tsplib52_baseline_vs_pure_ga")
    zdt2_hv = _comparison_lookup(comparison_rows, "zdt2_nsga2_vs_random_hv")
    zdt3_hv = _comparison_lookup(comparison_rows, "zdt3_nsga2_vs_random_hv")

    rows: list[dict[str, str]] = []
    rows.append(
        {
            "item": "onemax hill-climb practical default",
            "internal_status": "Internal official",
            "external_status": "Experimental / insufficient evidence",
            "evidence": "leadingones_hill_vs_pure_ga_fitness / trap_hill_vs_pure_ga_fitness",
            "note": (
                "LeadingOnes에서는 hill climb가 유리하지만, deceptive trap까지 같은 practical-default 문장으로"
                " 일반화하기에는 근거가 약하다."
            ),
        }
    )

    knapsack_supported = (
        k50 is not None
        and k50_h is not None
        and k100 is not None
        and k100_h is not None
        and (k50["oriented_diff_ci_low"] or -1.0) >= 0.0
        and (k50_h["oriented_diff_ci_low"] or -1.0) >= 0.0
        and (k100["oriented_diff_ci_low"] or -1.0) >= 0.0
        and (k100_h["oriented_diff_ci_low"] or -1.0) >= 0.0
    )
    rows.append(
        {
            "item": "knapsack greedy_local_search practical default",
            "internal_status": "Internal official",
            "external_status": "External supported" if knapsack_supported else "Experimental / insufficient evidence",
            "evidence": "kplib50_greedy_vs_pure_ga / kplib50_greedy_vs_seedrepair / kplib100_greedy_vs_pure_ga / kplib100_greedy_vs_seedrepair",
            "note": (
                "tested kplib subset에서는 greedy_local_search가 계속 실용 기본선으로 남는다."
                if knapsack_supported
                else "external knapsack subset에서도 greedy baseline이 대체로 경쟁적이지만, support를 확정하려면 더 넓은 subset이 필요하다."
            ),
        }
    )

    hybrid_supported = (
        tsp_medium_h is not None
        and tsp_medium_h["p_value"] < 0.05
        and (tsp_medium_h["oriented_diff_ci_low"] or -1.0) > 0.0
    )
    rows.append(
        {
            "item": "tsp medium hybrid quality-first path",
            "internal_status": "Internal official",
            "external_status": "External supported" if hybrid_supported else "Experimental / insufficient evidence",
            "evidence": "tsplib22_hybrid_vs_baseline",
            "note": (
                "ulysses22에서도 hybrid가 baseline을 이겼다."
                if hybrid_supported
                else "ulysses22 결과만으로 internal medium hybrid 승격을 external claim으로 확장하기는 어렵다."
            ),
        }
    )

    tsp_default_supported = (
        tsp_large_p is not None
        and tsp_large_p["p_value"] < 0.05
        and (tsp_large_p["oriented_diff_ci_low"] or -1.0) > 0.0
    )
    rows.append(
        {
            "item": "tsp nearest-neighbor + 2-opt practical default",
            "internal_status": "Internal official",
            "external_status": "External supported" if tsp_default_supported else "Experimental / insufficient evidence",
            "evidence": "tsplib52_baseline_vs_pure_ga",
            "note": (
                "tested TSPLIB subset에서는 nearest-neighbor + 2-opt practical default가 유지된다."
                if tsp_default_supported
                else "tested TSPLIB subset만으로 practical default support를 더 넓게 일반화하긴 이르다."
            ),
        }
    )

    zdt_supported = (
        zdt2_hv is not None
        and zdt3_hv is not None
        and zdt2_hv["p_value"] < 0.05
        and (zdt2_hv["oriented_diff_ci_low"] or -1.0) > 0.0
        and zdt3_hv["p_value"] < 0.05
        and (zdt3_hv["oriented_diff_ci_low"] or -1.0) > 0.0
    )
    rows.append(
        {
            "item": "zdt NSGA-II default path",
            "internal_status": "Internal official",
            "external_status": "External supported" if zdt_supported else "Experimental / insufficient evidence",
            "evidence": "zdt2_nsga2_vs_random_hv / zdt3_nsga2_vs_random_hv",
            "note": (
                "ZDT2/ZDT3 tested subset에서는 NSGA-II default가 random archive 대비 일관되게 우세하다."
                if zdt_supported
                else "ZDT family support 신호는 있지만, metric-priority 문장을 더 넓게 고정하기엔 아직 좁다."
            ),
        }
    )
    return rows


def _freeze_rows(claim_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "problem_family": "bitstring / onemax-adjacent",
            "current_internal_official_claim": "onemax validated range에서는 hill climb가 practical default",
            "current_strongest_comparator": "hill_climb",
            "validated_internal_range": "32 / 64 / 128",
            "external_validity_status": "partially tested",
            "most_useful_external_benchmark_target": "LeadingOnes and deceptive trap at 128 bits",
        },
        {
            "problem_family": "knapsack",
            "current_internal_official_claim": "greedy_local_search가 practical default",
            "current_strongest_comparator": "greedy_local_search",
            "validated_internal_range": "20 / 30 / 80",
            "external_validity_status": "tested",
            "most_useful_external_benchmark_target": "kplib uncorrelated n=50 and strongly correlated n=100",
        },
        {
            "problem_family": "tsp",
            "current_internal_official_claim": "practical default는 nearest-neighbor + 2-opt, hybrid official path는 medium(20)만",
            "current_strongest_comparator": "nearest_neighbor_2opt",
            "validated_internal_range": "10 / 20 / 50",
            "external_validity_status": "tested",
            "most_useful_external_benchmark_target": "TSPLIB ulysses22 and berlin52",
        },
        {
            "problem_family": "multi-objective / zdt",
            "current_internal_official_claim": "pure NSGA-II가 default path",
            "current_strongest_comparator": "random_archive / mutation_archive",
            "validated_internal_range": "10 / 20 / 50",
            "external_validity_status": "tested",
            "most_useful_external_benchmark_target": "ZDT2 and ZDT3",
        },
    ]


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# External benchmark summary",
        "",
        "## Current internal claims freeze",
        "",
        "| Problem family | Current internal official claim | Current external supported claim | Current strongest comparator | Remaining gap | Should this become family-conditional? | Evidence still missing |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["freeze_rows"]:
        lines.append(
            "| {problem_family} | {current_internal_official_claim} | {current_external_supported_claim} | {current_strongest_comparator} | {remaining_gap} | {should_this_become_family_conditional} | {evidence_still_missing} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## External benchmark inventory and provenance",
            "",
            "| Instance | Problem | Family | Size | Source | Redistribution | Note |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["benchmark_inventory_rows"]:
        lines.append(
            f"| `{row['instance_id']}` | {row['problem']} | {row['family']} | {row['size']} | {row['source_name']} | {row['redistribution_policy']} | {row['reference_note']} |"
        )

    lines.extend(
        [
            "",
            "## Fairness policy",
            "",
            "- 1차 비교 기준은 matched function evaluation budget이다.",
            "- 초기 population 평가와 final 평가를 budget에 포함한다.",
            "- hybrid local search가 objective call을 쓰면 `extra_evaluations_from_hybrid`에 기록하고 같은 budget 안에 포함한다.",
            "- runtime은 같이 기록하지만 1차 승부 기준은 아니다.",
            "",
            "## Claim status",
            "",
            "| Item | Internal status | External status | Evidence | Note |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["claim_rows"]:
        lines.append(
            f"| {row['item']} | {row['internal_status']} | {row['external_status']} | {row['evidence']} | {row['note']} |"
        )
    return "\n".join(lines) + "\n"


def _family_supported(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    p_value = row.get("p_value")
    ci_low = row.get("oriented_diff_ci_low")
    return isinstance(p_value, int | float) and isinstance(ci_low, int | float) and p_value < 0.05 and ci_low > 0.0


def _family_right_supported(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    p_value = row.get("p_value")
    ci_high = row.get("oriented_diff_ci_high")
    return isinstance(p_value, int | float) and isinstance(ci_high, int | float) and p_value < 0.05 and ci_high < 0.0


def _family_conditional_claim_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    onemax_hill = _comparison_lookup(comparison_rows, "onemax_hill_vs_pure_evals")
    leading_hill = _comparison_lookup(comparison_rows, "leadingones_hill_vs_pure_fitness")
    trap_pure_hill = _comparison_lookup(comparison_rows, "trap_pure_vs_hill_fitness")
    trap_pure_mut = _comparison_lookup(comparison_rows, "trap_pure_vs_mutation_fitness")
    jump_pure_hill = _comparison_lookup(comparison_rows, "jump_pure_vs_hill_fitness")
    jump_mut_hill = _comparison_lookup(comparison_rows, "jump_mutation_vs_hill_fitness")
    jump_pure_mut = _comparison_lookup(comparison_rows, "jump_pure_vs_mutation_fitness")

    uncorr_greedy_pure = _comparison_lookup(comparison_rows, "kplib_uncorr50_greedy_vs_pure")
    uncorr_seed_greedy = _comparison_lookup(comparison_rows, "kplib_uncorr50_seedrepair_vs_greedy")
    weak_greedy_pure = _comparison_lookup(comparison_rows, "kplib_weak50_greedy_vs_pure")
    weak_seed_greedy = _comparison_lookup(comparison_rows, "kplib_weak50_seedrepair_vs_greedy")
    strong_greedy_pure = _comparison_lookup(comparison_rows, "kplib_strong100_greedy_vs_pure")
    strong_seed_greedy = _comparison_lookup(comparison_rows, "kplib_strong100_seedrepair_vs_greedy")
    subset_greedy_pure = _comparison_lookup(comparison_rows, "kplib_subset100_greedy_vs_pure")
    subset_seed_greedy = _comparison_lookup(comparison_rows, "kplib_subset100_seedrepair_vs_greedy")

    tsp_medium_h = _comparison_lookup(comparison_rows, "tsplib22_hybrid_vs_baseline")
    tsp_medium_p = _comparison_lookup(comparison_rows, "tsplib22_baseline_vs_pure_ga")
    tsp_large_p = _comparison_lookup(comparison_rows, "tsplib52_baseline_vs_pure_ga")
    tsp_large_h = _comparison_lookup(comparison_rows, "tsplib52_baseline_vs_large_hybrid")
    zdt2_hv = _comparison_lookup(comparison_rows, "zdt2_nsga2_vs_random_hv")
    zdt3_hv = _comparison_lookup(comparison_rows, "zdt3_nsga2_vs_random_hv")

    monotone_supported = _family_supported(onemax_hill) and _family_supported(leading_hill)
    trap_supported = _family_supported(trap_pure_hill) and _family_supported(trap_pure_mut)
    jump_pure_supported = _family_supported(jump_pure_hill)
    jump_mut_supported = _family_supported(jump_mut_hill)
    jump_pure_over_mut = _family_supported(jump_pure_mut)
    jump_mut_over_pure = _family_right_supported(jump_pure_mut)

    def knapsack_status(greedy_row: dict[str, Any] | None, seed_row: dict[str, Any] | None) -> tuple[str, str]:
        greedy_over_pure = _family_supported(greedy_row)
        seed_over_greedy = _family_supported(seed_row)
        if seed_over_greedy and greedy_over_pure:
            return (
                "Family-conditional external",
                "테스트한 family에서는 seed/repair hybrid가 greedy보다 낫고, pure GA 대비 gap closing도 유지된다.",
            )
        if greedy_over_pure and not seed_over_greedy:
            return (
                "Family-conditional external",
                "테스트한 family에서는 greedy_local_search가 pure GA보다 낫고, hybrid도 default를 바꿀 만큼 일관적이지 않다.",
            )
        return (
            "Experimental / insufficient evidence",
            "테스트한 family에서 solver ranking 신호는 있으나 외부 기본 경로로 고정하기에는 아직 약하다.",
        )

    uncorr_status, uncorr_note = knapsack_status(uncorr_greedy_pure, uncorr_seed_greedy)
    weak_status, weak_note = knapsack_status(weak_greedy_pure, weak_seed_greedy)
    strong_status, strong_note = knapsack_status(strong_greedy_pure, strong_seed_greedy)
    subset_status, subset_note = knapsack_status(subset_greedy_pure, subset_seed_greedy)

    tsp_default_supported = _family_supported(tsp_medium_p) and _family_supported(tsp_large_p)
    tsp_hybrid_supported = _family_supported(tsp_medium_h) and not _family_supported(tsp_large_h)
    zdt_supported = _family_supported(zdt2_hv) and _family_supported(zdt3_hv)

    return [
        {
            "item": "bitstring monotone family practical default",
            "internal_status": "Internal official",
            "external_status": "Family-conditional external" if monotone_supported else "Experimental / insufficient evidence",
            "evidence": "onemax_hill_vs_pure_evals / leadingones_hill_vs_pure_fitness",
            "note": (
                "OneMax와 LeadingOnes tested subset에서는 hill climb practical default가 유지된다."
                if monotone_supported
                else "monotone family support가 일부 있지만, tested subset만으로 확정하기에는 아직 좁다."
            ),
        },
        {
            "item": "bitstring deceptive trap family",
            "internal_status": "Note only",
            "external_status": "Family-conditional external" if trap_supported else "Experimental / insufficient evidence",
            "evidence": "trap_pure_vs_hill_fitness / trap_pure_vs_mutation_fitness",
            "note": (
                "tested trap family에서는 pure GA가 hill climb와 mutation-only EA보다 강하다."
                if trap_supported
                else "deceptive trap family에서 GA 우위 신호는 있지만 tested subset만으로 고정하기엔 이르다."
            ),
        },
        {
            "item": "bitstring Jump_k family",
            "internal_status": "Note only",
            "external_status": (
                "Family-conditional external"
                if jump_pure_supported or jump_mut_supported
                else "Experimental / insufficient evidence"
            ),
            "evidence": "jump_pure_vs_hill_fitness / jump_mutation_vs_hill_fitness / jump_pure_vs_mutation_fitness",
            "note": (
                "tested Jump_k family에서는 pure GA가 hill climb보다 낫고 mutation-only EA보다도 우세하다."
                if jump_pure_supported and jump_pure_over_mut
                else "tested Jump_k family에서는 mutation-only EA가 hill climb보다 낫고 pure GA보다도 우세하다."
                if jump_mut_supported and jump_mut_over_pure
                else "tested Jump_k family에서는 GA-family가 hill climb보다 낫지만 pure GA와 mutation-only EA의 우열은 아직 좁다."
                if jump_pure_supported or jump_mut_supported
                else "Jump_k tested subset만으로는 multimodal bitstring default를 고정하기 어렵다."
            ),
        },
        {
            "item": "knapsack uncorrelated family",
            "internal_status": "Internal official",
            "external_status": uncorr_status,
            "evidence": "kplib_uncorr50_greedy_vs_pure / kplib_uncorr50_seedrepair_vs_greedy",
            "note": uncorr_note,
        },
        {
            "item": "knapsack weakly correlated family",
            "internal_status": "Internal official",
            "external_status": weak_status,
            "evidence": "kplib_weak50_greedy_vs_pure / kplib_weak50_seedrepair_vs_greedy",
            "note": weak_note,
        },
        {
            "item": "knapsack strongly correlated family",
            "internal_status": "Internal official",
            "external_status": strong_status,
            "evidence": "kplib_strong100_greedy_vs_pure / kplib_strong100_seedrepair_vs_greedy",
            "note": strong_note,
        },
        {
            "item": "knapsack subset-sum family",
            "internal_status": "Internal official",
            "external_status": subset_status,
            "evidence": "kplib_subset100_greedy_vs_pure / kplib_subset100_seedrepair_vs_greedy",
            "note": subset_note,
        },
        {
            "item": "tsp nearest-neighbor + 2-opt practical default",
            "internal_status": "Internal official",
            "external_status": "External supported" if tsp_default_supported else "Experimental / insufficient evidence",
            "evidence": "tsplib22_baseline_vs_pure_ga / tsplib52_baseline_vs_pure_ga",
            "note": (
                "tested TSPLIB subset에서는 nearest-neighbor + 2-opt practical default가 유지된다."
                if tsp_default_supported
                else "tested TSPLIB subset만으로 practical default support를 더 넓게 일반화하긴 이르다."
            ),
        },
        {
            "item": "tsp medium hybrid quality-first path",
            "internal_status": "Internal official",
            "external_status": "Experimental / insufficient evidence",
            "evidence": "tsplib22_hybrid_vs_baseline / tsplib52_baseline_vs_large_hybrid",
            "note": (
                "external subset에서는 medium hybrid를 official external path로 넓힐 근거가 아직 없다."
                if not tsp_hybrid_supported
                else "ulysses22에서는 hybrid 이득이 있으나, berlin52 maintenance check까지 합치면 external official path로 넓히긴 아직 좁다."
            ),
        },
        {
            "item": "zdt NSGA-II default path",
            "internal_status": "Internal official",
            "external_status": "External supported" if zdt_supported else "Experimental / insufficient evidence",
            "evidence": "zdt2_nsga2_vs_random_hv / zdt3_nsga2_vs_random_hv",
            "note": (
                "ZDT2/ZDT3 tested subset에서는 NSGA-II default가 random archive 대비 일관되게 우세하다."
                if zdt_supported
                else "ZDT family tested subset에서 support 신호는 있으나 더 넓은 일반화는 아직 이르다."
            ),
        },
    ]


def _family_conditional_freeze_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    monotone_supported = _family_supported(_comparison_lookup(comparison_rows, "onemax_hill_vs_pure_evals")) and _family_supported(
        _comparison_lookup(comparison_rows, "leadingones_hill_vs_pure_fitness")
    )
    trap_supported = _family_supported(_comparison_lookup(comparison_rows, "trap_pure_vs_hill_fitness"))
    jump_supported = _family_supported(_comparison_lookup(comparison_rows, "jump_pure_vs_hill_fitness")) or _family_supported(
        _comparison_lookup(comparison_rows, "jump_mutation_vs_hill_fitness")
    )
    uncorr_seed = _family_supported(_comparison_lookup(comparison_rows, "kplib_uncorr50_seedrepair_vs_greedy"))
    weak_seed = _family_supported(_comparison_lookup(comparison_rows, "kplib_weak50_seedrepair_vs_greedy"))
    strong_seed = _family_supported(_comparison_lookup(comparison_rows, "kplib_strong100_seedrepair_vs_greedy"))
    subset_seed = _family_supported(_comparison_lookup(comparison_rows, "kplib_subset100_seedrepair_vs_greedy"))
    tsp_supported = _family_supported(_comparison_lookup(comparison_rows, "tsplib22_baseline_vs_pure_ga")) and _family_supported(
        _comparison_lookup(comparison_rows, "tsplib52_baseline_vs_pure_ga")
    )
    zdt_supported = _family_supported(_comparison_lookup(comparison_rows, "zdt2_nsga2_vs_random_hv")) and _family_supported(
        _comparison_lookup(comparison_rows, "zdt3_nsga2_vs_random_hv")
    )

    return [
        {
            "problem_family": "bitstring family",
            "current_internal_official_claim": "onemax validated size에서는 hill climb가 practical default다.",
            "current_external_supported_claim": (
                "monotone family(OneMax, LeadingOnes)에서는 hill climb practical default가 유지된다."
                if monotone_supported
                else "아직 broad external default는 없다."
            ),
            "current_strongest_comparator": "hill_climb / pure_ga / mutation_only_ea",
            "remaining_gap": (
                "deceptive trap과 Jump_k는 monotone과 solver ranking이 다르다."
                if trap_supported or jump_supported
                else "deceptive / multimodal family에서 broad solver rule이 아직 닫히지 않았다."
            ),
            "should_this_become_family_conditional": "Yes",
            "evidence_still_missing": "추가 Jump_k 설정, 더 다양한 deceptive family, crossover-heavy GA comparator",
        },
        {
            "problem_family": "knapsack family",
            "current_internal_official_claim": "validated internal range에서는 greedy_local_search가 practical default다.",
            "current_external_supported_claim": (
                "external tested family에서는 single broad default보다 family-conditioned rule이 더 정직하다."
                if any((uncorr_seed, weak_seed, strong_seed, subset_seed))
                else "greedy_local_search broad external default는 아직 충분히 닫히지 않았다."
            ),
            "current_strongest_comparator": "greedy_local_search / seed_repair_hybrid",
            "remaining_gap": "uncorrelated, weakly correlated, strongly correlated, subset-sum family에서 ranking이 갈릴 수 있다.",
            "should_this_become_family_conditional": "Yes",
            "evidence_still_missing": "더 넓은 kplib subset, 추가 seed sets, structured family 반복 확인",
        },
        {
            "problem_family": "tsp",
            "current_internal_official_claim": "practical default는 nearest-neighbor + 2-opt이고, hybrid official path는 medium(20)만 유지한다.",
            "current_external_supported_claim": (
                "tested TSPLIB subset에서는 nearest-neighbor + 2-opt practical default가 유지된다."
                if tsp_supported
                else "nearest-neighbor + 2-opt default의 external support는 유지 확인이 더 필요하다."
            ),
            "current_strongest_comparator": "nearest_neighbor_2opt",
            "remaining_gap": "medium hybrid를 external official path로 넓힐 근거는 아직 약하다.",
            "should_this_become_family_conditional": "No",
            "evidence_still_missing": "추가 TSPLIB subset은 있으면 좋지만, 현재 목표에는 필수는 아니다.",
        },
        {
            "problem_family": "multi-objective / zdt",
            "current_internal_official_claim": "validated internal range에서는 pure NSGA-II가 default path다.",
            "current_external_supported_claim": (
                "tested ZDT2/ZDT3 subset에서는 NSGA-II가 random archive 대비 우세하다."
                if zdt_supported
                else "NSGA-II external support는 일부 있으나 아직 더 넓은 family 일반화는 조심해야 한다."
            ),
            "current_strongest_comparator": "random_archive / mutation_archive",
            "remaining_gap": "large tier와 family별로 HV, spread, Pareto coverage tradeoff 문구가 계속 필요하다.",
            "should_this_become_family_conditional": "No",
            "evidence_still_missing": "추가 ZDT family와 metric-priority별 반복 확인",
        },
    ]


def run_manifests(
    manifest_paths: list[str | Path],
    *,
    output_root: str | Path,
    summary_stem: str = "external_benchmark_summary",
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(cache_root).resolve()
        if cache_root is not None
        else (PROJECT_ROOT / "benchmarks").resolve()
    )

    entries: dict[str, ExternalEntry] = {}
    run_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest, manifest_entries = load_manifest(manifest_path)
        manifests.append(manifest)
        for entry in manifest_entries:
            entries[entry.entry_id] = entry
            for offset in range(entry.seeds):
                seed = entry.seed_start + offset
                for method in entry.methods:
                    run_rows.append(_run_method(entry, method, seed=seed, cache_root=cache_dir))

    aggregate_rows = _aggregate_rows(run_rows)
    comparison_rows = _comparison_rows(entries, run_rows)
    claim_rows = _family_conditional_claim_rows(comparison_rows)
    freeze_rows = _family_conditional_freeze_rows(comparison_rows)
    run_metadata = build_run_metadata(
        project_root=PROJECT_ROOT,
        summary_stem=summary_stem,
        output_root=output_dir,
        manifest_paths=manifest_paths,
        extra={
            "suite_kind": "external_benchmark",
            "manifest_count": len(manifests),
            "run_row_count": len(run_rows),
            "cache_root": str(cache_dir),
        },
    )
    summary = {
        "summary_schema_version": 2,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifests": manifests,
        "freeze_rows": freeze_rows,
        "benchmark_source_rows": BENCHMARK_SOURCES,
        "benchmark_inventory_rows": benchmark_inventory_rows(),
        "run_rows": run_rows,
        "aggregate_rows": aggregate_rows,
        "comparison_rows": comparison_rows,
        "claim_rows": claim_rows,
        "run_metadata": run_metadata,
    }

    (output_dir / f"{summary_stem}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_run_metadata(output_dir / f"{summary_stem}_run_metadata.json", run_metadata)
    basecmp._write_csv(output_dir / f"{summary_stem}.csv", comparison_rows, COMPARISON_COLUMNS)
    (output_dir / f"{summary_stem}.md").write_text(
        _markdown_summary(summary),
        encoding="utf-8",
    )
    return summary
