from __future__ import annotations

import math
from statistics import mean
from typing import Any, cast

from ga_lab.config import GAConfig
from ga_lab.runner import RunResult


def _finite_values(summaries: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for summary in summaries:
        value = summary.get(key)
        if isinstance(value, int | float) and math.isfinite(value):
            values.append(float(value))
    return values


def _mean_if_present(summaries: list[dict[str, Any]], key: str) -> float | None:
    values = _finite_values(summaries, key)
    if not values:
        return None
    return mean(values)


def _common_value_if_same(summaries: list[dict[str, Any]], key: str) -> tuple[bool, Any]:
    sentinel = object()
    values = [summary.get(key, sentinel) for summary in summaries]
    if not values or any(value is sentinel for value in values):
        return False, None
    first = values[0]
    if all(value == first for value in values[1:]):
        return True, first
    return False, None


def _aggregate_multiobjective_progress(
    run_results: list[RunResult],
) -> list[dict[str, float | int]] | None:
    buckets: dict[int, dict[str, object]] = {}
    metric_names = (
        "pareto_front_size",
        "pareto_ratio",
        "hypervolume",
        "spread",
        "normalized_hypervolume",
        "convergence_speed",
    )

    for result in run_results:
        if not result.history or "pareto_front_size" not in result.history[0]:
            continue
        for row in result.history:
            generation = int(row["generation"])
            bucket = buckets.setdefault(generation, {"runs": 0})
            run_count = bucket.get("runs", 0)
            bucket["runs"] = run_count + 1 if isinstance(run_count, int) else 1
            for metric_name in metric_names:
                value = row.get(metric_name)
                if not (isinstance(value, int | float) and math.isfinite(value)):
                    continue
                metric_values = cast(list[float], bucket.setdefault(metric_name, []))
                metric_values.append(float(value))

    if not buckets:
        return None

    progress: list[dict[str, float | int]] = []
    for generation in sorted(buckets):
        bucket = buckets[generation]
        run_count = bucket.get("runs", 0)
        record: dict[str, float | int] = {
            "generation": generation,
            "runs": run_count if isinstance(run_count, int) else 0,
        }
        for metric_name in metric_names:
            metric_payload = bucket.get(metric_name)
            if isinstance(metric_payload, list) and metric_payload:
                metric_values = cast(list[float], metric_payload)
                record[f"mean_{metric_name}"] = mean(metric_values)
        progress.append(record)
    return progress


def build_grid_summary(
    base_config: GAConfig,
    run_results: list[RunResult],
    seeds: int,
) -> dict[str, Any]:
    summaries = [result.summary for result in run_results]
    aggregate: dict[str, Any] = {
        "run_name": base_config.run_name,
        "problem": base_config.problem,
        "seeds": seeds,
        "seed_range_start": base_config.seed,
        "mean_best_fitness": mean(float(summary["best_fitness"]) for summary in summaries),
        "mean_final_generation": mean(float(summary["final_generation"]) for summary in summaries),
        "runs": summaries,
    }

    for source_key, target_key in (
        ("pareto_ratio", "mean_pareto_ratio"),
        ("hypervolume", "mean_hypervolume"),
        ("spread", "mean_spread"),
        ("mean_convergence_speed", "mean_convergence_speed"),
    ):
        value = _mean_if_present(summaries, source_key)
        if value is not None:
            aggregate[target_key] = value

    for key in (
        "hypervolume_reference_point",
        "hypervolume_reference_point_preset",
        "hypervolume_reference_point_override",
        "hypervolume_reference_point_source",
    ):
        present, value = _common_value_if_same(summaries, key)
        if present:
            aggregate[key] = value

    multiobjective_progress = _aggregate_multiobjective_progress(run_results)
    if multiobjective_progress is not None:
        aggregate["multiobjective_progress"] = multiobjective_progress

    return aggregate
