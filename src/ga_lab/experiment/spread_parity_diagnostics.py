from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

from ga_lab.experiment.nsga2_diagnostics import (
    compute_objective_bins,
    compute_segment_distribution,
    compute_spacing_by_segment,
    compute_zdt1_components,
    hash_solution_or_objective,
)


_LOWER_IS_BETTER_METRICS = {
    "empty_bins",
    "segment_load_std",
    "segment_load_gini",
    "spacing",
}

_HIGHER_IS_BETTER_METRICS = {
    "occupied_bins",
    "point_count_entropy",
    "total_nondominated_count",
}


@dataclass(slots=True)
class SpreadParityConfig:
    spread_parity_trace_enabled: bool = False
    segment_count: int = 6
    low_g_threshold: float = 1.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    value_float = float(value)
    if not math.isfinite(value_float):
        return None
    return value_float


def _safe_mean(values: Sequence[Any]) -> float | None:
    numeric = [value for item in values if (value := _finite(item)) is not None]
    if not numeric:
        return None
    return float(mean(numeric))


def _safe_std(values: Sequence[Any]) -> float | None:
    numeric = [value for item in values if (value := _finite(item)) is not None]
    if not numeric:
        return None
    baseline = float(mean(numeric))
    variance = sum((value - baseline) ** 2 for value in numeric) / len(numeric)
    return float(math.sqrt(variance))


def _segment_loads(segment_counts: Mapping[str, Any], *, bins: int) -> list[int]:
    return [int(segment_counts.get(str(segment_id), 0)) for segment_id in range(max(1, bins))]


def compute_segment_load_entropy(segment_loads: Sequence[int | float]) -> float | None:
    numeric = [max(0.0, float(value)) for value in segment_loads]
    total = sum(numeric)
    if total <= 0.0:
        return None
    probabilities = [value / total for value in numeric if value > 0.0]
    if not probabilities:
        return None
    return float(-sum(probability * math.log(probability, 2) for probability in probabilities))


def compute_segment_load_gini(segment_loads: Sequence[int | float]) -> float | None:
    numeric = sorted(max(0.0, float(value)) for value in segment_loads)
    total = sum(numeric)
    if total <= 0.0:
        return None
    count = len(numeric)
    weighted_sum = sum((2 * index - count - 1) * value for index, value in enumerate(numeric, start=1))
    return float(weighted_sum / (count * total))


def _safe_segment_range(segment_ranges: Sequence[Sequence[float]], segment_id: int) -> list[float]:
    if 0 <= segment_id < len(segment_ranges):
        raw = segment_ranges[segment_id]
        if isinstance(raw, Sequence):
            return [float(raw[0]), float(raw[1])] if len(raw) >= 2 else [0.0, 1.0]
    return [0.0, 1.0]


def _collect_population_components(
    decision_vectors: Sequence[Sequence[float | int]],
    objective_vectors: Sequence[Sequence[float]],
    *,
    bins: int,
    front_objective_vectors: Sequence[Sequence[float]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    aligned_count = min(len(decision_vectors), len(objective_vectors))
    if not decision_vectors:
        warnings.append("decision_vectors_unavailable")
    if not objective_vectors:
        warnings.append("objective_vectors_unavailable")
    if len(decision_vectors) != len(objective_vectors):
        warnings.append("decision_objective_length_mismatch")

    front_signatures = {
        hash_solution_or_objective(vector)
        for vector in (front_objective_vectors or [])
        if isinstance(vector, Sequence)
    }
    components: list[dict[str, Any]] = []
    for index in range(aligned_count):
        objective_signature = hash_solution_or_objective(objective_vectors[index])
        component = compute_zdt1_components(
            decision_vectors[index],
            objective_vectors[index],
            bins=bins,
            nondominated=(objective_signature in front_signatures) if front_signatures else None,
        )
        components.append(component)
        warnings.extend(
            warning
            for warning in component.get("warnings", [])
            if isinstance(warning, str)
        )
    return components, sorted(set(warnings))


def summarize_segment_allocation(
    decision_vectors: Sequence[Sequence[float | int]],
    objective_vectors: Sequence[Sequence[float]],
    front_objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    if not objective_vectors:
        return {
            "segment_count": bins,
            "segment_rows": [],
            "warnings": ["objective_vectors_unavailable"],
        }

    segment_distribution = compute_segment_distribution(objective_vectors, directions, bins=bins)
    point_segments = [int(value) for value in segment_distribution.get("point_segments", [])]
    segment_ranges = list(segment_distribution.get("segment_ranges", []))
    front_signatures = {
        hash_solution_or_objective(vector)
        for vector in front_objective_vectors
        if isinstance(vector, Sequence)
    }
    components, component_warnings = _collect_population_components(
        decision_vectors,
        objective_vectors,
        bins=bins,
        front_objective_vectors=front_objective_vectors,
    )

    grouped_components: dict[int, list[dict[str, Any]]] = {segment_id: [] for segment_id in range(max(1, bins))}
    segment_nondominated_counts: Counter[int] = Counter()
    segment_point_counts: Counter[int] = Counter()

    for index, objective_vector in enumerate(objective_vectors):
        segment_id = point_segments[index] if index < len(point_segments) else 0
        segment_point_counts[segment_id] += 1
        if hash_solution_or_objective(objective_vector) in front_signatures:
            segment_nondominated_counts[segment_id] += 1
        if index < len(components):
            grouped_components[segment_id].append(components[index])

    segment_rows: list[dict[str, Any]] = []
    total_points = len(objective_vectors)
    for segment_id in range(max(1, bins)):
        component_rows = grouped_components.get(segment_id, [])
        point_count = int(segment_point_counts.get(segment_id, 0))
        segment_rows.append(
            {
                "segment_id": segment_id,
                "segment_range": _safe_segment_range(segment_ranges, segment_id),
                "point_count": point_count,
                "nondominated_point_count": int(segment_nondominated_counts.get(segment_id, 0)),
                "segment_coverage_rate": (float(point_count / total_points) if total_points > 0 else None),
                "empty_segment": point_count == 0,
                "mean_g": _safe_mean([row.get("g") for row in component_rows]),
                "mean_distance": _safe_mean(
                    [row.get("distance_to_zdt1_front") for row in component_rows]
                ),
                "mean_f1": _safe_mean([row.get("f1") for row in component_rows]),
                "mean_f2": _safe_mean([row.get("f2") for row in component_rows]),
            }
        )

    warnings = list(segment_distribution.get("warnings", [])) + component_warnings
    return {
        "segment_count": bins,
        "segment_rows": segment_rows,
        "warnings": sorted(set(warnings)),
    }


def summarize_segment_spacing_contribution(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    spacing_summary = compute_spacing_by_segment(objective_vectors, directions, bins=bins)
    largest_gap_segment_id = spacing_summary.get("max_gap_segment_id")
    segment_rows = [
        {
            "segment_id": int(row.get("segment_id", 0)),
            "segment_range": list(row.get("segment_range", [0.0, 1.0])),
            "point_count": int(row.get("points_in_segment", 0)),
            "mean_local_gap": _finite(row.get("mean_gap")),
            "max_local_gap": _finite(row.get("max_gap")),
            "local_spacing_contribution": _finite(row.get("local_spacing_contribution")),
            "boundary_adjacent": int(row.get("segment_id", 0)) in {0, max(0, bins - 1)},
            "largest_gap_flag": (
                int(row.get("segment_id", -1)) == int(largest_gap_segment_id)
                if largest_gap_segment_id is not None
                else False
            ),
        }
        for row in spacing_summary.get("segment_rows", [])
    ]
    return {
        "segment_count": bins,
        "weakest_segment_id": spacing_summary.get("weak_segment_id"),
        "largest_gap_segment_id": spacing_summary.get("max_gap_segment_id"),
        "global_mean_gap": _finite(spacing_summary.get("global_mean_gap")),
        "segment_rows": segment_rows,
        "warnings": list(spacing_summary.get("warnings", [])),
    }


def summarize_occupancy_uniformity(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    occupancy = compute_objective_bins(objective_vectors, directions, bins=bins)
    segment_distribution = compute_segment_distribution(objective_vectors, directions, bins=bins)
    segment_counts = dict(segment_distribution.get("segment_counts", {}))
    loads = _segment_loads(segment_counts, bins=bins)
    nonempty_loads = [value for value in loads if value > 0]
    return {
        "occupied_bins": int(occupancy.get("occupied_bins", 0)),
        "empty_bins": int(occupancy.get("empty_bins", 0)),
        "point_count_entropy": compute_segment_load_entropy(loads),
        "max_segment_load": max(loads, default=0),
        "min_nonempty_segment_load": min(nonempty_loads) if nonempty_loads else 0,
        "segment_load_std": _safe_std(loads),
        "segment_load_gini": compute_segment_load_gini(loads),
        "segment_loads": {str(segment_id): int(value) for segment_id, value in enumerate(loads)},
        "warnings": sorted(
            set(list(occupancy.get("warnings", [])) + list(segment_distribution.get("warnings", [])))
        ),
    }


def summarize_nondominated_distribution(
    objective_vectors: Sequence[Sequence[float]],
    front_objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    if not objective_vectors:
        return {
            "total_nondominated_count": 0,
            "segment_rows": [],
            "warnings": ["objective_vectors_unavailable"],
        }

    segment_distribution = compute_segment_distribution(objective_vectors, directions, bins=bins)
    point_segments = [int(value) for value in segment_distribution.get("point_segments", [])]
    front_signatures = {
        hash_solution_or_objective(vector)
        for vector in front_objective_vectors
        if isinstance(vector, Sequence)
    }
    segment_point_counts: Counter[int] = Counter()
    segment_nondominated_counts: Counter[int] = Counter()
    for index, objective_vector in enumerate(objective_vectors):
        segment_id = point_segments[index] if index < len(point_segments) else 0
        segment_point_counts[segment_id] += 1
        if hash_solution_or_objective(objective_vector) in front_signatures:
            segment_nondominated_counts[segment_id] += 1

    segment_rows: list[dict[str, Any]] = []
    for segment_id in range(max(1, bins)):
        point_count = int(segment_point_counts.get(segment_id, 0))
        nondominated_count = int(segment_nondominated_counts.get(segment_id, 0))
        dominated_count = max(0, point_count - nondominated_count)
        segment_rows.append(
            {
                "segment_id": segment_id,
                "segment_nondominated_count": nondominated_count,
                "segment_nondominated_rate": (
                    float(nondominated_count / point_count) if point_count > 0 else None
                ),
                "segment_dominated_count": dominated_count,
                "segment_dominance_loss_rate": (
                    float(dominated_count / point_count) if point_count > 0 else None
                ),
            }
        )

    return {
        "total_nondominated_count": int(len(front_objective_vectors)),
        "segment_rows": segment_rows,
        "warnings": list(segment_distribution.get("warnings", [])),
    }


def summarize_decision_to_segment_mapping(
    decision_vectors: Sequence[Sequence[float | int]],
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    if not objective_vectors or not decision_vectors:
        warnings: list[str] = []
        if not decision_vectors:
            warnings.append("decision_vectors_unavailable")
        if not objective_vectors:
            warnings.append("objective_vectors_unavailable")
        return {
            "segment_count": bins,
            "segment_rows": [],
            "warnings": warnings,
        }

    segment_distribution = compute_segment_distribution(objective_vectors, directions, bins=bins)
    point_segments = [int(value) for value in segment_distribution.get("point_segments", [])]
    segment_ranges = list(segment_distribution.get("segment_ranges", []))
    components, component_warnings = _collect_population_components(
        decision_vectors,
        objective_vectors,
        bins=bins,
        front_objective_vectors=[],
    )

    grouped_components: dict[int, list[dict[str, Any]]] = {segment_id: [] for segment_id in range(max(1, bins))}
    for index, component in enumerate(components):
        segment_id = point_segments[index] if index < len(point_segments) else 0
        grouped_components[segment_id].append(component)

    segment_rows: list[dict[str, Any]] = []
    for segment_id in range(max(1, bins)):
        component_rows = grouped_components.get(segment_id, [])
        segment_rows.append(
            {
                "segment_id": segment_id,
                "segment_range": _safe_segment_range(segment_ranges, segment_id),
                "x0_mean": _safe_mean([row.get("x0") for row in component_rows]),
                "x0_std": _safe_std([row.get("x0") for row in component_rows]),
                "tail_mean_mean": _safe_mean([row.get("tail_mean") for row in component_rows]),
                "tail_mean_std": _safe_std([row.get("tail_mean") for row in component_rows]),
                "g_mean": _safe_mean([row.get("g") for row in component_rows]),
                "g_std": _safe_std([row.get("g") for row in component_rows]),
                "distance_mean": _safe_mean(
                    [row.get("distance_to_zdt1_front") for row in component_rows]
                ),
                "point_count": len(component_rows),
            }
        )

    warnings = list(segment_distribution.get("warnings", [])) + component_warnings
    return {
        "segment_count": bins,
        "segment_rows": segment_rows,
        "warnings": sorted(set(warnings)),
    }


def _largest_gap_segment(
    candidate_counts: Mapping[str, Any],
    reference_counts: Mapping[str, Any],
) -> str | None:
    all_segments = sorted(set(candidate_counts) | set(reference_counts), key=lambda value: int(value))
    if not all_segments:
        return None
    return str(
        max(
            all_segments,
            key=lambda segment: abs(
                float(candidate_counts.get(segment, 0.0)) - float(reference_counts.get(segment, 0.0))
            ),
        )
    )


def summarize_parity_spread_gap(
    algorithm_summaries: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidate_j = dict(algorithm_summaries.get("candidate_j_h_lite_retry2", {}))
    candidate_n = dict(algorithm_summaries.get("candidate_n_low_g_tail_mutation_light", {}))
    pymoo = dict(algorithm_summaries.get("pymoo_nsga2", {}))

    metrics = [
        "occupied_bins",
        "empty_bins",
        "point_count_entropy",
        "segment_load_std",
        "segment_load_gini",
        "spacing",
        "total_nondominated_count",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        candidate_j_value = _finite(candidate_j.get(metric))
        candidate_n_value = _finite(candidate_n.get(metric))
        pymoo_value = _finite(pymoo.get(metric))
        delta_vs_j = (
            None
            if candidate_j_value is None or candidate_n_value is None
            else float(candidate_n_value - candidate_j_value)
        )
        gap_vs_pymoo = (
            None
            if pymoo_value is None or candidate_n_value is None
            else float(candidate_n_value - pymoo_value)
        )

        point_gap_segment = _largest_gap_segment(
            candidate_n.get("segment_point_counts", {}),
            pymoo.get("segment_point_counts", {}),
        )
        nondominated_gap_segment = _largest_gap_segment(
            candidate_n.get("segment_nondominated_counts", {}),
            pymoo.get("segment_nondominated_counts", {}),
        )
        if metric == "spacing":
            gap_segment = candidate_n.get("weakest_segment_id")
        elif metric == "total_nondominated_count":
            gap_segment = nondominated_gap_segment
        else:
            gap_segment = point_gap_segment

        if candidate_n_value is None:
            interpretation = "candidate_n value unavailable"
        elif metric in _LOWER_IS_BETTER_METRICS:
            if candidate_j_value is not None and candidate_n_value < candidate_j_value:
                interpretation = "candidate_n improved versus candidate_j on a lower-is-better spread metric"
            elif candidate_j_value is not None and math.isclose(
                candidate_n_value,
                candidate_j_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                interpretation = "candidate_n stayed effectively tied with candidate_j"
            else:
                interpretation = "candidate_n did not improve over candidate_j on this lower-is-better spread metric"
            if pymoo_value is not None and candidate_n_value > pymoo_value:
                interpretation += "; pymoo remains stronger"
        elif metric in _HIGHER_IS_BETTER_METRICS:
            if candidate_j_value is not None and candidate_n_value > candidate_j_value:
                interpretation = "candidate_n improved versus candidate_j on a higher-is-better spread metric"
            elif candidate_j_value is not None and math.isclose(
                candidate_n_value,
                candidate_j_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                interpretation = "candidate_n stayed effectively tied with candidate_j"
            else:
                interpretation = "candidate_n did not improve over candidate_j on this higher-is-better spread metric"
            if pymoo_value is not None and candidate_n_value < pymoo_value:
                interpretation += "; pymoo remains stronger"
        else:
            interpretation = "metric direction undefined"

        rows.append(
            {
                "metric": metric,
                "candidate_j": candidate_j_value,
                "candidate_n": candidate_n_value,
                "pymoo": pymoo_value,
                "candidate_n_vs_j_delta": delta_vs_j,
                "candidate_n_vs_pymoo_gap": gap_vs_pymoo,
                "gap_segment": gap_segment,
                "interpretation": interpretation,
            }
        )
    return rows
