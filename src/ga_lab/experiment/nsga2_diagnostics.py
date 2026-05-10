from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Mapping, Sequence

from ga_lab.experiment.diversity_diagnostics import unique_vector_count
from ga_lab.experiment.mo_metrics import nondominated_vectors, spacing_metric
from ga_lab.problems.zdt1 import ZDT1Problem


_ZDT1_COMPONENT_PROBLEM = ZDT1Problem("zdt1")


def _signature(
    vector: Sequence[float | int],
    *,
    precision: int = 12,
) -> tuple[float, ...]:
    return tuple(round(float(value), precision) for value in vector)


def _finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        return None
    value_float = float(value)
    if not math.isfinite(value_float):
        return None
    return value_float


def _safe_mean(values: Sequence[float | int | None]) -> float | None:
    cleaned = [_finite_or_none(value) for value in values]
    finite_values = [value for value in cleaned if value is not None]
    if not finite_values:
        return None
    return float(mean(finite_values))


def _safe_median(values: Sequence[float | int | None]) -> float | None:
    cleaned = [_finite_or_none(value) for value in values]
    finite_values = [value for value in cleaned if value is not None]
    if not finite_values:
        return None
    return float(median(finite_values))


def _safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(right - left)


def _safe_std(values: Sequence[float | int]) -> float | None:
    numeric = [float(value) for value in values]
    if not numeric:
        return None
    mean_value = float(mean(numeric))
    variance = sum((value - mean_value) ** 2 for value in numeric) / len(numeric)
    return float(math.sqrt(variance))


def _positive_int(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def _to_minimization(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> list[list[float]]:
    return [
        [
            -float(value) if directions[index] else float(value)
            for index, value in enumerate(vector)
        ]
        for vector in objective_vectors
    ]


def _boundary_signature_details(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> tuple[set[tuple[float, ...]], list[dict[str, Any]]]:
    if not objective_vectors:
        return set(), []

    normalized = _to_minimization(objective_vectors, directions)
    objective_count = len(normalized[0])
    boundary_signatures: set[tuple[float, ...]] = set()
    objective_rows: list[dict[str, Any]] = []

    for objective_index in range(objective_count):
        column = [vector[objective_index] for vector in normalized]
        min_value = min(column)
        max_value = max(column)
        min_signatures: set[tuple[float, ...]] = set()
        max_signatures: set[tuple[float, ...]] = set()
        for raw_vector, normalized_vector in zip(objective_vectors, normalized, strict=True):
            signature = _signature(raw_vector)
            value = normalized_vector[objective_index]
            if math.isclose(value, min_value, rel_tol=1e-12, abs_tol=1e-12):
                min_signatures.add(signature)
                boundary_signatures.add(signature)
            if math.isclose(value, max_value, rel_tol=1e-12, abs_tol=1e-12):
                max_signatures.add(signature)
                boundary_signatures.add(signature)
        objective_rows.append(
            {
                "objective": objective_index,
                "min_boundary_count": len(min_signatures),
                "max_boundary_count": len(max_signatures),
                "min_signatures": sorted(min_signatures),
                "max_signatures": sorted(max_signatures),
            }
        )

    return boundary_signatures, objective_rows


def _occupancy_snapshot(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int,
) -> dict[str, Any]:
    if not objective_vectors:
        return {
            "used_dimensions": 0,
            "total_bins": 0,
            "occupied_bins": 0,
            "empty_bins": 0,
            "max_bin_occupancy": 0,
            "sparse_bin_count": 0,
            "occupancy_entropy": None,
            "point_bins": [],
            "bin_counts": {},
            "sparse_threshold": 1,
            "warnings": [],
        }

    warnings: list[str] = []
    normalized = _to_minimization(objective_vectors, directions)
    objective_count = len(normalized[0])
    used_dimensions = min(objective_count, 2)
    if objective_count > used_dimensions:
        warnings.append(
            f"occupancy projected to first {used_dimensions} objectives from {objective_count}"
        )

    minima = [min(vector[index] for vector in normalized) for index in range(used_dimensions)]
    maxima = [max(vector[index] for vector in normalized) for index in range(used_dimensions)]
    point_bins: list[tuple[int, ...]] = []
    for vector in normalized:
        bin_key: list[int] = []
        for index in range(used_dimensions):
            low = minima[index]
            high = maxima[index]
            scale = high - low
            if math.isclose(scale, 0.0, abs_tol=1e-12):
                bin_key.append(0)
                continue
            position = (vector[index] - low) / scale
            if position >= 1.0:
                bin_key.append(bins - 1)
            else:
                bin_key.append(max(0, min(bins - 1, int(position * bins))))
        point_bins.append(tuple(bin_key))

    counts = Counter(point_bins)
    occupied_bins = len(counts)
    total_bins = bins**used_dimensions
    probabilities = [count / len(point_bins) for count in counts.values()]
    occupancy_entropy = None
    if probabilities:
        entropy = -sum(probability * math.log(probability, 2) for probability in probabilities)
        occupancy_entropy = float(entropy)
    sparse_threshold = max(1, math.floor(len(point_bins) / max(1, occupied_bins)))

    return {
        "used_dimensions": used_dimensions,
        "total_bins": total_bins,
        "occupied_bins": occupied_bins,
        "empty_bins": max(0, total_bins - occupied_bins),
        "max_bin_occupancy": max(counts.values(), default=0),
        "sparse_bin_count": sum(1 for count in counts.values() if count <= sparse_threshold),
        "occupancy_entropy": occupancy_entropy,
        "point_bins": point_bins,
        "bin_counts": {str(key): int(value) for key, value in counts.items()},
        "sparse_threshold": sparse_threshold,
        "warnings": warnings,
    }


def summarize_parent_contributions(
    parent_events: Sequence[Mapping[str, Any]],
    *,
    population_size: int,
    top_parent_limit: int = 5,
) -> dict[str, Any]:
    if not parent_events:
        return {
            "selection_event_count": 0,
            "unique_parent_count": 0,
            "parent_selection_diversity_ratio": 0.0,
            "boundary_parent_selection_rate": 0.0,
            "sparse_parent_selection_rate": 0.0,
            "bias_trigger_rate": 0.0,
            "mean_parent_rank": None,
            "mean_parent_crowding": None,
            "inf_crowding_selection_count": 0,
            "sample_same_rank_frequency": 0.0,
            "sample_crowding_tie_frequency": 0.0,
            "same_parent_repeat_rate": 0.0,
            "selection_kind_counts": {},
            "top_selected_parents": [],
            "warnings": [],
        }

    parent_counts = Counter(int(event["winner_index"]) for event in parent_events)
    boundary_count = sum(1 for event in parent_events if bool(event.get("is_boundary")))
    sparse_count = sum(1 for event in parent_events if bool(event.get("is_sparse")))
    bias_count = sum(1 for event in parent_events if bool(event.get("bias_applied")))
    ranks = [
        int(event["winner_rank"])
        for event in parent_events
        if isinstance(event.get("winner_rank"), int)
    ]
    crowding_values = [_finite_or_none(event.get("winner_crowding")) for event in parent_events]
    finite_crowding = [value for value in crowding_values if value is not None]

    consecutive_repeats = 0
    previous_winner: int | None = None
    for event in parent_events:
        winner_index = int(event["winner_index"])
        if previous_winner is not None and winner_index == previous_winner:
            consecutive_repeats += 1
        previous_winner = winner_index

    top_selected_parents: list[dict[str, Any]] = []
    for winner_index, count in parent_counts.most_common(max(1, top_parent_limit)):
        winner_events = [
            event for event in parent_events if int(event["winner_index"]) == int(winner_index)
        ]
        top_selected_parents.append(
            {
                "index": int(winner_index),
                "selection_count": int(count),
                "selection_rate": count / len(parent_events),
                "mean_rank": _safe_mean([event.get("winner_rank") for event in winner_events]),
                "mean_crowding": _safe_mean(
                    [event.get("winner_crowding") for event in winner_events]
                ),
                "boundary_count": sum(
                    1 for event in winner_events if bool(event.get("is_boundary"))
                ),
                "sparse_count": sum(
                    1 for event in winner_events if bool(event.get("is_sparse"))
                ),
            }
        )

    return {
        "selection_event_count": len(parent_events),
        "unique_parent_count": len(parent_counts),
        "parent_selection_diversity_ratio": (
            len(parent_counts) / population_size if population_size > 0 else 0.0
        ),
        "boundary_parent_selection_rate": boundary_count / len(parent_events),
        "sparse_parent_selection_rate": sparse_count / len(parent_events),
        "bias_trigger_rate": bias_count / len(parent_events),
        "mean_parent_rank": _safe_mean(ranks),
        "mean_parent_crowding": _safe_mean(finite_crowding),
        "inf_crowding_selection_count": sum(
            1
            for event in parent_events
            if isinstance(event.get("winner_crowding"), int | float)
            and math.isinf(float(event["winner_crowding"]))
        ),
        "sample_same_rank_frequency": sum(
            1 for event in parent_events if bool(event.get("sample_same_rank"))
        )
        / len(parent_events),
        "sample_crowding_tie_frequency": sum(
            1 for event in parent_events if bool(event.get("sample_crowding_tie"))
        )
        / len(parent_events),
        "same_parent_repeat_rate": (
            consecutive_repeats / max(1, len(parent_events) - 1)
            if len(parent_events) > 1
            else 0.0
        ),
        "selection_kind_counts": {
            str(key): int(value)
            for key, value in Counter(
                str(event.get("selection_kind", "unknown")) for event in parent_events
            ).items()
        },
        "top_selected_parents": top_selected_parents,
        "warnings": [],
    }


def summarize_survivor_replacement(
    *,
    population_size: int,
    combined_fronts: Sequence[Sequence[int]],
    combined_ranks: Sequence[int],
    combined_crowding: Sequence[float],
    combined_objective_vectors: Sequence[Sequence[float]],
    survivor_indices: Sequence[int],
    directions: Sequence[bool],
    partial_front_strategy: str,
    partial_front_dedup_mode: str,
    bins: int = 6,
) -> dict[str, Any]:
    selected_set = {int(index) for index in survivor_indices}
    full_fronts_kept = 0
    selected_before_partial = 0
    truncated_front: list[int] = []
    truncated_front_rank: int | None = None

    for front in combined_fronts:
        if selected_before_partial + len(front) <= population_size:
            full_fronts_kept += 1
            selected_before_partial += len(front)
            continue
        truncated_front = [int(index) for index in front]
        if truncated_front:
            truncated_front_rank = int(combined_ranks[truncated_front[0]])
        break

    warnings: list[str] = []
    if not truncated_front:
        warnings.append("no_partial_front_truncation")

    front_vectors = [combined_objective_vectors[index] for index in truncated_front]
    boundary_signatures, _ = _boundary_signature_details(front_vectors, directions)
    occupancy = _occupancy_snapshot(front_vectors, directions, bins=bins)
    point_bins = [tuple(value) for value in occupancy.get("point_bins", [])]
    bin_counts = Counter(point_bins)
    sparse_threshold = int(occupancy.get("sparse_threshold", 1))

    boundary_selected = 0
    boundary_discarded = 0
    sparse_selected = 0
    sparse_discarded = 0
    selected_from_truncated = 0
    discarded_from_truncated = 0
    front_signatures = [_signature(vector) for vector in front_vectors]
    for position, index in enumerate(truncated_front):
        signature = front_signatures[position] if position < len(front_signatures) else None
        bin_key = point_bins[position] if position < len(point_bins) else None
        is_boundary = signature in boundary_signatures if signature is not None else False
        is_sparse = (
            bin_key is not None and bin_counts.get(bin_key, 0) <= sparse_threshold
        )
        if index in selected_set:
            selected_from_truncated += 1
            if is_boundary:
                boundary_selected += 1
            if is_sparse:
                sparse_selected += 1
        else:
            discarded_from_truncated += 1
            if is_boundary:
                boundary_discarded += 1
            if is_sparse:
                sparse_discarded += 1

    finite_crowding = [
        round(float(combined_crowding[index]), 12)
        for index in truncated_front
        if isinstance(combined_crowding[index], int | float)
        and math.isfinite(float(combined_crowding[index]))
    ]
    crowding_counter = Counter(finite_crowding)
    same_rank_tie_frequency = (
        sum(count - 1 for count in crowding_counter.values() if count > 1)
        / max(1, len(finite_crowding))
        if finite_crowding
        else 0.0
    )
    boundary_total = boundary_selected + boundary_discarded
    sparse_total = sparse_selected + sparse_discarded

    return {
        "combined_population_size": sum(len(front) for front in combined_fronts),
        "survivor_population_size": len(survivor_indices),
        "full_fronts_kept": full_fronts_kept,
        "truncated_front_rank": truncated_front_rank,
        "truncated_front_size": len(truncated_front),
        "selected_before_partial": selected_before_partial,
        "selected_from_truncated_front": selected_from_truncated,
        "discarded_from_truncated_front": discarded_from_truncated,
        "boundary_selected_from_truncated_front": boundary_selected,
        "boundary_discarded_from_truncated_front": boundary_discarded,
        "sparse_selected_from_truncated_front": sparse_selected,
        "sparse_discarded_from_truncated_front": sparse_discarded,
        "boundary_selected_rate": (
            boundary_selected / boundary_total if boundary_total > 0 else None
        ),
        "sparse_selected_rate": sparse_selected / sparse_total if sparse_total > 0 else None,
        "same_rank_tie_frequency": same_rank_tie_frequency,
        "partial_front_strategy": partial_front_strategy,
        "partial_front_dedup_mode": partial_front_dedup_mode,
        "warnings": warnings + list(occupancy.get("warnings", [])),
    }


def summarize_boundary_retention(
    previous_front_vectors: Sequence[Sequence[float]],
    current_front_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> dict[str, Any]:
    previous_boundary_signatures, previous_boundary_rows = _boundary_signature_details(
        previous_front_vectors,
        directions,
    )
    current_signatures = {_signature(vector) for vector in current_front_vectors}
    retained = previous_boundary_signatures & current_signatures

    per_objective_rows: list[dict[str, Any]] = []
    for row in previous_boundary_rows:
        min_signatures = {tuple(value) for value in row["min_signatures"]}
        max_signatures = {tuple(value) for value in row["max_signatures"]}
        per_objective_rows.append(
            {
                "objective": int(row["objective"]),
                "min_boundary_total": int(row["min_boundary_count"]),
                "min_boundary_retained": len(min_signatures & current_signatures),
                "max_boundary_total": int(row["max_boundary_count"]),
                "max_boundary_retained": len(max_signatures & current_signatures),
            }
        )

    previous_boundary_count = len(previous_boundary_signatures)
    retained_count = len(retained)
    return {
        "previous_boundary_count": previous_boundary_count,
        "retained_boundary_count": retained_count,
        "boundary_loss_count": max(0, previous_boundary_count - retained_count),
        "boundary_retention_rate": (
            retained_count / previous_boundary_count if previous_boundary_count > 0 else None
        ),
        "per_objective_retention": per_objective_rows,
        "warnings": [],
    }


def summarize_rank_crowding(
    ranks: Sequence[int],
    crowding: Sequence[float],
) -> dict[str, Any]:
    if not ranks or not crowding:
        return {
            "rank_counts": {},
            "crowding_finite_mean": None,
            "crowding_finite_median": None,
            "crowding_finite_min": None,
            "crowding_finite_max": None,
            "inf_crowding_count": 0,
            "same_rank_tie_frequency": 0.0,
            "warnings": [],
        }

    finite_crowding = [
        float(value) for value in crowding if isinstance(value, int | float) and math.isfinite(float(value))
    ]
    grouped: dict[int, list[float]] = defaultdict(list)
    for rank, crowding_value in zip(ranks, crowding, strict=True):
        if isinstance(crowding_value, int | float) and math.isfinite(float(crowding_value)):
            grouped[int(rank)].append(round(float(crowding_value), 12))
    duplicate_count = 0
    comparable_count = 0
    for values in grouped.values():
        comparable_count += len(values)
        counter = Counter(values)
        duplicate_count += sum(count - 1 for count in counter.values() if count > 1)

    return {
        "rank_counts": {
            str(rank): int(count) for rank, count in Counter(int(value) for value in ranks).items()
        },
        "crowding_finite_mean": _safe_mean(finite_crowding),
        "crowding_finite_median": _safe_median(finite_crowding),
        "crowding_finite_min": min(finite_crowding) if finite_crowding else None,
        "crowding_finite_max": max(finite_crowding) if finite_crowding else None,
        "inf_crowding_count": sum(
            1
            for value in crowding
            if isinstance(value, int | float) and math.isinf(float(value))
        ),
        "same_rank_tie_frequency": (
            duplicate_count / comparable_count if comparable_count > 0 else 0.0
        ),
        "warnings": [],
    }


def summarize_objective_occupancy(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    occupancy = _occupancy_snapshot(objective_vectors, directions, bins=bins)
    return {
        "objective_count": len(objective_vectors[0]) if objective_vectors else 0,
        "projected_dimensions": occupancy["used_dimensions"],
        "occupied_bins": occupancy["occupied_bins"],
        "total_bins": occupancy["total_bins"],
        "empty_bins": occupancy["empty_bins"],
        "sparse_bin_count": occupancy["sparse_bin_count"],
        "max_bin_occupancy": occupancy["max_bin_occupancy"],
        "occupancy_entropy": occupancy["occupancy_entropy"],
        "unique_objective_count": unique_vector_count(objective_vectors),
        "warnings": list(occupancy.get("warnings", [])),
    }


def summarize_front_change(
    previous_front_vectors: Sequence[Sequence[float]],
    current_front_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> dict[str, Any]:
    previous_signatures = {_signature(vector) for vector in previous_front_vectors}
    current_signatures = {_signature(vector) for vector in current_front_vectors}
    overlap = previous_signatures & current_signatures
    previous_spacing = _finite_or_none(spacing_metric(previous_front_vectors, directions))
    current_spacing = _finite_or_none(spacing_metric(current_front_vectors, directions))
    warnings: list[str] = []
    if previous_spacing is None:
        warnings.append("previous_front_spacing_undefined")
    if current_spacing is None:
        warnings.append("current_front_spacing_undefined")

    return {
        "previous_front_size": len(previous_front_vectors),
        "current_front_size": len(current_front_vectors),
        "front_overlap_count": len(overlap),
        "front_overlap_rate": (
            len(overlap) / len(previous_signatures) if previous_signatures else None
        ),
        "unique_objective_count_previous": unique_vector_count(previous_front_vectors),
        "unique_objective_count_current": unique_vector_count(current_front_vectors),
        "unique_objective_count_delta": (
            unique_vector_count(current_front_vectors) - unique_vector_count(previous_front_vectors)
        ),
        "nondominated_count_previous": len(previous_front_vectors),
        "nondominated_count_current": len(current_front_vectors),
        "nondominated_count_delta": len(current_front_vectors) - len(previous_front_vectors),
        "previous_spacing": previous_spacing,
        "current_spacing": current_spacing,
        "spacing_delta": _safe_delta(previous_spacing, current_spacing),
        "warnings": warnings,
    }


def hash_solution_or_objective(
    vector: Sequence[float | int],
    *,
    precision: int = 12,
) -> str:
    return "|".join(f"{value:.{precision}f}" for value in _signature(vector, precision=precision))


def compute_objective_bins(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    return _occupancy_snapshot(objective_vectors, directions, bins=bins)


def compute_objective_segment(
    objective_vector: Sequence[float],
    reference_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> int | None:
    if not objective_vector or not reference_vectors:
        return None
    snapshot = compute_objective_bins(reference_vectors, directions, bins=bins)
    point_bins = [
        tuple(int(value) for value in bin_key)
        for bin_key in snapshot.get("point_bins", [])
        if isinstance(bin_key, (list, tuple))
    ]
    signatures = [_signature(vector) for vector in reference_vectors]
    target_signature = _signature(objective_vector)
    for signature, bin_key in zip(signatures, point_bins, strict=False):
        if signature == target_signature and bin_key:
            return int(bin_key[0])
    return None


def compute_boundary_adjacent_flag(
    bin_key: Sequence[int] | None,
    *,
    bins: int = 6,
) -> bool:
    if bin_key is None:
        return False
    values = [int(value) for value in bin_key]
    if not values:
        return False
    upper = max(0, bins - 1)
    return any(value in {0, upper} for value in values)


def compute_segment_distribution(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    occupancy = compute_objective_bins(objective_vectors, directions, bins=bins)
    point_bins = [
        tuple(int(value) for value in bin_key)
        for bin_key in occupancy.get("point_bins", [])
        if isinstance(bin_key, (list, tuple))
    ]
    point_segments = [int(bin_key[0]) for bin_key in point_bins if bin_key]
    segment_counts = Counter(point_segments)
    total_points = len(point_segments)
    boundary_adjacent_count = sum(
        1 for bin_key in point_bins if compute_boundary_adjacent_flag(bin_key, bins=bins)
    )

    normalized = _to_minimization(objective_vectors, directions) if objective_vectors else []
    segment_ranges: list[list[float]] = []
    warnings = list(occupancy.get("warnings", []))
    if normalized:
        x_values = [float(vector[0]) for vector in normalized]
        x_min = min(x_values)
        x_max = max(x_values)
        x_scale = x_max - x_min
        for segment_index in range(max(1, bins)):
            if math.isclose(x_scale, 0.0, abs_tol=1e-12):
                segment_ranges.append([float(x_min), float(x_max)])
                continue
            low = x_min + (x_scale * segment_index / bins)
            high = x_min + (x_scale * (segment_index + 1) / bins)
            segment_ranges.append([float(low), float(high)])
    else:
        segment_ranges = [[0.0, 1.0] for _ in range(max(1, bins))]

    return {
        "segment_count": bins,
        "segment_counts": {str(key): int(value) for key, value in sorted(segment_counts.items())},
        "segment_rates": {
            str(key): (float(value) / total_points if total_points > 0 else 0.0)
            for key, value in sorted(segment_counts.items())
        },
        "occupied_segments": len(segment_counts),
        "segment_ranges": segment_ranges,
        "point_bins": [list(bin_key) for bin_key in point_bins],
        "point_segments": point_segments,
        "boundary_adjacent_count": boundary_adjacent_count,
        "warnings": warnings,
    }


def compare_segment_distributions(
    candidate_distribution: Mapping[str, Any],
    reference_distribution: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_counts = {
        int(key): int(value)
        for key, value in dict(candidate_distribution.get("segment_counts", {})).items()
    }
    reference_counts = {
        int(key): int(value)
        for key, value in dict(reference_distribution.get("segment_counts", {})).items()
    }
    all_segments = sorted(set(candidate_counts) | set(reference_counts))
    return {
        "segment_occupancy_diff": {
            str(segment): int(candidate_counts.get(segment, 0) - reference_counts.get(segment, 0))
            for segment in all_segments
        },
        "segment0_count_diff": int(candidate_counts.get(0, 0) - reference_counts.get(0, 0)),
        "boundary_count_diff": int(
            int(candidate_distribution.get("boundary_adjacent_count", 0))
            - int(reference_distribution.get("boundary_adjacent_count", 0))
        ),
        "occupied_segments_diff": int(
            int(candidate_distribution.get("occupied_segments", 0))
            - int(reference_distribution.get("occupied_segments", 0))
        ),
    }


def compute_spacing_by_segment(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    if not objective_vectors:
        return {
            "segment_count": bins,
            "empty_segment_count": bins,
            "global_mean_gap": None,
            "weak_segment_id": None,
            "max_gap_segment_id": None,
            "segment_rows": [],
            "front_objective_hashes": [],
            "front_objective_vectors": [],
            "warnings": ["empty_front"],
        }

    warnings: list[str] = []
    normalized = _to_minimization(objective_vectors, directions)
    objective_count = len(normalized[0])
    used_dimensions = min(objective_count, 2)
    if objective_count > used_dimensions:
        warnings.append(
            f"segment spacing projected to first {used_dimensions} objectives from {objective_count}"
        )
    if len(objective_vectors) <= 1:
        warnings.append("segment_spacing_singleton_or_empty_front")
        return {
            "segment_count": bins,
            "empty_segment_count": max(0, bins - len(objective_vectors)),
            "global_mean_gap": None,
            "weak_segment_id": None,
            "max_gap_segment_id": None,
            "segment_rows": [
                {
                    "segment_id": 0,
                    "segment_range": [0.0, 1.0],
                    "points_in_segment": len(objective_vectors),
                    "mean_gap": None,
                    "max_gap": None,
                    "local_spacing_contribution": None,
                    "empty_segment": len(objective_vectors) == 0,
                }
            ],
            "front_objective_hashes": [
                hash_solution_or_objective(vector) for vector in objective_vectors
            ],
            "front_objective_vectors": [list(vector) for vector in objective_vectors],
            "warnings": warnings,
        }

    ordered = sorted(
        range(len(normalized)),
        key=lambda idx: tuple(normalized[idx][:used_dimensions]),
    )
    ordered_vectors = [normalized[idx][:used_dimensions] for idx in ordered]
    x_values = [vector[0] for vector in ordered_vectors]
    x_min = min(x_values)
    x_max = max(x_values)
    x_scale = x_max - x_min

    def _segment_for_x(x_value: float) -> int:
        if math.isclose(x_scale, 0.0, abs_tol=1e-12):
            return 0
        position = (x_value - x_min) / x_scale
        if position >= 1.0:
            return bins - 1
        return max(0, min(bins - 1, int(position * bins)))

    segment_point_counts: Counter[int] = Counter()
    gap_rows: list[dict[str, Any]] = []
    for vector in ordered_vectors:
        segment_point_counts[_segment_for_x(vector[0])] += 1

    gaps: list[float] = []
    for left_vector, right_vector in zip(ordered_vectors, ordered_vectors[1:], strict=False):
        if used_dimensions == 1:
            gap = abs(right_vector[0] - left_vector[0])
        else:
            gap = math.dist(left_vector, right_vector)
        midpoint = (left_vector[0] + right_vector[0]) / 2.0
        segment_id = _segment_for_x(midpoint)
        gaps.append(gap)
        gap_rows.append({"segment_id": segment_id, "gap": gap})

    global_mean_gap = _safe_mean(gaps)
    segment_gaps: dict[int, list[float]] = defaultdict(list)
    for row in gap_rows:
        segment_gaps[int(row["segment_id"])].append(float(row["gap"]))

    segment_rows: list[dict[str, Any]] = []
    weakest_segment_id: int | None = None
    weakest_score = -1.0
    max_gap_segment_id: int | None = None
    max_gap_value = -1.0
    empty_segment_count = 0

    for segment_id in range(max(1, bins)):
        gaps_for_segment = segment_gaps.get(segment_id, [])
        mean_gap = _safe_mean(gaps_for_segment)
        max_gap = max(gaps_for_segment) if gaps_for_segment else None
        local_contribution = None
        if gaps_for_segment and global_mean_gap is not None:
            local_contribution = float(
                mean(abs(gap - global_mean_gap) for gap in gaps_for_segment)
            )
        points_in_segment = int(segment_point_counts.get(segment_id, 0))
        empty_segment = points_in_segment == 0
        if empty_segment:
            empty_segment_count += 1
        if local_contribution is not None and local_contribution > weakest_score:
            weakest_score = local_contribution
            weakest_segment_id = segment_id
        if isinstance(max_gap, int | float) and float(max_gap) > max_gap_value:
            max_gap_value = float(max_gap)
            max_gap_segment_id = segment_id
        segment_rows.append(
            {
                "segment_id": segment_id,
                "segment_range": [segment_id / bins, (segment_id + 1) / bins],
                "points_in_segment": points_in_segment,
                "mean_gap": mean_gap,
                "max_gap": float(max_gap) if isinstance(max_gap, int | float) else None,
                "local_spacing_contribution": local_contribution,
                "empty_segment": empty_segment,
            }
        )

    return {
        "segment_count": bins,
        "empty_segment_count": empty_segment_count,
        "global_mean_gap": global_mean_gap,
        "weak_segment_id": weakest_segment_id,
        "max_gap_segment_id": max_gap_segment_id,
        "segment_rows": segment_rows,
        "front_objective_hashes": [
            hash_solution_or_objective(objective_vectors[idx]) for idx in ordered
        ],
        "front_objective_vectors": [list(objective_vectors[idx]) for idx in ordered],
        "warnings": warnings,
    }


def summarize_decision_component_summary(
    genome: Sequence[float | int],
) -> dict[str, Any]:
    if not genome:
        return {
            "x0": None,
            "tail_mean": None,
            "tail_min": None,
            "tail_max": None,
            "tail_std": None,
        }

    numeric = [float(value) for value in genome]
    tail = numeric[1:]
    if not tail:
        return {
            "x0": float(numeric[0]),
            "tail_mean": None,
            "tail_min": None,
            "tail_max": None,
            "tail_std": None,
        }
    return {
        "x0": float(numeric[0]),
        "tail_mean": float(sum(tail) / len(tail)),
        "tail_min": float(min(tail)),
        "tail_max": float(max(tail)),
        "tail_std": _safe_std(tail),
    }


def _zdt1_segment_id(f1: float | None, *, bins: int) -> int | None:
    if f1 is None:
        return None
    clamped = min(1.0, max(0.0, float(f1)))
    if clamped >= 1.0:
        return max(0, bins - 1)
    return max(0, min(bins - 1, int(clamped * bins)))


def _zdt1_segment_range(segment_id: int | None, *, bins: int) -> list[float] | None:
    if segment_id is None:
        return None
    lower = float(segment_id) / max(1, bins)
    upper = float(segment_id + 1) / max(1, bins)
    if segment_id >= max(0, bins - 1):
        upper = 1.0
    return [lower, upper]


def _zdt1_components_from_summary(
    summary: Mapping[str, Any],
    *,
    objective_vector: Sequence[float] | None = None,
    bins: int = 6,
    nondominated: bool | None = None,
    survived: bool | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    x0_raw = _finite_or_none(summary.get("x0"))
    tail_mean = _finite_or_none(summary.get("tail_mean"))
    tail_min = _finite_or_none(summary.get("tail_min"))
    tail_max = _finite_or_none(summary.get("tail_max"))
    tail_std = _finite_or_none(summary.get("tail_std"))

    if x0_raw is None or tail_mean is None:
        warnings.append("missing_zdt1_decision_summary")

    f1 = None if x0_raw is None else min(1.0, max(0.0, float(x0_raw)))
    g = None if tail_mean is None else float(1.0 + (9.0 * tail_mean))
    f2 = None
    if (
        isinstance(objective_vector, Sequence)
        and len(objective_vector) >= 2
        and all(isinstance(value, int | float) for value in objective_vector[:2])
    ):
        f1 = min(1.0, max(0.0, float(objective_vector[0])))
        f2 = float(objective_vector[1])
    elif f1 is not None and g is not None:
        f2 = float(_ZDT1_COMPONENT_PROBLEM.fitness([f1, tail_mean])[1])

    if g is not None and f2 is not None and not math.isclose(g, 0.0, abs_tol=1e-12):
        h = float(f2 / g)
    elif f1 is not None and g is not None and not math.isclose(g, 0.0, abs_tol=1e-12):
        ratio = max(0.0, f1 / g)
        h = float(1.0 - math.sqrt(ratio))
    else:
        h = None

    reference_front_f2 = (
        None if f1 is None else float(1.0 - math.sqrt(max(0.0, min(1.0, f1))))
    )
    distance = (
        None
        if f2 is None or reference_front_f2 is None
        else float(abs(f2 - reference_front_f2))
    )
    segment_id = _zdt1_segment_id(f1, bins=max(1, bins))

    return {
        "x0": x0_raw,
        "tail_mean": tail_mean,
        "tail_min": tail_min,
        "tail_max": tail_max,
        "tail_std": tail_std,
        "g": g,
        "h": h,
        "f1": f1,
        "f2": f2,
        "distance_to_reference_front": distance,
        "distance_to_zdt1_front": distance,
        "segment_id": segment_id,
        "segment_range": _zdt1_segment_range(segment_id, bins=max(1, bins)),
        "segment0_flag": segment_id == 0 if segment_id is not None else False,
        "boundary_adjacent_flag": (
            segment_id in {0, max(0, bins - 1)} if segment_id is not None else False
        ),
        "nondominated_flag": None if nondominated is None else bool(nondominated),
        "survived_flag": None if survived is None else bool(survived),
        "warnings": warnings,
    }


def compute_zdt1_components(
    genome: Sequence[float | int],
    objective_vector: Sequence[float] | None = None,
    *,
    bins: int = 6,
    nondominated: bool | None = None,
    survived: bool | None = None,
) -> dict[str, Any]:
    summary = summarize_decision_component_summary(genome)
    return _zdt1_components_from_summary(
        summary,
        objective_vector=objective_vector,
        bins=bins,
        nondominated=nondominated,
        survived=survived,
    )


def _collect_zdt1_population_components(
    population: Sequence[Sequence[float | int]],
    objective_vectors: Sequence[Sequence[float]],
    *,
    bins: int,
    nondominated_signatures: set[tuple[float, ...]] | None = None,
    survived_signatures: set[tuple[float, ...]] | None = None,
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for genome, objective_vector in zip(population, objective_vectors, strict=False):
        objective_signature = _signature(objective_vector) if objective_vector else None
        components.append(
            compute_zdt1_components(
                genome,
                objective_vector,
                bins=bins,
                nondominated=(
                    objective_signature in nondominated_signatures
                    if objective_signature is not None and nondominated_signatures is not None
                    else None
                ),
                survived=(
                    objective_signature in survived_signatures
                    if objective_signature is not None and survived_signatures is not None
                    else None
                ),
            )
        )
    return components


def summarize_zdt1_initial_component_coverage(
    population: Sequence[Sequence[float | int]],
    objective_vectors: Sequence[Sequence[float]],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    components = _collect_zdt1_population_components(
        population,
        objective_vectors,
        bins=bins,
    )
    segment0 = [component for component in components if bool(component.get("segment0_flag"))]
    occupancy = compute_objective_bins(objective_vectors, [False, False], bins=bins)
    return {
        "segment_count": bins,
        "segment0_range": [0.0, 1.0 / max(1, bins)],
        "x0_min": _safe_mean([min([component["x0"] for component in components if component.get("x0") is not None])] if any(component.get("x0") is not None for component in components) else []),
        "x0_mean": _safe_mean([component.get("x0") for component in components]),
        "x0_max": _safe_mean([max([component["x0"] for component in components if component.get("x0") is not None])] if any(component.get("x0") is not None for component in components) else []),
        "tail_mean_mean": _safe_mean([component.get("tail_mean") for component in components]),
        "g_min": _safe_mean([min([component["g"] for component in components if component.get("g") is not None])] if any(component.get("g") is not None for component in components) else []),
        "g_mean": _safe_mean([component.get("g") for component in components]),
        "g_max": _safe_mean([max([component["g"] for component in components if component.get("g") is not None])] if any(component.get("g") is not None for component in components) else []),
        "segment0_count": len(segment0),
        "segment0_g_mean": _safe_mean([component.get("g") for component in segment0]),
        "segment0_distance_mean": _safe_mean(
            [component.get("distance_to_zdt1_front") for component in segment0]
        ),
        "occupied_bins": int(occupancy.get("occupied_bins", 0)),
        "unique_objective_count": unique_vector_count(objective_vectors),
        "warnings": list(occupancy.get("warnings", [])),
    }


def summarize_zdt1_offspring_component_quality(
    lineage_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not lineage_records:
        return {
            "offspring_count": 0,
            "segment0_offspring_count": 0,
            "segment0_x0_mean": None,
            "segment0_g_mean": None,
            "segment0_f2_mean": None,
            "segment0_distance_mean": None,
            "segment0_nondominated_rate": None,
            "segment0_survival_rate": None,
            "non_segment0_g_mean": None,
            "non_segment0_distance_mean": None,
            "warnings": ["no_lineage_records"],
        }

    components = [
        dict(record.get("offspring_zdt1_components") or {})
        for record in lineage_records
        if isinstance(record.get("offspring_zdt1_components"), dict)
    ]
    segment0 = [component for component in components if bool(component.get("segment0_flag"))]
    non_segment0 = [component for component in components if not bool(component.get("segment0_flag"))]
    warnings = [
        warning
        for component in components
        for warning in list(component.get("warnings", []))
    ]
    return {
        "offspring_count": len(components),
        "segment0_offspring_count": len(segment0),
        "segment0_x0_mean": _safe_mean([component.get("x0") for component in segment0]),
        "segment0_g_mean": _safe_mean([component.get("g") for component in segment0]),
        "segment0_f2_mean": _safe_mean([component.get("f2") for component in segment0]),
        "segment0_distance_mean": _safe_mean(
            [component.get("distance_to_zdt1_front") for component in segment0]
        ),
        "segment0_nondominated_rate": (
            sum(1 for component in segment0 if bool(component.get("nondominated_flag")))
            / len(segment0)
            if segment0
            else None
        ),
        "segment0_survival_rate": (
            sum(1 for component in segment0 if bool(component.get("survived_flag"))) / len(segment0)
            if segment0
            else None
        ),
        "non_segment0_g_mean": _safe_mean([component.get("g") for component in non_segment0]),
        "non_segment0_distance_mean": _safe_mean(
            [component.get("distance_to_zdt1_front") for component in non_segment0]
        ),
        "warnings": sorted(set(warnings)),
    }


def summarize_zdt1_parent_child_component_delta(
    lineage_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not lineage_records:
        return {
            "parent_x0_mean": None,
            "offspring_x0_mean": None,
            "delta_x0_mean": None,
            "parent_g_mean": None,
            "offspring_g_mean": None,
            "delta_g_mean": None,
            "parent_distance_mean": None,
            "offspring_distance_mean": None,
            "delta_distance_mean": None,
            "segment0_entry_delta_count": 0,
            "segment0_exit_delta_count": 0,
            "warnings": ["no_lineage_records"],
        }

    parent_x0_values: list[float] = []
    offspring_x0_values: list[float] = []
    parent_g_values: list[float] = []
    offspring_g_values: list[float] = []
    parent_distance_values: list[float] = []
    offspring_distance_values: list[float] = []
    delta_x0_values: list[float] = []
    delta_g_values: list[float] = []
    delta_distance_values: list[float] = []
    segment0_entry_count = 0
    segment0_exit_count = 0
    warnings: list[str] = []

    for record in lineage_records:
        offspring_component = dict(record.get("offspring_zdt1_components") or {})
        parent_edges = [
            dict(edge)
            for edge in list(record.get("parent_edges", []))
            if isinstance(edge, Mapping) and isinstance(edge.get("parent_zdt1_components"), Mapping)
        ]
        if not offspring_component or not parent_edges:
            continue
        parent_components = [
            dict(edge.get("parent_zdt1_components") or {})
            for edge in parent_edges
        ]
        parent_x0 = _safe_mean([component.get("x0") for component in parent_components])
        parent_g = _safe_mean([component.get("g") for component in parent_components])
        parent_distance = _safe_mean(
            [component.get("distance_to_zdt1_front") for component in parent_components]
        )
        offspring_x0 = _finite_or_none(offspring_component.get("x0"))
        offspring_g = _finite_or_none(offspring_component.get("g"))
        offspring_distance = _finite_or_none(offspring_component.get("distance_to_zdt1_front"))

        if parent_x0 is not None:
            parent_x0_values.append(parent_x0)
        if offspring_x0 is not None:
            offspring_x0_values.append(offspring_x0)
        if parent_g is not None:
            parent_g_values.append(parent_g)
        if offspring_g is not None:
            offspring_g_values.append(offspring_g)
        if parent_distance is not None:
            parent_distance_values.append(parent_distance)
        if offspring_distance is not None:
            offspring_distance_values.append(offspring_distance)

        delta_x0 = _safe_delta(parent_x0, offspring_x0)
        delta_g = _safe_delta(parent_g, offspring_g)
        delta_distance = _safe_delta(parent_distance, offspring_distance)
        if delta_x0 is not None:
            delta_x0_values.append(delta_x0)
        if delta_g is not None:
            delta_g_values.append(delta_g)
        if delta_distance is not None:
            delta_distance_values.append(delta_distance)

        parent_segment0 = any(bool(component.get("segment0_flag")) for component in parent_components)
        offspring_segment0 = bool(offspring_component.get("segment0_flag"))
        if not parent_segment0 and offspring_segment0:
            segment0_entry_count += 1
        if parent_segment0 and not offspring_segment0:
            segment0_exit_count += 1

        for component in parent_components + [offspring_component]:
            warnings.extend(list(component.get("warnings", [])))

    return {
        "parent_x0_mean": _safe_mean(parent_x0_values),
        "offspring_x0_mean": _safe_mean(offspring_x0_values),
        "delta_x0_mean": _safe_mean(delta_x0_values),
        "parent_g_mean": _safe_mean(parent_g_values),
        "offspring_g_mean": _safe_mean(offspring_g_values),
        "delta_g_mean": _safe_mean(delta_g_values),
        "parent_distance_mean": _safe_mean(parent_distance_values),
        "offspring_distance_mean": _safe_mean(offspring_distance_values),
        "delta_distance_mean": _safe_mean(delta_distance_values),
        "segment0_entry_delta_count": segment0_entry_count,
        "segment0_exit_delta_count": segment0_exit_count,
        "warnings": sorted(set(warnings)),
    }


def summarize_zdt1_mutation_retry_component_effect(
    lineage_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    retry_records = [
        record
        for record in lineage_records
        if int(record.get("retry_attempt_count", 0) or 0) > 0
    ]
    if not retry_records:
        return {
            "retry_count": 0,
            "retry_success_count": 0,
            "retry_x0_delta_mean": None,
            "retry_g_delta_mean": None,
            "retry_distance_delta_mean": None,
            "retry_objective_changed_rate": None,
            "retry_survived_count": 0,
            "retry_segment0_count": 0,
            "retry_segment0_survived_count": 0,
            "warnings": ["no_retry_records"],
        }

    x0_deltas: list[float] = []
    g_deltas: list[float] = []
    distance_deltas: list[float] = []
    objective_changed = 0
    warnings: list[str] = []

    for record in retry_records:
        initial_component = dict(record.get("retry_initial_zdt1_components") or {})
        final_component = dict(record.get("retry_final_zdt1_components") or {})
        delta_x0 = _safe_delta(
            _finite_or_none(initial_component.get("x0")),
            _finite_or_none(final_component.get("x0")),
        )
        delta_g = _safe_delta(
            _finite_or_none(initial_component.get("g")),
            _finite_or_none(final_component.get("g")),
        )
        delta_distance = _safe_delta(
            _finite_or_none(initial_component.get("distance_to_zdt1_front")),
            _finite_or_none(final_component.get("distance_to_zdt1_front")),
        )
        if delta_x0 is not None:
            x0_deltas.append(delta_x0)
        if delta_g is not None:
            g_deltas.append(delta_g)
        if delta_distance is not None:
            distance_deltas.append(delta_distance)
        if any(
            not math.isclose(
                float(initial_component.get(key) or 0.0),
                float(final_component.get(key) or 0.0),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for key in ("f1", "f2", "g")
            if initial_component.get(key) is not None and final_component.get(key) is not None
        ):
            objective_changed += 1
        warnings.extend(list(initial_component.get("warnings", [])))
        warnings.extend(list(final_component.get("warnings", [])))

    return {
        "retry_count": sum(int(record.get("retry_attempt_count", 0) or 0) for record in retry_records),
        "retry_success_count": sum(
            1 for record in retry_records if bool(record.get("retry_success"))
        ),
        "retry_x0_delta_mean": _safe_mean(x0_deltas),
        "retry_g_delta_mean": _safe_mean(g_deltas),
        "retry_distance_delta_mean": _safe_mean(distance_deltas),
        "retry_objective_changed_rate": objective_changed / len(retry_records),
        "retry_survived_count": sum(
            1 for record in retry_records if bool(record.get("offspring_survived_next_generation"))
        ),
        "retry_segment0_count": sum(
            1
            for record in retry_records
            if bool(dict(record.get("retry_final_zdt1_components") or {}).get("segment0_flag"))
        ),
        "retry_segment0_survived_count": sum(
            1
            for record in retry_records
            if bool(dict(record.get("retry_final_zdt1_components") or {}).get("segment0_flag"))
            and bool(record.get("offspring_survived_next_generation"))
        ),
        "warnings": sorted(set(warnings)),
    }


def summarize_zdt1_segment0_quality_funnel(
    *,
    current_population: Sequence[Sequence[float | int]],
    current_objective_vectors: Sequence[Sequence[float]],
    lineage_records: Sequence[Mapping[str, Any]],
    next_population: Sequence[Sequence[float | int]],
    next_objective_vectors: Sequence[Sequence[float]],
    next_front_vectors: Sequence[Sequence[float]],
    bins: int = 6,
    low_g_threshold: float = 1.1,
) -> dict[str, Any]:
    current_components = _collect_zdt1_population_components(
        current_population,
        current_objective_vectors,
        bins=bins,
    )
    next_components = _collect_zdt1_population_components(
        next_population,
        next_objective_vectors,
        bins=bins,
    )
    front_components = _collect_zdt1_population_components(
        next_population[: len(next_front_vectors)],
        next_front_vectors,
        bins=bins,
    )
    offspring_components = [
        dict(record.get("offspring_zdt1_components") or {})
        for record in lineage_records
        if isinstance(record.get("offspring_zdt1_components"), Mapping)
    ]
    segment0_initial = [component for component in current_components if bool(component.get("segment0_flag"))]
    segment0_offspring = [component for component in offspring_components if bool(component.get("segment0_flag"))]
    segment0_survivor = [
        component
        for component in offspring_components
        if bool(component.get("segment0_flag")) and bool(component.get("survived_flag"))
    ]
    segment0_front = [component for component in front_components if bool(component.get("segment0_flag"))]

    warnings = [
        warning
        for component in current_components + next_components + offspring_components + front_components
        for warning in list(component.get("warnings", []))
    ]

    return {
        "low_g_threshold": float(low_g_threshold),
        "segment0_initial_count": len(segment0_initial),
        "segment0_offspring_count": len(segment0_offspring),
        "segment0_low_g_count": sum(
            1
            for component in segment0_offspring
            if isinstance(component.get("g"), int | float)
            and float(component["g"]) <= float(low_g_threshold)
        ),
        "segment0_nondominated_count": sum(
            1 for component in segment0_offspring if bool(component.get("nondominated_flag"))
        ),
        "segment0_survivor_count": len(segment0_survivor),
        "segment0_final_front_count": len(segment0_front),
        "segment0_mean_g_initial": _safe_mean([component.get("g") for component in segment0_initial]),
        "segment0_mean_g_offspring": _safe_mean([component.get("g") for component in segment0_offspring]),
        "segment0_mean_g_survivor": _safe_mean([component.get("g") for component in segment0_survivor]),
        "segment0_mean_distance_final": _safe_mean(
            [component.get("distance_to_zdt1_front") for component in segment0_front]
        ),
        "warnings": sorted(set(warnings)),
    }


def summarize_internal_external_zdt1_component_distribution(
    decision_vectors: Sequence[Sequence[float | int]],
    objective_vectors: Sequence[Sequence[float]],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    warnings: list[str] = []
    if not decision_vectors or len(decision_vectors) != len(objective_vectors):
        warnings.append("decision_vectors_unavailable_or_misaligned")
        components = []
        for objective_vector in objective_vectors:
            if len(objective_vector) >= 2:
                pseudo_summary = {
                    "x0": float(objective_vector[0]),
                    "tail_mean": None,
                    "tail_min": None,
                    "tail_max": None,
                    "tail_std": None,
                }
                components.append(
                    _zdt1_components_from_summary(
                        pseudo_summary,
                        objective_vector=objective_vector,
                        bins=bins,
                    )
                )
    else:
        components = _collect_zdt1_population_components(
            decision_vectors,
            objective_vectors,
            bins=bins,
        )

    segment0 = [component for component in components if bool(component.get("segment0_flag"))]
    occupancy = compute_objective_bins(objective_vectors, [False, False], bins=bins)
    return {
        "x0_min": _safe_mean([min([component["x0"] for component in components if component.get("x0") is not None])] if any(component.get("x0") is not None for component in components) else []),
        "x0_mean": _safe_mean([component.get("x0") for component in components]),
        "x0_max": _safe_mean([max([component["x0"] for component in components if component.get("x0") is not None])] if any(component.get("x0") is not None for component in components) else []),
        "g_min": _safe_mean([min([component["g"] for component in components if component.get("g") is not None])] if any(component.get("g") is not None for component in components) else []),
        "g_mean": _safe_mean([component.get("g") for component in components]),
        "g_max": _safe_mean([max([component["g"] for component in components if component.get("g") is not None])] if any(component.get("g") is not None for component in components) else []),
        "segment0_count": len(segment0),
        "segment0_g_mean": _safe_mean([component.get("g") for component in segment0]),
        "segment0_distance_mean": _safe_mean(
            [component.get("distance_to_zdt1_front") for component in segment0]
        ),
        "occupied_bins": int(occupancy.get("occupied_bins", 0)),
        "nondominated_count": len(objective_vectors),
        "spacing": _finite_or_none(spacing_metric(objective_vectors, [False, False])),
        "warnings": list(occupancy.get("warnings", [])) + warnings,
    }


def summarize_parent_to_offspring(
    lineage_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not lineage_records:
        return {
            "offspring_count": 0,
            "edge_count": 0,
            "unique_parent_count": 0,
            "unique_offspring_count": 0,
            "offspring_survival_rate": None,
            "offspring_nondominated_rate": None,
            "offspring_with_sparse_parent_count": 0,
            "offspring_with_sparse_parent_survived_count": 0,
            "offspring_with_sparse_parent_survival_rate": None,
            "offspring_with_boundary_parent_count": 0,
            "offspring_with_boundary_parent_survived_count": 0,
            "offspring_with_boundary_parent_survival_rate": None,
            "lineage_rows": [],
            "warnings": ["no_lineage_records"],
        }

    lineage_rows: list[dict[str, Any]] = []
    parent_hashes: set[str] = set()
    offspring_hashes: set[str] = set()
    sparse_parent_records = 0
    sparse_parent_survived = 0
    boundary_parent_records = 0
    boundary_parent_survived = 0
    survived_count = 0
    nondominated_count = 0

    for record in lineage_records:
        offspring_hash = str(record.get("offspring_decision_hash"))
        offspring_hashes.add(offspring_hash)
        survived = bool(record.get("offspring_survived_next_generation"))
        nondominated = bool(record.get("offspring_is_nondominated"))
        if survived:
            survived_count += 1
        if nondominated:
            nondominated_count += 1
        parent_edges = list(record.get("parent_edges", []))
        has_sparse_parent = any(bool(edge.get("parent_is_sparse")) for edge in parent_edges)
        has_boundary_parent = any(bool(edge.get("parent_is_boundary")) for edge in parent_edges)
        if has_sparse_parent:
            sparse_parent_records += 1
            if survived:
                sparse_parent_survived += 1
        if has_boundary_parent:
            boundary_parent_records += 1
            if survived:
                boundary_parent_survived += 1
        for edge in parent_edges:
            parent_hash = str(edge.get("parent_id", ""))
            if parent_hash:
                parent_hashes.add(parent_hash)
            lineage_rows.append(
                {
                    "offspring_id": offspring_hash,
                    "offspring_objective": list(record.get("offspring_objective") or []),
                    "offspring_objective_hash": record.get("offspring_objective_hash"),
                    "offspring_rank_after_evaluation": record.get(
                        "offspring_rank_after_evaluation"
                    ),
                    "offspring_is_nondominated": nondominated,
                    "offspring_survived_next_generation": survived,
                    "parent_id": edge.get("parent_id"),
                    "parent_rank": edge.get("parent_rank"),
                    "parent_crowding": edge.get("parent_crowding"),
                    "parent_is_boundary": edge.get("parent_is_boundary"),
                    "parent_sparse_bin": edge.get("parent_sparse_bin"),
                    "selection_kind": edge.get("selection_kind"),
                    "bias_applied": edge.get("bias_applied"),
                }
            )

    offspring_count = len(lineage_records)
    return {
        "offspring_count": offspring_count,
        "edge_count": len(lineage_rows),
        "unique_parent_count": len(parent_hashes),
        "unique_offspring_count": len(offspring_hashes),
        "offspring_survival_rate": survived_count / offspring_count if offspring_count > 0 else None,
        "offspring_nondominated_rate": (
            nondominated_count / offspring_count if offspring_count > 0 else None
        ),
        "offspring_with_sparse_parent_count": sparse_parent_records,
        "offspring_with_sparse_parent_survived_count": sparse_parent_survived,
        "offspring_with_sparse_parent_survival_rate": (
            sparse_parent_survived / sparse_parent_records
            if sparse_parent_records > 0
            else None
        ),
        "offspring_with_boundary_parent_count": boundary_parent_records,
        "offspring_with_boundary_parent_survived_count": boundary_parent_survived,
        "offspring_with_boundary_parent_survival_rate": (
            boundary_parent_survived / boundary_parent_records
            if boundary_parent_records > 0
            else None
        ),
        "lineage_rows": lineage_rows,
        "warnings": [],
    }


def summarize_offspring_to_survivor(
    lineage_records: Sequence[Mapping[str, Any]],
    *,
    directions: Sequence[bool],
    bins: int = 6,
) -> dict[str, Any]:
    if not lineage_records:
        return {
            "offspring_count": 0,
            "offspring_survived_count": 0,
            "offspring_survival_rate": None,
            "sparse_offspring_count": 0,
            "sparse_offspring_survived_count": 0,
            "sparse_offspring_survival_rate": None,
            "boundary_offspring_count": 0,
            "boundary_offspring_survived_count": 0,
            "boundary_offspring_survival_rate": None,
            "offspring_occupied_bins": 0,
            "surviving_offspring_occupied_bins": 0,
            "offspring_unique_objective_count": 0,
            "surviving_offspring_unique_objective_count": 0,
            "offspring_rows": [],
            "warnings": ["no_lineage_records"],
        }

    objective_vectors = [
        list(record.get("offspring_objective") or [])
        for record in lineage_records
        if isinstance(record.get("offspring_objective"), list)
    ]
    occupancy = compute_objective_bins(objective_vectors, directions, bins=bins)
    point_bins = [tuple(value) for value in occupancy.get("point_bins", [])]
    bin_counts = Counter(point_bins)
    sparse_threshold = int(occupancy.get("sparse_threshold", 1))
    boundary_signatures, _ = _boundary_signature_details(objective_vectors, directions)

    offspring_rows: list[dict[str, Any]] = []
    survived_vectors: list[list[float]] = []
    sparse_count = 0
    sparse_survived = 0
    boundary_count = 0
    boundary_survived = 0

    for index, record in enumerate(lineage_records):
        objective_vector = list(record.get("offspring_objective") or [])
        objective_hash = hash_solution_or_objective(objective_vector) if objective_vector else None
        bin_key = point_bins[index] if index < len(point_bins) else None
        is_sparse = bin_key is not None and bin_counts.get(bin_key, 0) <= sparse_threshold
        is_boundary = (
            _signature(objective_vector) in boundary_signatures if objective_vector else False
        )
        survived = bool(record.get("offspring_survived_next_generation"))
        if is_sparse:
            sparse_count += 1
            if survived:
                sparse_survived += 1
        if is_boundary:
            boundary_count += 1
            if survived:
                boundary_survived += 1
        if survived and objective_vector:
            survived_vectors.append(objective_vector)
        offspring_rows.append(
            {
                "offspring_id": record.get("offspring_decision_hash"),
                "offspring_objective_hash": objective_hash,
                "offspring_rank_after_evaluation": record.get("offspring_rank_after_evaluation"),
                "offspring_survived_next_generation": survived,
                "offspring_is_sparse": is_sparse,
                "offspring_is_boundary": is_boundary,
                "offspring_bin": list(bin_key) if bin_key is not None else None,
            }
        )

    survived_occupancy = compute_objective_bins(survived_vectors, directions, bins=bins)
    offspring_count = len(lineage_records)
    survived_count = sum(
        1 for record in lineage_records if bool(record.get("offspring_survived_next_generation"))
    )
    return {
        "offspring_count": offspring_count,
        "offspring_survived_count": survived_count,
        "offspring_survival_rate": survived_count / offspring_count if offspring_count > 0 else None,
        "sparse_offspring_count": sparse_count,
        "sparse_offspring_survived_count": sparse_survived,
        "sparse_offspring_survival_rate": (
            sparse_survived / sparse_count if sparse_count > 0 else None
        ),
        "boundary_offspring_count": boundary_count,
        "boundary_offspring_survived_count": boundary_survived,
        "boundary_offspring_survival_rate": (
            boundary_survived / boundary_count if boundary_count > 0 else None
        ),
        "offspring_occupied_bins": occupancy.get("occupied_bins", 0),
        "surviving_offspring_occupied_bins": survived_occupancy.get("occupied_bins", 0),
        "offspring_unique_objective_count": unique_vector_count(objective_vectors),
        "surviving_offspring_unique_objective_count": unique_vector_count(survived_vectors),
        "offspring_rows": offspring_rows,
        "warnings": list(occupancy.get("warnings", []))
        + list(survived_occupancy.get("warnings", [])),
    }


def summarize_survivor_set_diff(
    candidate_front_vectors: Sequence[Sequence[float]],
    reference_front_vectors: Sequence[Sequence[float]],
    *,
    directions: Sequence[bool],
    bins: int = 6,
) -> dict[str, Any]:
    candidate_hashes = {
        hash_solution_or_objective(vector) for vector in candidate_front_vectors
    }
    reference_hashes = {
        hash_solution_or_objective(vector) for vector in reference_front_vectors
    }
    intersection = candidate_hashes & reference_hashes
    union = candidate_hashes | reference_hashes

    def _mean_nearest(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float | None:
        if not left or not right:
            return None
        distances = [
            min(math.dist(list(left_vector), list(right_vector)) for right_vector in right)
            for left_vector in left
        ]
        return _safe_mean(distances)

    candidate_boundary, _ = _boundary_signature_details(candidate_front_vectors, directions)
    reference_boundary, _ = _boundary_signature_details(reference_front_vectors, directions)
    candidate_occupancy = compute_objective_bins(candidate_front_vectors, directions, bins=bins)
    reference_occupancy = compute_objective_bins(reference_front_vectors, directions, bins=bins)

    def _sparse_bin_keys(snapshot: Mapping[str, Any]) -> set[tuple[int, ...]]:
        bin_counts = {
            tuple(int(part) for part in key.strip("()").split(",") if part.strip())
            if isinstance(key, str)
            else tuple(key): int(value)
            for key, value in snapshot.get("bin_counts", {}).items()
        }
        threshold = int(snapshot.get("sparse_threshold", 1))
        return {key for key, value in bin_counts.items() if value <= threshold}

    sparse_diff = _sparse_bin_keys(candidate_occupancy) ^ _sparse_bin_keys(reference_occupancy)

    return {
        "candidate_front_size": len(candidate_front_vectors),
        "reference_front_size": len(reference_front_vectors),
        "objective_overlap_rate": len(intersection) / len(union) if union else None,
        "nearest_neighbor_distance_to_reference": _mean_nearest(
            candidate_front_vectors,
            reference_front_vectors,
        ),
        "reference_neighbor_distance_to_candidate": _mean_nearest(
            reference_front_vectors,
            candidate_front_vectors,
        ),
        "unique_to_candidate_count": len(candidate_hashes - reference_hashes),
        "unique_to_reference_count": len(reference_hashes - candidate_hashes),
        "boundary_diff_count": len(candidate_boundary ^ reference_boundary),
        "sparse_bin_diff_count": len(sparse_diff),
        "warnings": list(candidate_occupancy.get("warnings", []))
        + list(reference_occupancy.get("warnings", [])),
    }


def summarize_objective_boundary_retention_detail(
    previous_front_vectors: Sequence[Sequence[float]],
    current_front_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    previous_front_ranks: Sequence[int] | None = None,
    previous_front_crowding: Sequence[float] | None = None,
) -> dict[str, Any]:
    current_signatures = {_signature(vector) for vector in current_front_vectors}
    _, boundary_rows = _boundary_signature_details(previous_front_vectors, directions)
    signature_to_index = {
        _signature(vector): index for index, vector in enumerate(previous_front_vectors)
    }
    detail_rows: list[dict[str, Any]] = []
    min_total = 0
    min_retained = 0
    max_total = 0
    max_retained = 0

    for row in boundary_rows:
        min_signatures = [tuple(value) for value in row["min_signatures"]]
        max_signatures = [tuple(value) for value in row["max_signatures"]]
        min_signature = min_signatures[0] if min_signatures else None
        max_signature = max_signatures[0] if max_signatures else None
        min_index = signature_to_index.get(min_signature) if min_signature is not None else None
        max_index = signature_to_index.get(max_signature) if max_signature is not None else None
        min_is_retained = any(signature in current_signatures for signature in min_signatures)
        max_is_retained = any(signature in current_signatures for signature in max_signatures)
        min_total += 1 if min_signature is not None else 0
        max_total += 1 if max_signature is not None else 0
        min_retained += 1 if min_is_retained else 0
        max_retained += 1 if max_is_retained else 0
        detail_rows.append(
            {
                "objective_index": int(row["objective"]),
                "min_point_hash": (
                    hash_solution_or_objective(min_signature) if min_signature is not None else None
                ),
                "max_point_hash": (
                    hash_solution_or_objective(max_signature) if max_signature is not None else None
                ),
                "min_retained_next_generation": min_is_retained,
                "max_retained_next_generation": max_is_retained,
                "min_rank": (
                    int(previous_front_ranks[min_index])
                    if min_index is not None and previous_front_ranks is not None
                    else None
                ),
                "max_rank": (
                    int(previous_front_ranks[max_index])
                    if max_index is not None and previous_front_ranks is not None
                    else None
                ),
                "min_crowding": (
                    _finite_or_none(previous_front_crowding[min_index])
                    if min_index is not None and previous_front_crowding is not None
                    else None
                ),
                "max_crowding": (
                    _finite_or_none(previous_front_crowding[max_index])
                    if max_index is not None and previous_front_crowding is not None
                    else None
                ),
            }
        )

    return {
        "objective_count": len(detail_rows),
        "min_retained_rate": min_retained / min_total if min_total > 0 else None,
        "max_retained_rate": max_retained / max_total if max_total > 0 else None,
        "detail_rows": detail_rows,
        "warnings": [],
    }


def summarize_segment_spacing_attribution(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    return compute_spacing_by_segment(objective_vectors, directions, bins=bins)


def summarize_crowding_decision_attribution(
    *,
    population_size: int,
    combined_fronts: Sequence[Sequence[int]],
    combined_ranks: Sequence[int],
    combined_crowding: Sequence[float],
    survivor_indices: Sequence[int],
) -> dict[str, Any]:
    selected_before_partial = 0
    truncated_front: list[int] = []
    truncated_rank: int | None = None
    for front in combined_fronts:
        if selected_before_partial + len(front) <= population_size:
            selected_before_partial += len(front)
            continue
        truncated_front = [int(index) for index in front]
        truncated_rank = int(combined_ranks[truncated_front[0]]) if truncated_front else None
        break

    if not truncated_front:
        return {
            "rank": None,
            "candidate_count": 0,
            "selected_count": 0,
            "crowding_mean_selected": None,
            "crowding_mean_rejected": None,
            "inf_crowding_selected_count": 0,
            "inf_crowding_rejected_count": 0,
            "same_rank_tie_count": 0,
            "rows": [],
            "warnings": ["no_partial_front_truncation"],
        }

    selected_set = {int(index) for index in survivor_indices}
    selected_values = [
        float(combined_crowding[index])
        for index in truncated_front
        if index in selected_set and math.isfinite(float(combined_crowding[index]))
    ]
    rejected_values = [
        float(combined_crowding[index])
        for index in truncated_front
        if index not in selected_set and math.isfinite(float(combined_crowding[index]))
    ]
    finite_values = [
        round(float(combined_crowding[index]), 12)
        for index in truncated_front
        if math.isfinite(float(combined_crowding[index]))
    ]
    value_counts = Counter(finite_values)
    same_rank_tie_count = sum(count - 1 for count in value_counts.values() if count > 1)
    row = {
        "rank": truncated_rank,
        "candidate_count": len(truncated_front),
        "selected_count": sum(1 for index in truncated_front if index in selected_set),
        "crowding_mean_selected": _safe_mean(selected_values),
        "crowding_mean_rejected": _safe_mean(rejected_values),
        "inf_crowding_selected_count": sum(
            1
            for index in truncated_front
            if index in selected_set and math.isinf(float(combined_crowding[index]))
        ),
        "inf_crowding_rejected_count": sum(
            1
            for index in truncated_front
            if index not in selected_set and math.isinf(float(combined_crowding[index]))
        ),
        "same_rank_tie_count": same_rank_tie_count,
    }
    return {
        **row,
        "rows": [row],
        "warnings": [],
    }


def _duplicate_count(values: Sequence[str | tuple[float, ...]]) -> int:
    return max(0, len(values) - len(set(values)))


def _duplicate_rate(values: Sequence[str | tuple[float, ...]]) -> float | None:
    if not values:
        return None
    return _duplicate_count(values) / len(values)


def _candidate_signature_for_mode(
    index: int,
    population: Sequence[Sequence[float | int]] | None,
    objective_vectors: Sequence[Sequence[float]] | None,
    mode: str,
) -> tuple[float, ...] | None:
    if mode == "decision":
        if population is None or index >= len(population):
            return None
        return _signature(population[index])
    if mode == "objective":
        if objective_vectors is None or index >= len(objective_vectors):
            return None
        return _signature(objective_vectors[index])
    return None


def _boundary_preservation_order(
    front: Sequence[int],
    crowding: Sequence[float],
    objective_vectors: Sequence[Sequence[float]] | None,
) -> list[int]:
    ordered = sorted(front, key=lambda idx: crowding[idx], reverse=True)
    if objective_vectors is None or len(ordered) <= 2:
        return ordered

    objective_len = len(objective_vectors[0])
    boundary_scores: dict[int, float] = {idx: 0.0 for idx in ordered}
    for objective_index in range(objective_len):
        values = [objective_vectors[idx][objective_index] for idx in ordered]
        min_value = min(values)
        max_value = max(values)
        scale = max_value - min_value
        if math.isclose(scale, 0.0, abs_tol=1e-12):
            continue
        for idx in ordered:
            normalized = (objective_vectors[idx][objective_index] - min_value) / scale
            boundary_scores[idx] += abs((2.0 * normalized) - 1.0)
    return sorted(
        ordered,
        key=lambda idx: (
            0 if math.isinf(crowding[idx]) else 1,
            -(crowding[idx] if math.isfinite(crowding[idx]) else 0.0),
            -boundary_scores[idx],
        ),
    )


def _simulate_partial_front_selection(
    *,
    ordered_front: Sequence[int],
    remaining: int,
    selected_prefix: Sequence[int],
    partial_front_dedup_mode: str,
    combined_population: Sequence[Sequence[float | int]] | None,
    combined_objective_vectors: Sequence[Sequence[float]] | None,
) -> list[int]:
    if remaining <= 0:
        return []
    if (
        partial_front_dedup_mode not in {"decision", "objective"}
        or combined_objective_vectors is None
    ):
        return list(ordered_front[:remaining])

    selected_signatures = {
        signature
        for idx in selected_prefix
        if (
            signature := _candidate_signature_for_mode(
                idx,
                combined_population,
                combined_objective_vectors,
                partial_front_dedup_mode,
            )
        )
        is not None
    }
    chosen: list[int] = []
    deferred: list[int] = []
    for idx in ordered_front:
        signature = _candidate_signature_for_mode(
            idx,
            combined_population,
            combined_objective_vectors,
            partial_front_dedup_mode,
        )
        if signature is not None and signature in selected_signatures:
            deferred.append(idx)
            continue
        chosen.append(idx)
        if signature is not None:
            selected_signatures.add(signature)
        if len(chosen) >= remaining:
            break
    if len(chosen) < remaining:
        chosen.extend(deferred[: remaining - len(chosen)])
    return chosen[:remaining]


def summarize_lineage_retention_funnel(
    lineage_records: Sequence[Mapping[str, Any]],
    *,
    next_front_vectors: Sequence[Sequence[float]],
) -> dict[str, Any]:
    if not lineage_records:
        return {
            "selected_parent_count": 0,
            "altered_parent_count": 0,
            "altered_parent_offspring_count": 0,
            "altered_offspring_evaluated_count": 0,
            "altered_offspring_nondominated_count": 0,
            "altered_offspring_survived_count": 0,
            "altered_offspring_final_front_count": 0,
            "retained_lineage_ratio_parent_to_offspring": None,
            "retained_lineage_ratio_offspring_to_survivor": None,
            "retained_lineage_ratio_survivor_to_front": None,
            "warnings": ["no_lineage_records"],
        }

    next_front_hashes = {
        hash_solution_or_objective(vector) for vector in next_front_vectors if vector
    }
    selected_parent_count = 0
    altered_parent_count = 0
    altered_parent_unique: set[str] = set()
    altered_offspring_records: list[Mapping[str, Any]] = []
    for record in lineage_records:
        parent_edges = list(record.get("parent_edges", []))
        selected_parent_count += len(parent_edges)
        altered_edges = [
            edge
            for edge in parent_edges
            if bool(edge.get("bias_applied"))
            or str(edge.get("selection_kind")) == "sparse_parent_bias_light"
        ]
        altered_parent_count += len(altered_edges)
        for edge in altered_edges:
            parent_id = str(edge.get("parent_id") or "")
            if parent_id:
                altered_parent_unique.add(parent_id)
        if altered_edges:
            altered_offspring_records.append(record)

    altered_offspring_nondominated_count = sum(
        1 for record in altered_offspring_records if bool(record.get("offspring_is_nondominated"))
    )
    altered_offspring_survived_count = sum(
        1
        for record in altered_offspring_records
        if bool(record.get("offspring_survived_next_generation"))
    )
    altered_offspring_final_front_count = sum(
        1
        for record in altered_offspring_records
        if bool(record.get("offspring_survived_next_generation"))
        and str(record.get("offspring_objective_hash") or "") in next_front_hashes
    )

    altered_offspring_count = len(altered_offspring_records)
    return {
        "selected_parent_count": selected_parent_count,
        "selected_parent_unique_count": len(
            {
                str(edge.get("parent_id") or "")
                for record in lineage_records
                for edge in list(record.get("parent_edges", []))
                if str(edge.get("parent_id") or "")
            }
        ),
        "altered_parent_count": altered_parent_count,
        "altered_parent_unique_count": len(altered_parent_unique),
        "altered_parent_offspring_count": altered_offspring_count,
        "altered_offspring_evaluated_count": altered_offspring_count,
        "altered_offspring_nondominated_count": altered_offspring_nondominated_count,
        "altered_offspring_survived_count": altered_offspring_survived_count,
        "altered_offspring_final_front_count": altered_offspring_final_front_count,
        "retained_lineage_ratio_parent_to_offspring": (
            altered_offspring_count / altered_parent_count
            if altered_parent_count > 0
            else None
        ),
        "retained_lineage_ratio_offspring_to_survivor": (
            altered_offspring_survived_count / altered_offspring_count
            if altered_offspring_count > 0
            else None
        ),
        "retained_lineage_ratio_survivor_to_front": (
            altered_offspring_final_front_count / altered_offspring_survived_count
            if altered_offspring_survived_count > 0
            else None
        ),
        "warnings": [] if altered_parent_count > 0 else ["no_altered_lineage_detected"],
    }


def summarize_sparse_lineage_quality(
    lineage_records: Sequence[Mapping[str, Any]],
    *,
    directions: Sequence[bool],
    next_front_vectors: Sequence[Sequence[float]],
    bins: int = 6,
) -> dict[str, Any]:
    if not lineage_records:
        return {
            "sparse_parent_count": 0,
            "sparse_parent_offspring_count": 0,
            "sparse_offspring_mean_rank": None,
            "sparse_offspring_nondominated_rate": None,
            "sparse_offspring_survival_rate": None,
            "sparse_offspring_mean_distance_to_front": None,
            "sparse_offspring_segment_distribution": {},
            "warnings": ["no_lineage_records"],
        }

    sparse_parent_count = 0
    sparse_offspring_records: list[Mapping[str, Any]] = []
    sparse_vectors: list[list[float]] = []
    for record in lineage_records:
        parent_edges = list(record.get("parent_edges", []))
        sparse_edges = [edge for edge in parent_edges if bool(edge.get("parent_is_sparse"))]
        sparse_parent_count += len(sparse_edges)
        if sparse_edges:
            sparse_offspring_records.append(record)
            objective_vector = list(record.get("offspring_objective") or [])
            if objective_vector:
                sparse_vectors.append(objective_vector)

    occupancy = compute_objective_bins(sparse_vectors, directions, bins=bins)
    point_bins = [
        tuple(int(value) for value in bin_key)
        for bin_key in occupancy.get("point_bins", [])
        if isinstance(bin_key, (list, tuple))
    ]
    segment_distribution = {
        str(key): int(value) for key, value in Counter(point_bins).items()
    }

    def _distance_to_front(vector: Sequence[float]) -> float | None:
        if not vector or not next_front_vectors:
            return None
        distances = [math.dist(list(vector), list(front)) for front in next_front_vectors]
        return _safe_mean(distances[:1]) if distances else None

    return {
        "sparse_parent_count": sparse_parent_count,
        "sparse_parent_offspring_count": len(sparse_offspring_records),
        "sparse_offspring_mean_rank": _safe_mean(
            [record.get("offspring_rank_after_evaluation") for record in sparse_offspring_records]
        ),
        "sparse_offspring_nondominated_rate": (
            sum(
                1
                for record in sparse_offspring_records
                if bool(record.get("offspring_is_nondominated"))
            )
            / len(sparse_offspring_records)
            if sparse_offspring_records
            else None
        ),
        "sparse_offspring_survival_rate": (
            sum(
                1
                for record in sparse_offspring_records
                if bool(record.get("offspring_survived_next_generation"))
            )
            / len(sparse_offspring_records)
            if sparse_offspring_records
            else None
        ),
        "sparse_offspring_mean_distance_to_front": _safe_mean(
            [
                min(
                    math.dist(list(record.get("offspring_objective") or []), list(front))
                    for front in next_front_vectors
                )
                for record in sparse_offspring_records
                if list(record.get("offspring_objective") or []) and next_front_vectors
            ]
        ),
        "sparse_offspring_segment_distribution": segment_distribution,
        "warnings": list(occupancy.get("warnings", []))
        + ([] if sparse_parent_count > 0 else ["no_sparse_parent_lineage_detected"]),
    }


def summarize_survivor_divergence_by_generation(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "mean_divergence_vs_reference": None,
            "mean_convergence_back_to_reference_rate": None,
            "mean_unique_candidate_points": None,
            "rows": [],
            "warnings": ["no_divergence_rows"],
        }

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("seed", 0)),
            int(row.get("generation", 0)),
        ),
    )
    previous_divergence: float | None = None
    enriched_rows: list[dict[str, Any]] = []
    for row in ordered_rows:
        overlap = _finite_or_none(row.get("objective_overlap_rate"))
        divergence = None if overlap is None else float(1.0 - overlap)
        delta = None
        convergence = None
        if previous_divergence is not None and divergence is not None:
            delta = float(divergence - previous_divergence)
            if previous_divergence > 0.0:
                convergence = max(0.0, previous_divergence - divergence) / previous_divergence
        enriched_rows.append(
            {
                **row,
                "divergence_vs_reference": divergence,
                "divergence_delta_from_previous_generation": delta,
                "convergence_back_to_reference_rate": convergence,
            }
        )
        if divergence is not None:
            previous_divergence = divergence

    return {
        "mean_divergence_vs_reference": _safe_mean(
            [row.get("divergence_vs_reference") for row in enriched_rows]
        ),
        "mean_convergence_back_to_reference_rate": _safe_mean(
            [row.get("convergence_back_to_reference_rate") for row in enriched_rows]
        ),
        "mean_unique_candidate_points": _safe_mean(
            [row.get("unique_to_candidate_count") for row in enriched_rows]
        ),
        "rows": enriched_rows,
        "warnings": [],
    }


def summarize_segment0_spacing_detail(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    summary = compute_spacing_by_segment(objective_vectors, directions, bins=bins)
    segment_rows = list(summary.get("segment_rows", []))
    segment0 = (
        segment_rows[0]
        if segment_rows and isinstance(segment_rows[0], dict)
        else {
            "segment_id": 0,
            "segment_range": [0.0, 1.0 / max(1, bins)],
            "points_in_segment": 0,
            "mean_gap": None,
            "max_gap": None,
            "local_spacing_contribution": None,
            "empty_segment": True,
        }
    )
    warnings = list(summary.get("warnings", []))
    return {
        "segment_id": 0,
        "segment_range": list(segment0.get("segment_range", [0.0, 1.0 / max(1, bins)])),
        "point_count": int(segment0.get("points_in_segment", 0) or 0),
        "empty_segment": bool(segment0.get("empty_segment")),
        "left_gap": segment0.get("mean_gap"),
        "right_gap": segment0.get("max_gap"),
        "max_local_gap": segment0.get("max_gap"),
        "local_spacing_contribution": segment0.get("local_spacing_contribution"),
        "boundary_adjacent": True,
        "affected_objective_range": list(
            segment0.get("segment_range", [0.0, 1.0 / max(1, bins)])
        ),
        "warnings": warnings,
    }


def summarize_duplicate_to_diversity_funnel(
    *,
    current_population: Sequence[Sequence[float | int]],
    current_objective_vectors: Sequence[Sequence[float]],
    next_population: Sequence[Sequence[float | int]],
    next_objective_vectors: Sequence[Sequence[float]],
    next_front_vectors: Sequence[Sequence[float]],
    lineage_records: Sequence[Mapping[str, Any]],
    directions: Sequence[bool],
    bins: int = 6,
) -> dict[str, Any]:
    current_decision_hashes = [hash_solution_or_objective(genome) for genome in current_population]
    next_decision_hashes = [hash_solution_or_objective(genome) for genome in next_population]
    current_objective_hashes = [
        hash_solution_or_objective(vector) for vector in current_objective_vectors
    ]
    next_objective_hashes = [
        hash_solution_or_objective(vector) for vector in next_objective_vectors
    ]
    next_front_hashes = [hash_solution_or_objective(vector) for vector in next_front_vectors]

    replacement_candidates = [
        record
        for record in lineage_records
        if str(record.get("offspring_decision_hash") or "") not in current_decision_hashes
    ]
    replacement_survived = [
        record
        for record in replacement_candidates
        if bool(record.get("offspring_survived_next_generation"))
    ]
    occupancy = compute_objective_bins(next_front_vectors, directions, bins=bins)

    return {
        "decision_duplicate_rate": _duplicate_rate(next_decision_hashes),
        "objective_duplicate_rate": _duplicate_rate(next_objective_hashes),
        "archive_duplicate_rate": _duplicate_rate(next_front_hashes),
        "unique_decision_count": len(set(next_decision_hashes)),
        "unique_objective_count": len(set(next_objective_hashes)),
        "occupied_bins": int(occupancy.get("occupied_bins", 0)),
        "nondominated_count": len(next_front_vectors),
        "spacing": _finite_or_none(spacing_metric(next_objective_vectors, directions)),
        "duplicate_removed_count": max(
            0,
            _duplicate_count(current_decision_hashes) - _duplicate_count(next_decision_hashes),
        ),
        "replacement_candidate_count": len(replacement_candidates),
        "replacement_survived_count": len(replacement_survived),
        "warnings": list(occupancy.get("warnings", [])),
    }


def summarize_boundary_intervention_count(
    *,
    population_size: int,
    combined_fronts: Sequence[Sequence[int]],
    combined_crowding: Sequence[float],
    combined_population: Sequence[Sequence[float | int]],
    combined_objective_vectors: Sequence[Sequence[float]],
    partial_front_strategy: str,
    partial_front_dedup_mode: str,
) -> dict[str, Any]:
    selected_before_partial = 0
    truncated_front: list[int] = []
    selected_prefix: list[int] = []
    for front in combined_fronts:
        if selected_before_partial + len(front) <= population_size:
            selected_prefix.extend(int(index) for index in front)
            selected_before_partial += len(front)
            continue
        truncated_front = [int(index) for index in front]
        break

    if not truncated_front:
        return {
            "boundary_candidate_count": 0,
            "boundary_preference_trigger_count": 0,
            "boundary_preference_changed_selection_count": 0,
            "boundary_retained_due_to_preference_count": 0,
            "boundary_already_retained_without_preference_estimate": 0,
            "boundary_effect_size_estimate": 0.0,
            "warnings": ["no_partial_front_truncation"],
        }

    remaining = max(0, population_size - len(selected_prefix))
    if partial_front_strategy != "boundary_preservation_light":
        boundary_candidates = _boundary_signature_details(
            [combined_objective_vectors[index] for index in truncated_front],
            [False] * len(combined_objective_vectors[0]) if combined_objective_vectors else [],
        )[0]
        return {
            "boundary_candidate_count": len(boundary_candidates),
            "boundary_preference_trigger_count": 0,
            "boundary_preference_changed_selection_count": 0,
            "boundary_retained_due_to_preference_count": 0,
            "boundary_already_retained_without_preference_estimate": 0,
            "boundary_effect_size_estimate": 0.0,
            "warnings": ["boundary_preference_not_active"],
        }

    baseline_order = sorted(truncated_front, key=lambda idx: combined_crowding[idx], reverse=True)
    boundary_order = _boundary_preservation_order(
        truncated_front,
        combined_crowding,
        combined_objective_vectors,
    )
    baseline_selected = _simulate_partial_front_selection(
        ordered_front=baseline_order,
        remaining=remaining,
        selected_prefix=selected_prefix,
        partial_front_dedup_mode=partial_front_dedup_mode,
        combined_population=combined_population,
        combined_objective_vectors=combined_objective_vectors,
    )
    boundary_selected = _simulate_partial_front_selection(
        ordered_front=boundary_order,
        remaining=remaining,
        selected_prefix=selected_prefix,
        partial_front_dedup_mode=partial_front_dedup_mode,
        combined_population=combined_population,
        combined_objective_vectors=combined_objective_vectors,
    )

    front_vectors = [combined_objective_vectors[index] for index in truncated_front]
    boundary_signatures, _ = _boundary_signature_details(
        front_vectors,
        [False] * len(front_vectors[0]) if front_vectors else [],
    )
    index_to_signature = {
        index: _signature(combined_objective_vectors[index]) for index in truncated_front
    }
    boundary_candidate_indices = {
        index
        for index in truncated_front
        if index_to_signature.get(index) in boundary_signatures
    }
    boundary_selected_set = set(boundary_selected)
    baseline_selected_set = set(baseline_selected)
    gained_indices = boundary_selected_set - baseline_selected_set
    retained_due_to_preference = sum(
        1 for index in gained_indices if index in boundary_candidate_indices
    )

    return {
        "boundary_candidate_count": len(boundary_candidate_indices),
        "boundary_preference_trigger_count": 1,
        "boundary_preference_changed_selection_count": len(gained_indices),
        "boundary_retained_due_to_preference_count": retained_due_to_preference,
        "boundary_already_retained_without_preference_estimate": sum(
            1 for index in baseline_selected if index in boundary_candidate_indices
        ),
        "boundary_effect_size_estimate": (
            len(gained_indices) / remaining if remaining > 0 else 0.0
        ),
        "warnings": [],
    }


def summarize_initialization_segment_coverage(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    distribution = compute_segment_distribution(objective_vectors, directions, bins=bins)
    normalized = _to_minimization(objective_vectors, directions) if objective_vectors else []
    objective_min = (
        [float(min(vector[index] for vector in normalized)) for index in range(len(normalized[0]))]
        if normalized
        else []
    )
    objective_max = (
        [float(max(vector[index] for vector in normalized)) for index in range(len(normalized[0]))]
        if normalized
        else []
    )
    objective_range = (
        [float(maximum - minimum) for minimum, maximum in zip(objective_min, objective_max, strict=True)]
        if objective_min and objective_max
        else []
    )
    segment_counts = dict(distribution.get("segment_counts", {}))
    segment_rates = dict(distribution.get("segment_rates", {}))
    return {
        "population_size": len(objective_vectors),
        "occupied_bins": int(distribution.get("occupied_segments", 0)),
        "segment0_count": int(segment_counts.get("0", 0)),
        "segment0_rate": float(segment_rates.get("0", 0.0)),
        "boundary_adjacent_count": int(distribution.get("boundary_adjacent_count", 0)),
        "objective_min": objective_min,
        "objective_max": objective_max,
        "objective_range": objective_range,
        "unique_objective_count": len({_signature(vector) for vector in objective_vectors}),
        "segment0_range": list(
            (distribution.get("segment_ranges", []) or [[0.0, 1.0 / max(1, bins)]])[0]
        ),
        "warnings": list(distribution.get("warnings", [])),
    }


def summarize_variation_segment_transition(
    lineage_records: Sequence[Mapping[str, Any]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    if not lineage_records:
        return {
            "transition_count": 0,
            "segment0_entry_count": 0,
            "segment0_exit_count": 0,
            "segment0_entry_rate": None,
            "segment0_exit_rate": None,
            "boundary_entry_count": 0,
            "boundary_exit_count": 0,
            "boundary_entry_rate": None,
            "boundary_exit_rate": None,
            "dominant_transitions": [],
            "transition_rows": [],
            "warnings": ["no_lineage_records"],
        }

    offspring_vectors = [
        list(record.get("offspring_objective") or [])
        for record in lineage_records
        if isinstance(record.get("offspring_objective"), list)
    ]
    distribution = compute_segment_distribution(offspring_vectors, directions, bins=bins)
    point_segments = list(distribution.get("point_segments", []))
    point_bins = [
        tuple(int(value) for value in bin_key)
        for bin_key in distribution.get("point_bins", [])
        if isinstance(bin_key, list)
    ]

    transition_counts: Counter[str] = Counter()
    transition_rows: list[dict[str, Any]] = []
    segment0_entry_count = 0
    segment0_exit_count = 0
    boundary_entry_count = 0
    boundary_exit_count = 0

    for index, record in enumerate(lineage_records):
        parent_edges = list(record.get("parent_edges", []))
        parent_segments = [
            int(edge.get("parent_sparse_bin", [])[0])
            for edge in parent_edges
            if isinstance(edge.get("parent_sparse_bin"), list)
            and edge.get("parent_sparse_bin")
        ]
        parent_segment_pair = tuple(parent_segments)
        offspring_segment = (
            int(point_segments[index]) if index < len(point_segments) else None
        )
        offspring_bin = point_bins[index] if index < len(point_bins) else None
        parent_boundary = any(
            compute_boundary_adjacent_flag(edge.get("parent_sparse_bin"), bins=bins)
            for edge in parent_edges
        )
        offspring_boundary = compute_boundary_adjacent_flag(offspring_bin, bins=bins)
        transition_label = f"{list(parent_segment_pair)}->{offspring_segment}"
        transition_counts[transition_label] += 1

        if offspring_segment == 0 and not all(segment == 0 for segment in parent_segments):
            segment0_entry_count += 1
        if parent_segments and any(segment == 0 for segment in parent_segments) and offspring_segment != 0:
            segment0_exit_count += 1
        if offspring_boundary and not parent_boundary:
            boundary_entry_count += 1
        if parent_boundary and not offspring_boundary:
            boundary_exit_count += 1

        transition_rows.append(
            {
                "parent_segment_pair": list(parent_segment_pair),
                "offspring_segment": offspring_segment,
                "transition_label": transition_label,
                "parent_boundary": parent_boundary,
                "offspring_boundary": offspring_boundary,
            }
        )

    total = len(lineage_records)
    dominant_transitions = [
        {
            "transition": label,
            "count": int(count),
            "rate": float(count / total) if total > 0 else None,
        }
        for label, count in transition_counts.most_common(5)
    ]
    return {
        "transition_count": total,
        "segment0_entry_count": segment0_entry_count,
        "segment0_exit_count": segment0_exit_count,
        "segment0_entry_rate": (
            float(segment0_entry_count / total) if total > 0 else None
        ),
        "segment0_exit_rate": (
            float(segment0_exit_count / total) if total > 0 else None
        ),
        "boundary_entry_count": boundary_entry_count,
        "boundary_exit_count": boundary_exit_count,
        "boundary_entry_rate": (
            float(boundary_entry_count / total) if total > 0 else None
        ),
        "boundary_exit_rate": (
            float(boundary_exit_count / total) if total > 0 else None
        ),
        "dominant_transitions": dominant_transitions,
        "transition_rows": transition_rows,
        "warnings": list(distribution.get("warnings", [])),
    }


def summarize_operator_offspring_quality(
    lineage_records: Sequence[Mapping[str, Any]],
    *,
    directions: Sequence[bool],
    next_front_vectors: Sequence[Sequence[float]],
    bins: int = 6,
) -> dict[str, Any]:
    if not lineage_records:
        return {
            "offspring_count": 0,
            "offspring_nondominated_rate": None,
            "offspring_mean_rank": None,
            "offspring_survival_rate": None,
            "segment0_offspring_count": 0,
            "segment0_offspring_nondominated_rate": None,
            "segment0_offspring_survival_rate": None,
            "segment0_offspring_mean_distance_to_front": None,
            "boundary_offspring_count": 0,
            "boundary_offspring_survival_rate": None,
            "warnings": ["no_lineage_records"],
        }

    offspring_vectors = [
        list(record.get("offspring_objective") or [])
        for record in lineage_records
        if isinstance(record.get("offspring_objective"), list)
    ]
    distribution = compute_segment_distribution(offspring_vectors, directions, bins=bins)
    point_segments = list(distribution.get("point_segments", []))
    point_bins = [
        tuple(int(value) for value in bin_key)
        for bin_key in distribution.get("point_bins", [])
        if isinstance(bin_key, list)
    ]

    segment0_records: list[Mapping[str, Any]] = []
    boundary_records: list[Mapping[str, Any]] = []
    segment0_distances: list[float] = []
    for index, record in enumerate(lineage_records):
        objective_vector = list(record.get("offspring_objective") or [])
        offspring_segment = int(point_segments[index]) if index < len(point_segments) else None
        offspring_bin = point_bins[index] if index < len(point_bins) else None
        if offspring_segment == 0:
            segment0_records.append(record)
            if objective_vector and next_front_vectors:
                segment0_distances.append(
                    min(
                        math.dist(objective_vector, list(front))
                        for front in next_front_vectors
                    )
                )
        if compute_boundary_adjacent_flag(offspring_bin, bins=bins):
            boundary_records.append(record)

    return {
        "offspring_count": len(lineage_records),
        "offspring_nondominated_rate": (
            sum(1 for record in lineage_records if bool(record.get("offspring_is_nondominated")))
            / len(lineage_records)
            if lineage_records
            else None
        ),
        "offspring_mean_rank": _safe_mean(
            [record.get("offspring_rank_after_evaluation") for record in lineage_records]
        ),
        "offspring_survival_rate": (
            sum(
                1 for record in lineage_records if bool(record.get("offspring_survived_next_generation"))
            )
            / len(lineage_records)
            if lineage_records
            else None
        ),
        "segment0_offspring_count": len(segment0_records),
        "segment0_offspring_nondominated_rate": (
            sum(1 for record in segment0_records if bool(record.get("offspring_is_nondominated")))
            / len(segment0_records)
            if segment0_records
            else None
        ),
        "segment0_offspring_survival_rate": (
            sum(
                1 for record in segment0_records if bool(record.get("offspring_survived_next_generation"))
            )
            / len(segment0_records)
            if segment0_records
            else None
        ),
        "segment0_offspring_mean_distance_to_front": _safe_mean(segment0_distances),
        "boundary_offspring_count": len(boundary_records),
        "boundary_offspring_survival_rate": (
            sum(
                1 for record in boundary_records if bool(record.get("offspring_survived_next_generation"))
            )
            / len(boundary_records)
            if boundary_records
            else None
        ),
        "warnings": list(distribution.get("warnings", [])),
    }


def summarize_mutation_retry_objective_effect(
    lineage_records: Sequence[Mapping[str, Any]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    retry_records = [
        record
        for record in lineage_records
        if int(record.get("retry_attempt_count", 0) or 0) > 0
        or bool(record.get("duplicate_detected"))
    ]
    if not retry_records:
        return {
            "retry_count": 0,
            "retry_success_count": 0,
            "reinitialized_count": 0,
            "retry_offspring_unique_decision_count": 0,
            "retry_offspring_unique_objective_count": 0,
            "retry_offspring_survived_count": 0,
            "decision_changed_after_retry_rate": None,
            "objective_changed_after_retry_rate": None,
            "retry_offspring_segment_distribution": {},
            "warnings": [
                "no_retry_events",
                "objective_changed_after_retry_requires_extra_evaluation",
            ],
        }

    retry_vectors = [
        list(record.get("offspring_objective") or [])
        for record in retry_records
        if isinstance(record.get("offspring_objective"), list)
    ]
    distribution = compute_segment_distribution(retry_vectors, directions, bins=bins)
    point_segments = list(distribution.get("point_segments", []))
    return {
        "retry_count": sum(int(record.get("retry_attempt_count", 0) or 0) for record in retry_records),
        "retry_success_count": sum(
            1 for record in retry_records if bool(record.get("retry_success"))
        ),
        "reinitialized_count": sum(
            1 for record in retry_records if bool(record.get("retry_reinitialized"))
        ),
        "retry_offspring_unique_decision_count": len(
            {str(record.get("offspring_decision_hash")) for record in retry_records}
        ),
        "retry_offspring_unique_objective_count": len(
            {
                _signature(record.get("offspring_objective") or [])
                for record in retry_records
                if isinstance(record.get("offspring_objective"), list)
                and record.get("offspring_objective")
            }
        ),
        "retry_offspring_survived_count": sum(
            1 for record in retry_records if bool(record.get("offspring_survived_next_generation"))
        ),
        "decision_changed_after_retry_rate": (
            sum(1 for record in retry_records if bool(record.get("decision_changed_after_retry")))
            / len(retry_records)
            if retry_records
            else None
        ),
        "objective_changed_after_retry_rate": None,
        "retry_offspring_segment_distribution": {
            str(segment): int(count)
            for segment, count in Counter(point_segments).items()
        },
        "warnings": list(distribution.get("warnings", []))
        + ["objective_changed_after_retry_requires_extra_evaluation"],
    }


def summarize_segment0_supply_funnel(
    *,
    current_objective_vectors: Sequence[Sequence[float]],
    lineage_records: Sequence[Mapping[str, Any]],
    next_front_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    bins: int = 6,
) -> dict[str, Any]:
    current_distribution = compute_segment_distribution(current_objective_vectors, directions, bins=bins)
    offspring_vectors = [
        list(record.get("offspring_objective") or [])
        for record in lineage_records
        if isinstance(record.get("offspring_objective"), list)
    ]
    offspring_distribution = compute_segment_distribution(offspring_vectors, directions, bins=bins)
    next_front_distribution = compute_segment_distribution(next_front_vectors, directions, bins=bins)
    offspring_segments = list(offspring_distribution.get("point_segments", []))

    segment0_offspring_records = [
        record
        for index, record in enumerate(lineage_records)
        if index < len(offspring_segments) and int(offspring_segments[index]) == 0
    ]
    segment0_nondominated_count = sum(
        1 for record in segment0_offspring_records if bool(record.get("offspring_is_nondominated"))
    )
    segment0_survivor_count = sum(
        1
        for record in segment0_offspring_records
        if bool(record.get("offspring_survived_next_generation"))
    )
    segment0_final_front_count = int(next_front_distribution.get("segment_counts", {}).get("0", 0))
    segment0_initial_count = int(current_distribution.get("segment_counts", {}).get("0", 0))
    segment0_offspring_count = len(segment0_offspring_records)

    return {
        "segment0_initial_count": segment0_initial_count,
        "segment0_offspring_count": segment0_offspring_count,
        "segment0_nondominated_count": segment0_nondominated_count,
        "segment0_survivor_count": segment0_survivor_count,
        "segment0_final_front_count": segment0_final_front_count,
        "segment0_retention_init_to_survivor": (
            float(segment0_survivor_count / segment0_initial_count)
            if segment0_initial_count > 0
            else None
        ),
        "segment0_retention_offspring_to_survivor": (
            float(segment0_survivor_count / segment0_offspring_count)
            if segment0_offspring_count > 0
            else None
        ),
        "segment0_retention_survivor_to_front": (
            float(segment0_final_front_count / segment0_survivor_count)
            if segment0_survivor_count > 0
            else None
        ),
        "warnings": list(current_distribution.get("warnings", []))
        + list(offspring_distribution.get("warnings", []))
        + list(next_front_distribution.get("warnings", [])),
    }


def summarize_internal_external_distribution_comparison(
    objective_vectors: Sequence[Sequence[float]],
    reference_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    candidate_distribution = compute_segment_distribution(objective_vectors, directions, bins=bins)
    reference_distribution = compute_segment_distribution(reference_vectors, directions, bins=bins)
    diff = compare_segment_distributions(candidate_distribution, reference_distribution)

    candidate_spacing = _finite_or_none(spacing_metric(objective_vectors, directions))
    reference_spacing = _finite_or_none(spacing_metric(reference_vectors, directions))
    candidate_segment0 = summarize_segment0_spacing_detail(objective_vectors, directions, bins=bins)
    reference_segment0 = summarize_segment0_spacing_detail(reference_vectors, directions, bins=bins)
    candidate_normalized = _to_minimization(objective_vectors, directions) if objective_vectors else []
    reference_normalized = _to_minimization(reference_vectors, directions) if reference_vectors else []

    def _column_extrema(vectors: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
        if not vectors:
            return [], []
        minima = [float(min(vector[index] for vector in vectors)) for index in range(len(vectors[0]))]
        maxima = [float(max(vector[index] for vector in vectors)) for index in range(len(vectors[0]))]
        return minima, maxima

    candidate_min, candidate_max = _column_extrema(candidate_normalized)
    reference_min, reference_max = _column_extrema(reference_normalized)
    return {
        "segment_occupancy_diff": diff.get("segment_occupancy_diff", {}),
        "objective_min_diff": [
            float(candidate - reference)
            for candidate, reference in zip(candidate_min, reference_min, strict=True)
        ]
        if candidate_min and reference_min
        else [],
        "objective_max_diff": [
            float(candidate - reference)
            for candidate, reference in zip(candidate_max, reference_max, strict=True)
        ]
        if candidate_max and reference_max
        else [],
        "spacing_segment_diff": _safe_delta(
            reference_spacing,
            candidate_spacing,
        ),
        "nondominated_count_diff": float(len(objective_vectors) - len(reference_vectors)),
        "segment0_count_diff": int(diff.get("segment0_count_diff", 0)),
        "boundary_count_diff": int(diff.get("boundary_count_diff", 0)),
        "candidate_segment0_count": int(candidate_distribution.get("segment_counts", {}).get("0", 0)),
        "reference_segment0_count": int(reference_distribution.get("segment_counts", {}).get("0", 0)),
        "candidate_occupied_bins": int(candidate_distribution.get("occupied_segments", 0)),
        "reference_occupied_bins": int(reference_distribution.get("occupied_segments", 0)),
        "candidate_spacing_signal": candidate_spacing,
        "reference_spacing_signal": reference_spacing,
        "candidate_segment0_local_gap": candidate_segment0.get("max_local_gap"),
        "reference_segment0_local_gap": reference_segment0.get("max_local_gap"),
        "warnings": list(candidate_distribution.get("warnings", []))
        + list(reference_distribution.get("warnings", []))
        + list(candidate_segment0.get("warnings", []))
        + list(reference_segment0.get("warnings", [])),
    }


def _trace_entry(
    *,
    run_id: str,
    algorithm: str,
    candidate_id: str | None,
    problem: str,
    seed: int | None,
    generation: int,
    population_size: int,
    evaluations_so_far: int,
    trace_type: str,
    metrics: dict[str, Any],
    warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "candidate_id": candidate_id,
        "problem": problem,
        "seed": seed,
        "generation": generation,
        "population_size": population_size,
        "evaluations_so_far": evaluations_so_far,
        "trace_type": trace_type,
        "metrics": metrics,
        "warnings": list(warnings),
    }


def _aggregate_trace_entries(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace["trace_type"])].append(trace)

    aggregate: dict[str, Any] = {
        "generation_count": len({int(trace["generation"]) for trace in traces}),
        "trace_types": sorted(grouped),
    }
    for trace_type, trace_rows in grouped.items():
        aggregate_metrics: dict[str, Any] = {
            "trace_count": len(trace_rows),
            "generations": sorted(int(trace["generation"]) for trace in trace_rows),
        }
        numeric_values: dict[str, list[float]] = defaultdict(list)
        warnings: set[str] = set()
        for trace in trace_rows:
            for warning in trace.get("warnings", []):
                warnings.add(str(warning))
            for key, value in trace["metrics"].items():
                if isinstance(value, int | float) and math.isfinite(float(value)):
                    numeric_values[key].append(float(value))
        for key, values in numeric_values.items():
            aggregate_metrics[f"{key}_mean"] = _safe_mean(values)
            aggregate_metrics[f"{key}_median"] = _safe_median(values)
        aggregate_metrics["warnings"] = sorted(warnings)
        aggregate[trace_type] = aggregate_metrics
    return aggregate


@dataclass(frozen=True, slots=True)
class Nsga2DeepDiagnosticsConfig:
    deep_trace_enabled: bool = False
    generation_sample_stride: int = 1


@dataclass(frozen=True, slots=True)
class Nsga2LineageDiagnosticsConfig:
    lineage_trace_enabled: bool = False


@dataclass(frozen=True, slots=True)
class Nsga2OperatorSupplyDiagnosticsConfig:
    operator_supply_trace_enabled: bool = False
    segment_count: int = 6


@dataclass(frozen=True, slots=True)
class Nsga2Zdt1ComponentDiagnosticsConfig:
    zdt1_component_trace_enabled: bool = False


@dataclass(frozen=True, slots=True)
class Nsga2DiagnosticsConfig:
    trace_enabled: bool = False
    run_id: str = "nsga2_diagnostics_disabled"
    occupancy_bins: int = 6
    top_parent_limit: int = 5
    deep: Nsga2DeepDiagnosticsConfig = field(default_factory=Nsga2DeepDiagnosticsConfig)
    lineage: Nsga2LineageDiagnosticsConfig = field(
        default_factory=Nsga2LineageDiagnosticsConfig
    )
    operator_supply: Nsga2OperatorSupplyDiagnosticsConfig = field(
        default_factory=Nsga2OperatorSupplyDiagnosticsConfig
    )
    zdt1_component: Nsga2Zdt1ComponentDiagnosticsConfig = field(
        default_factory=Nsga2Zdt1ComponentDiagnosticsConfig
    )

    @classmethod
    def from_algorithm_options(
        cls,
        options: Mapping[str, object],
        *,
        default_run_id: str,
    ) -> Nsga2DiagnosticsConfig:
        return cls(
            trace_enabled=bool(options.get("nsga2_trace_enabled", False)),
            run_id=str(options.get("nsga2_trace_run_id", default_run_id)),
            occupancy_bins=_positive_int(options.get("nsga2_trace_occupancy_bins"), 6),
            top_parent_limit=_positive_int(options.get("nsga2_trace_top_parent_limit"), 5),
            deep=Nsga2DeepDiagnosticsConfig(
                deep_trace_enabled=bool(options.get("nsga2_deep_trace_enabled", False)),
                generation_sample_stride=_positive_int(
                    options.get("nsga2_trace_generation_sample_stride"),
                    1,
                ),
            ),
            lineage=Nsga2LineageDiagnosticsConfig(
                lineage_trace_enabled=bool(
                    options.get("nsga2_lineage_trace_enabled", False)
                ),
            ),
            operator_supply=Nsga2OperatorSupplyDiagnosticsConfig(
                operator_supply_trace_enabled=bool(
                    options.get("nsga2_operator_supply_trace_enabled", False)
                ),
                segment_count=_positive_int(
                    options.get("nsga2_trace_segment_count"),
                    _positive_int(options.get("nsga2_trace_occupancy_bins"), 6),
                ),
            ),
            zdt1_component=Nsga2Zdt1ComponentDiagnosticsConfig(
                zdt1_component_trace_enabled=bool(
                    options.get("nsga2_zdt1_component_trace_enabled", False)
                ),
            ),
        )


@dataclass(slots=True)
class Nsga2DiagnosticsRecorder:
    config: Nsga2DiagnosticsConfig
    algorithm: str
    problem: str
    seed: int | None = None
    candidate_id: str | None = None
    traces: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _generation_context: dict[str, Any] | None = None
    _parent_events: list[dict[str, Any]] = field(default_factory=list)
    _offspring_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def trace_enabled(self) -> bool:
        return self.config.trace_enabled

    @property
    def deep_trace_enabled(self) -> bool:
        return self.trace_enabled and self.config.deep.deep_trace_enabled

    @property
    def lineage_trace_enabled(self) -> bool:
        return self.trace_enabled and self.config.lineage.lineage_trace_enabled

    @property
    def operator_supply_trace_enabled(self) -> bool:
        return (
            self.trace_enabled
            and self.config.operator_supply.operator_supply_trace_enabled
        )

    @property
    def zdt1_component_trace_requested(self) -> bool:
        return (
            self.trace_enabled
            and self.config.zdt1_component.zdt1_component_trace_enabled
        )

    @property
    def zdt1_component_trace_enabled(self) -> bool:
        return self.zdt1_component_trace_requested and self.problem.strip().lower() == "zdt1"

    @property
    def parent_event_count(self) -> int:
        return len(self._parent_events)

    def parent_events_since(self, offset: int) -> list[dict[str, Any]]:
        if offset < 0:
            offset = 0
        return [dict(event) for event in self._parent_events[offset:]]

    def begin_generation(
        self,
        *,
        generation: int,
        population_size: int,
        population: Sequence[Sequence[float | int]],
        objective_vectors: Sequence[Sequence[float]],
        fronts: Sequence[Sequence[int]],
        ranks: Sequence[int],
        crowding: Sequence[float],
        directions: Sequence[bool],
        evaluations_so_far: int,
    ) -> None:
        if not self.trace_enabled:
            return
        current_front = [int(index) for index in (fronts[0] if fronts else [])]
        current_front_vectors = [list(objective_vectors[index]) for index in current_front]
        boundary_signatures, _ = _boundary_signature_details(current_front_vectors, directions)
        population_occupancy = _occupancy_snapshot(
            objective_vectors,
            directions,
            bins=self.config.occupancy_bins,
        )
        point_bins = [tuple(value) for value in population_occupancy.get("point_bins", [])]
        bin_counts = Counter(point_bins)
        sparse_threshold = int(population_occupancy.get("sparse_threshold", 1))
        vector_signatures = [_signature(vector) for vector in objective_vectors]
        decision_hashes = [hash_solution_or_objective(genome) for genome in population]
        objective_hashes = [hash_solution_or_objective(vector) for vector in objective_vectors]
        sampled_trace_enabled = (
            self.deep_trace_enabled
            or self.lineage_trace_enabled
            or self.operator_supply_trace_enabled
            or self.zdt1_component_trace_requested
        ) and (
            generation % max(1, self.config.deep.generation_sample_stride) == 0
        )

        warnings = list(population_occupancy.get("warnings", []))
        if self.zdt1_component_trace_requested and self.problem.strip().lower() != "zdt1":
            warnings.append("zdt1_component_trace_skipped_non_zdt1_problem")
            if "zdt1_component_trace_skipped_non_zdt1_problem" not in self.warnings:
                self.warnings.append("zdt1_component_trace_skipped_non_zdt1_problem")

        self._generation_context = {
            "generation": generation,
            "population_size": population_size,
            "evaluations_so_far": evaluations_so_far,
            "directions": [bool(value) for value in directions],
            "ranks": [int(value) for value in ranks],
            "crowding": [float(value) for value in crowding],
            "current_front_vectors": current_front_vectors,
            "boundary_signatures": boundary_signatures,
            "point_bins": point_bins,
            "bin_counts": bin_counts,
            "sparse_threshold": sparse_threshold,
            "vector_signatures": vector_signatures,
            "decision_hashes": decision_hashes,
            "objective_hashes": objective_hashes,
            "population": [list(genome) for genome in population],
            "objective_vectors": [list(vector) for vector in objective_vectors],
            "current_front_indices": current_front,
            "current_front_ranks": [int(ranks[index]) for index in current_front],
            "current_front_crowding": [float(crowding[index]) for index in current_front],
            "deep_generation_enabled": self.deep_trace_enabled and sampled_trace_enabled,
            "lineage_generation_enabled": self.lineage_trace_enabled
            and sampled_trace_enabled,
            "operator_supply_generation_enabled": self.operator_supply_trace_enabled
            and sampled_trace_enabled,
            "zdt1_component_generation_enabled": self.zdt1_component_trace_enabled
            and sampled_trace_enabled,
            "sampled_trace_enabled": sampled_trace_enabled,
            "warnings": warnings,
        }
        self._parent_events = []
        self._offspring_records = []

    def record_parent_selection(
        self,
        *,
        selection_kind: str,
        winner_index: int,
        candidate_indices: Sequence[int],
        reference_distance: float | None = None,
        bias_applied: bool = False,
    ) -> None:
        if not self.trace_enabled or self._generation_context is None:
            return
        context = self._generation_context
        ranks = context["ranks"]
        crowding = context["crowding"]
        candidate_ranks = [int(ranks[index]) for index in candidate_indices]
        finite_candidate_crowding = [
            round(float(crowding[index]), 12)
            for index in candidate_indices
            if math.isfinite(float(crowding[index]))
        ]
        point_bins = context["point_bins"]
        bin_counts: Counter[tuple[int, ...]] = context["bin_counts"]
        winner_bin = point_bins[winner_index] if winner_index < len(point_bins) else None
        winner_signature = context["vector_signatures"][winner_index]
        is_boundary = winner_signature in context["boundary_signatures"]
        is_sparse = (
            winner_bin is not None and bin_counts.get(winner_bin, 0) <= context["sparse_threshold"]
        )
        winner_zdt1_components = None
        if self.zdt1_component_trace_enabled:
            population = context.get("population", [])
            objective_vectors = context.get("objective_vectors", [])
            if winner_index < len(population) and winner_index < len(objective_vectors):
                winner_zdt1_components = compute_zdt1_components(
                    population[winner_index],
                    objective_vectors[winner_index],
                    bins=max(1, int(self.config.operator_supply.segment_count)),
                )
        self._parent_events.append(
            {
                "selection_kind": selection_kind,
                "winner_index": int(winner_index),
                "winner_rank": int(ranks[winner_index]),
                "winner_crowding": float(crowding[winner_index]),
                "is_boundary": is_boundary,
                "is_sparse": is_sparse,
                "sample_same_rank": len(set(candidate_ranks)) == 1,
                "sample_crowding_tie": len(finite_candidate_crowding)
                != len(set(finite_candidate_crowding)),
                "candidate_pool_size": len(candidate_indices),
                "bias_applied": bool(bias_applied),
                "reference_distance": _finite_or_none(reference_distance),
                "winner_decision_hash": context["decision_hashes"][winner_index],
                "winner_objective_hash": context["objective_hashes"][winner_index],
                "winner_sparse_bin": list(winner_bin) if winner_bin is not None else None,
                "winner_zdt1_components": winner_zdt1_components,
            }
        )

    def record_offspring_creation(
        self,
        *,
        offspring_genome: Sequence[float | int],
        parent_events: Sequence[Mapping[str, Any]],
        variation_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            not (
                self.deep_trace_enabled
                or self.lineage_trace_enabled
                or self.operator_supply_trace_enabled
                or self.zdt1_component_trace_enabled
            )
            or self._generation_context is None
        ):
            return
        if not bool(self._generation_context.get("sampled_trace_enabled")):
            return
        parent_edges = [
            {
                "parent_id": event.get("winner_decision_hash"),
                "parent_objective_hash": event.get("winner_objective_hash"),
                "parent_rank": event.get("winner_rank"),
                "parent_crowding": _finite_or_none(event.get("winner_crowding")),
                "parent_is_boundary": bool(event.get("is_boundary")),
                "parent_sparse_bin": event.get("winner_sparse_bin"),
                "parent_is_sparse": bool(event.get("is_sparse")),
                "selection_kind": event.get("selection_kind"),
                "bias_applied": bool(event.get("bias_applied")),
                "parent_zdt1_components": (
                    dict(event.get("winner_zdt1_components") or {})
                    if isinstance(event.get("winner_zdt1_components"), Mapping)
                    else None
                ),
            }
            for event in parent_events
        ]
        self._offspring_records.append(
            {
                "offspring_index": len(self._offspring_records),
                "offspring_decision_hash": hash_solution_or_objective(offspring_genome),
                "offspring_decision_components": summarize_decision_component_summary(
                    offspring_genome
                ),
                "parent_edges": parent_edges,
                "variation_metadata": (
                    dict(variation_metadata) if variation_metadata is not None else {}
                ),
            }
        )

    def _build_lineage_records(
        self,
        *,
        base_population_size: int,
        combined_objective_vectors: Sequence[Sequence[float]],
        combined_ranks: Sequence[int],
        survivor_indices: Sequence[int],
        directions: Sequence[bool],
    ) -> list[dict[str, Any]]:
        if not self._offspring_records:
            return []
        offspring_vectors = [
            list(vector)
            for vector in combined_objective_vectors[
                base_population_size : base_population_size + len(self._offspring_records)
            ]
        ]
        occupancy = compute_objective_bins(
            offspring_vectors,
            directions,
            bins=self.config.occupancy_bins,
        )
        point_bins = [tuple(value) for value in occupancy.get("point_bins", [])]
        bin_counts = Counter(point_bins)
        sparse_threshold = int(occupancy.get("sparse_threshold", 1))
        boundary_signatures, _ = _boundary_signature_details(offspring_vectors, directions)
        survivor_set = {int(index) for index in survivor_indices}

        lineage_records: list[dict[str, Any]] = []
        for local_index, record in enumerate(self._offspring_records):
            combined_index = base_population_size + local_index
            objective_vector = offspring_vectors[local_index] if local_index < len(offspring_vectors) else []
            objective_signature = _signature(objective_vector) if objective_vector else None
            bin_key = point_bins[local_index] if local_index < len(point_bins) else None
            variation_metadata = dict(record.get("variation_metadata") or {})
            offspring_decision_components = dict(record.get("offspring_decision_components") or {})
            initial_retry_components = dict(
                variation_metadata.get("initial_decision_components") or {}
            )
            final_retry_components = dict(
                variation_metadata.get("final_decision_components") or offspring_decision_components
            )
            offspring_is_nondominated = (
                combined_index < len(combined_ranks) and int(combined_ranks[combined_index]) == 0
            )
            offspring_survived = combined_index in survivor_set
            zdt1_components = None
            retry_initial_zdt1_components = None
            retry_final_zdt1_components = None
            if self.zdt1_component_trace_enabled:
                zdt1_components = _zdt1_components_from_summary(
                    offspring_decision_components,
                    objective_vector=objective_vector,
                    bins=max(1, int(self.config.operator_supply.segment_count)),
                    nondominated=offspring_is_nondominated,
                    survived=offspring_survived,
                )
                if initial_retry_components:
                    retry_initial_zdt1_components = _zdt1_components_from_summary(
                        initial_retry_components,
                        bins=max(1, int(self.config.operator_supply.segment_count)),
                    )
                if final_retry_components:
                    retry_final_zdt1_components = _zdt1_components_from_summary(
                        final_retry_components,
                        objective_vector=objective_vector,
                        bins=max(1, int(self.config.operator_supply.segment_count)),
                        nondominated=offspring_is_nondominated,
                        survived=offspring_survived,
                    )
            lineage_records.append(
                {
                    **record,
                    "offspring_objective": objective_vector,
                    "offspring_objective_hash": (
                        hash_solution_or_objective(objective_vector) if objective_vector else None
                    ),
                    "offspring_rank_after_evaluation": (
                        int(combined_ranks[combined_index])
                        if combined_index < len(combined_ranks)
                        else None
                    ),
                    "offspring_is_nondominated": offspring_is_nondominated,
                    "offspring_survived_next_generation": offspring_survived,
                    "offspring_is_sparse": (
                        bin_key is not None and bin_counts.get(bin_key, 0) <= sparse_threshold
                    ),
                    "offspring_is_boundary": (
                        objective_signature in boundary_signatures
                        if objective_signature is not None
                        else False
                    ),
                    "offspring_bin": list(bin_key) if bin_key is not None else None,
                    "variation_metadata": variation_metadata,
                    "offspring_decision_components": offspring_decision_components,
                    "offspring_zdt1_components": zdt1_components,
                    "retry_initial_zdt1_components": retry_initial_zdt1_components,
                    "retry_final_zdt1_components": retry_final_zdt1_components,
                    "duplicate_detected": bool(
                        variation_metadata.get("duplicate_detected")
                    ),
                    "retry_attempt_count": int(
                        variation_metadata.get("retry_attempt_count", 0) or 0
                    ),
                    "retry_success": bool(variation_metadata.get("retry_success")),
                    "retry_reinitialized": bool(variation_metadata.get("retry_reinitialized")),
                    "decision_changed_after_retry": bool(
                        variation_metadata.get("decision_changed_after_retry")
                    ),
                }
            )
        return lineage_records

    def record_generation_transition(
        self,
        *,
        base_population_size: int,
        next_population_size: int,
        next_population: Sequence[Sequence[float | int]],
        next_objective_vectors: Sequence[Sequence[float]],
        combined_population: Sequence[Sequence[float | int]],
        combined_fronts: Sequence[Sequence[int]],
        combined_ranks: Sequence[int],
        combined_crowding: Sequence[float],
        combined_objective_vectors: Sequence[Sequence[float]],
        survivor_indices: Sequence[int],
        partial_front_strategy: str,
        partial_front_dedup_mode: str,
        evaluations_so_far: int,
    ) -> None:
        if not self.trace_enabled or self._generation_context is None:
            return

        context = self._generation_context
        directions = [bool(value) for value in context["directions"]]
        current_front_vectors = list(context["current_front_vectors"])
        next_front_vectors = nondominated_vectors(next_objective_vectors, directions)
        lineage_records = self._build_lineage_records(
            base_population_size=base_population_size,
            combined_objective_vectors=combined_objective_vectors,
            combined_ranks=combined_ranks,
            survivor_indices=survivor_indices,
            directions=directions,
        )

        trace_specs = [
            (
                "parent_contribution_trace",
                summarize_parent_contributions(
                    self._parent_events,
                    population_size=int(context["population_size"]),
                    top_parent_limit=self.config.top_parent_limit,
                ),
            ),
            (
                "survivor_replacement_trace",
                summarize_survivor_replacement(
                    population_size=next_population_size,
                    combined_fronts=combined_fronts,
                    combined_ranks=combined_ranks,
                    combined_crowding=combined_crowding,
                    combined_objective_vectors=combined_objective_vectors,
                    survivor_indices=survivor_indices,
                    directions=directions,
                    partial_front_strategy=partial_front_strategy,
                    partial_front_dedup_mode=partial_front_dedup_mode,
                    bins=self.config.occupancy_bins,
                ),
            ),
            (
                "boundary_retention_trace",
                summarize_boundary_retention(
                    current_front_vectors,
                    next_front_vectors,
                    directions,
                ),
            ),
            (
                "rank_crowding_summary",
                summarize_rank_crowding(combined_ranks, combined_crowding),
            ),
            (
                "objective_occupancy_summary",
                summarize_objective_occupancy(
                    next_front_vectors,
                    directions,
                    bins=self.config.occupancy_bins,
                ),
            ),
            (
                "front_change_summary",
                summarize_front_change(
                    current_front_vectors,
                    next_front_vectors,
                    directions,
                ),
            ),
        ]
        if self.deep_trace_enabled and bool(context.get("deep_generation_enabled")):
            trace_specs.extend(
                [
                    (
                        "parent_to_offspring_trace",
                        summarize_parent_to_offspring(lineage_records),
                    ),
                    (
                        "offspring_to_survivor_trace",
                        summarize_offspring_to_survivor(
                            lineage_records,
                            directions=directions,
                            bins=self.config.occupancy_bins,
                        ),
                    ),
                    (
                        "objective_boundary_retention_detail",
                        summarize_objective_boundary_retention_detail(
                            current_front_vectors,
                            next_front_vectors,
                            directions,
                            previous_front_ranks=context.get("current_front_ranks"),
                            previous_front_crowding=context.get("current_front_crowding"),
                        ),
                    ),
                    (
                        "segment_spacing_attribution",
                        summarize_segment_spacing_attribution(
                            next_front_vectors,
                            directions,
                            bins=self.config.occupancy_bins,
                        ),
                    ),
                    (
                        "crowding_decision_attribution",
                        summarize_crowding_decision_attribution(
                            population_size=next_population_size,
                            combined_fronts=combined_fronts,
                            combined_ranks=combined_ranks,
                            combined_crowding=combined_crowding,
                            survivor_indices=survivor_indices,
                        ),
                    ),
                ]
            )
        if self.lineage_trace_enabled and bool(context.get("lineage_generation_enabled")):
            trace_specs.extend(
                [
                    (
                        "lineage_retention_funnel",
                        summarize_lineage_retention_funnel(
                            lineage_records,
                            next_front_vectors=next_front_vectors,
                        ),
                    ),
                    (
                        "sparse_lineage_quality",
                        summarize_sparse_lineage_quality(
                            lineage_records,
                            directions=directions,
                            next_front_vectors=next_front_vectors,
                            bins=self.config.occupancy_bins,
                        ),
                    ),
                    (
                        "segment0_spacing_detail",
                        summarize_segment0_spacing_detail(
                            next_front_vectors,
                            directions,
                            bins=self.config.occupancy_bins,
                        ),
                    ),
                    (
                        "duplicate_to_diversity_funnel",
                        summarize_duplicate_to_diversity_funnel(
                            current_population=context.get("population", []),
                            current_objective_vectors=context.get("objective_vectors", []),
                            next_population=next_population,
                            next_objective_vectors=next_objective_vectors,
                            next_front_vectors=next_front_vectors,
                            lineage_records=lineage_records,
                            directions=directions,
                            bins=self.config.occupancy_bins,
                        ),
                    ),
                    (
                        "boundary_intervention_count",
                        summarize_boundary_intervention_count(
                            population_size=next_population_size,
                            combined_fronts=combined_fronts,
                            combined_crowding=combined_crowding,
                            combined_population=combined_population,
                            combined_objective_vectors=combined_objective_vectors,
                            partial_front_strategy=partial_front_strategy,
                            partial_front_dedup_mode=partial_front_dedup_mode,
                        ),
                    ),
                ]
            )
        if self.operator_supply_trace_enabled and bool(
            context.get("operator_supply_generation_enabled")
        ):
            operator_bins = max(1, int(self.config.operator_supply.segment_count))
            if int(context.get("generation", -1)) == 0:
                trace_specs.append(
                    (
                        "initialization_segment_coverage",
                        summarize_initialization_segment_coverage(
                            context.get("objective_vectors", []),
                            directions,
                            bins=operator_bins,
                        ),
                    )
                )
            trace_specs.extend(
                [
                    (
                        "variation_segment_transition",
                        summarize_variation_segment_transition(
                            lineage_records,
                            directions,
                            bins=operator_bins,
                        ),
                    ),
                    (
                        "operator_offspring_quality",
                        summarize_operator_offspring_quality(
                            lineage_records,
                            directions=directions,
                            next_front_vectors=next_front_vectors,
                            bins=operator_bins,
                        ),
                    ),
                    (
                        "mutation_retry_objective_effect",
                        summarize_mutation_retry_objective_effect(
                            lineage_records,
                            directions,
                            bins=operator_bins,
                        ),
                    ),
                    (
                        "segment0_supply_funnel",
                        summarize_segment0_supply_funnel(
                            current_objective_vectors=context.get("objective_vectors", []),
                            lineage_records=lineage_records,
                            next_front_vectors=next_front_vectors,
                            directions=directions,
                            bins=operator_bins,
                        ),
                    ),
                ]
            )
        if self.zdt1_component_trace_enabled and bool(
            context.get("zdt1_component_generation_enabled")
        ):
            zdt1_bins = max(1, int(self.config.operator_supply.segment_count))
            if int(context.get("generation", -1)) == 0:
                trace_specs.append(
                    (
                        "zdt1_initial_component_coverage",
                        summarize_zdt1_initial_component_coverage(
                            context.get("population", []),
                            context.get("objective_vectors", []),
                            bins=zdt1_bins,
                        ),
                    )
                )
            trace_specs.extend(
                [
                    (
                        "zdt1_offspring_component_quality",
                        summarize_zdt1_offspring_component_quality(lineage_records),
                    ),
                    (
                        "zdt1_parent_child_component_delta",
                        summarize_zdt1_parent_child_component_delta(lineage_records),
                    ),
                    (
                        "zdt1_mutation_retry_component_effect",
                        summarize_zdt1_mutation_retry_component_effect(lineage_records),
                    ),
                    (
                        "zdt1_segment0_quality_funnel",
                        summarize_zdt1_segment0_quality_funnel(
                            current_population=context.get("population", []),
                            current_objective_vectors=context.get("objective_vectors", []),
                            lineage_records=lineage_records,
                            next_population=next_population,
                            next_objective_vectors=next_objective_vectors,
                            next_front_vectors=next_front_vectors,
                            bins=zdt1_bins,
                        ),
                    ),
                ]
            )

        for trace_type, summary in trace_specs:
            warnings = list(context.get("warnings", [])) + list(summary.get("warnings", []))
            self.traces.append(
                _trace_entry(
                    run_id=self.config.run_id,
                    algorithm=self.algorithm,
                    candidate_id=self.candidate_id,
                    problem=self.problem,
                    seed=self.seed,
                    generation=int(context["generation"]),
                    population_size=int(context["population_size"]),
                    evaluations_so_far=evaluations_so_far,
                    trace_type=trace_type,
                    metrics={key: value for key, value in summary.items() if key != "warnings"},
                    warnings=warnings,
                )
            )

        self._generation_context = None
        self._parent_events = []
        self._offspring_records = []

    def build_payload(self) -> dict[str, Any] | None:
        if not self.trace_enabled:
            return None
        return {
            "run_id": self.config.run_id,
            "trace_enabled": True,
            "algorithm": self.algorithm,
            "candidate_id": self.candidate_id,
            "problem": self.problem,
            "seed": self.seed,
            "trace_config": {
                "occupancy_bins": self.config.occupancy_bins,
                "top_parent_limit": self.config.top_parent_limit,
                "deep_trace_enabled": self.config.deep.deep_trace_enabled,
                "lineage_trace_enabled": self.config.lineage.lineage_trace_enabled,
                "operator_supply_trace_enabled": (
                    self.config.operator_supply.operator_supply_trace_enabled
                ),
                "zdt1_component_trace_enabled": (
                    self.config.zdt1_component.zdt1_component_trace_enabled
                ),
                "segment_count": self.config.operator_supply.segment_count,
                "generation_sample_stride": self.config.deep.generation_sample_stride,
            },
            "traces": self.traces,
            "aggregate": _aggregate_trace_entries(self.traces),
            "warnings": sorted(set(self.warnings)),
        }
