from __future__ import annotations

import math
from collections.abc import Sequence


def finite_or_none(value: float) -> float | None:
    if math.isfinite(value):
        return float(value)
    return None


def resolve_reference_point(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    explicit: Sequence[float] | None = None,
) -> list[float]:
    if explicit is not None:
        if len(explicit) != len(directions):
            raise ValueError("reference point length must match objective directions")
        return [float(value) for value in explicit]

    if not objective_vectors:
        raise ValueError("objective_vectors cannot be empty")
    objective_len = len(objective_vectors[0])
    if len(directions) != objective_len:
        raise ValueError("objective_directions length must match objective count")

    reference: list[float] = []
    for obj_idx in range(objective_len):
        values = [float(vector[obj_idx]) for vector in objective_vectors]
        low = min(values)
        high = max(values)
        span = high - low
        margin = span * 0.1
        if margin <= 0.0:
            margin = max(abs(low), abs(high), 1.0) * 0.1
        if directions[obj_idx]:
            reference.append(low - margin)
        else:
            reference.append(high + margin)
    return reference


def _to_minimization(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> list[list[float]]:
    return [
        [(-value if directions[idx] else value) for idx, value in enumerate(vector)]
        for vector in objective_vectors
    ]


def _filter_nondominated_2d(points: Sequence[Sequence[float]]) -> list[list[float]]:
    nondominated: list[list[float]] = []
    best_y = math.inf
    for x, y in sorted(points, key=lambda point: (point[0], point[1])):
        if y < best_y:
            nondominated.append([float(x), float(y)])
            best_y = float(y)
    return nondominated


def hypervolume_2d(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    reference_point: Sequence[float],
) -> float:
    if len(directions) != 2 or len(reference_point) != 2:
        return math.nan
    if not objective_vectors:
        return 0.0

    points = _to_minimization(objective_vectors, directions)
    reference = _to_minimization([reference_point], directions)[0]
    bounded_points = [
        point for point in points if point[0] <= reference[0] and point[1] <= reference[1]
    ]
    if not bounded_points:
        return 0.0

    hypervolume = 0.0
    previous_y = reference[1]
    for x, y in _filter_nondominated_2d(bounded_points):
        width = max(0.0, reference[0] - x)
        height = max(0.0, previous_y - y)
        hypervolume += width * height
        previous_y = min(previous_y, y)
    return hypervolume


def spread_metric(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> float:
    if not objective_vectors:
        return math.nan
    if len(directions) != 2:
        return math.nan

    points = _filter_nondominated_2d(_to_minimization(objective_vectors, directions))
    if len(points) < 3:
        return math.nan

    distances = [math.dist(points[idx], points[idx + 1]) for idx in range(len(points) - 1)]
    mean_distance = sum(distances) / len(distances)
    if mean_distance == 0.0:
        return 0.0
    return sum(abs(distance - mean_distance) for distance in distances) / (
        len(distances) * mean_distance
    )


def front_metrics(
    front_indices: Sequence[int],
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    reference_point: Sequence[float],
    population_size: int,
) -> dict[str, float]:
    front_vectors = [list(objective_vectors[idx]) for idx in front_indices]
    pareto_front_size = float(len(front_vectors))
    pareto_ratio = pareto_front_size / float(population_size) if population_size > 0 else 0.0
    return {
        "pareto_front_size": pareto_front_size,
        "pareto_ratio": pareto_ratio,
        "hypervolume": hypervolume_2d(front_vectors, directions, reference_point),
        "spread": spread_metric(front_vectors, directions),
    }
