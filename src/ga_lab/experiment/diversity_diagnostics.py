from __future__ import annotations

import math
from statistics import median
from collections.abc import Sequence

from ga_lab.experiment.mo_metrics import dominates


def _signature(vector: Sequence[float | int], *, precision: int = 12) -> tuple[float, ...]:
    return tuple(round(float(value), precision) for value in vector)


def unique_vector_count(
    vectors: Sequence[Sequence[float | int]],
    *,
    precision: int = 12,
) -> int:
    if not vectors:
        return 0
    return len({_signature(vector, precision=precision) for vector in vectors})


def duplicate_rate(
    vectors: Sequence[Sequence[float | int]],
    *,
    precision: int = 12,
) -> float:
    if not vectors:
        return 0.0
    unique = unique_vector_count(vectors, precision=precision)
    duplicates = max(0, len(vectors) - unique)
    return duplicates / float(len(vectors))


def raw_nondominated_vectors(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> list[list[float]]:
    if not objective_vectors:
        return []
    nondominated: list[list[float]] = []
    for index, candidate in enumerate(objective_vectors):
        dominated = False
        for other_index, other in enumerate(objective_vectors):
            if index == other_index:
                continue
            if dominates(other, candidate, directions):
                dominated = True
                break
        if not dominated:
            nondominated.append([float(value) for value in candidate])
    return nondominated


def archive_duplicate_rate(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    precision: int = 12,
) -> float:
    return duplicate_rate(
        raw_nondominated_vectors(objective_vectors, directions),
        precision=precision,
    )


def boundary_point_count(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    precision: int = 12,
) -> int:
    front = raw_nondominated_vectors(objective_vectors, directions)
    if not front:
        return 0
    converted = [
        [(-float(value) if directions[index] else float(value)) for index, value in enumerate(vector)]
        for vector in front
    ]
    boundary_signatures: set[tuple[float, ...]] = set()
    objective_count = len(converted[0])
    for objective_index in range(objective_count):
        column = [vector[objective_index] for vector in converted]
        low = min(column)
        high = max(column)
        for original, converted_vector in zip(front, converted, strict=True):
            value = converted_vector[objective_index]
            if math.isclose(value, low, rel_tol=1e-12, abs_tol=1e-12) or math.isclose(
                value, high, rel_tol=1e-12, abs_tol=1e-12
            ):
                boundary_signatures.add(_signature(original, precision=precision))
    return len(boundary_signatures)


def crowding_distance_summary(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> dict[str, float | int | None]:
    front = raw_nondominated_vectors(objective_vectors, directions)
    if not front:
        return {
            "crowding_distance_finite_mean": None,
            "crowding_distance_finite_median": None,
            "crowding_distance_finite_min": None,
            "crowding_distance_finite_max": None,
            "crowding_distance_infinite_count": 0,
        }

    converted = [
        [(-float(value) if directions[index] else float(value)) for index, value in enumerate(vector)]
        for vector in front
    ]
    size = len(converted)
    if size == 1:
        return {
            "crowding_distance_finite_mean": None,
            "crowding_distance_finite_median": None,
            "crowding_distance_finite_min": None,
            "crowding_distance_finite_max": None,
            "crowding_distance_infinite_count": 1,
        }

    objective_count = len(converted[0])
    distances = [0.0] * size
    infinite = [False] * size
    for objective_index in range(objective_count):
        order = sorted(range(size), key=lambda idx: converted[idx][objective_index])
        infinite[order[0]] = True
        infinite[order[-1]] = True
        min_value = converted[order[0]][objective_index]
        max_value = converted[order[-1]][objective_index]
        scale = max_value - min_value
        if scale == 0.0:
            continue
        for left_idx, center_idx, right_idx in zip(order, order[1:], order[2:], strict=False):
            distances[center_idx] += (
                converted[right_idx][objective_index] - converted[left_idx][objective_index]
            ) / scale

    finite_distances = [distance for distance, inf in zip(distances, infinite, strict=True) if not inf]
    if not finite_distances:
        return {
            "crowding_distance_finite_mean": None,
            "crowding_distance_finite_median": None,
            "crowding_distance_finite_min": None,
            "crowding_distance_finite_max": None,
            "crowding_distance_infinite_count": sum(infinite),
        }
    return {
        "crowding_distance_finite_mean": sum(finite_distances) / len(finite_distances),
        "crowding_distance_finite_median": median(finite_distances),
        "crowding_distance_finite_min": min(finite_distances),
        "crowding_distance_finite_max": max(finite_distances),
        "crowding_distance_infinite_count": sum(infinite),
    }


def evaluate_diversity_diagnostics(
    objective_vectors: Sequence[Sequence[float]],
    *,
    directions: Sequence[bool],
    decision_vectors: Sequence[Sequence[float | int]] | None = None,
    precision: int = 12,
) -> dict[str, float | int | None]:
    diagnostics: dict[str, float | int | None] = {
        "decision_duplicate_rate": None,
        "objective_duplicate_rate": duplicate_rate(objective_vectors, precision=precision),
        "archive_duplicate_rate": archive_duplicate_rate(
            objective_vectors,
            directions,
            precision=precision,
        ),
        "unique_decision_count": None,
        "unique_objective_count": unique_vector_count(objective_vectors, precision=precision),
        "boundary_point_count": boundary_point_count(
            objective_vectors,
            directions,
            precision=precision,
        ),
    }
    if decision_vectors is not None:
        diagnostics["decision_duplicate_rate"] = duplicate_rate(
            decision_vectors,
            precision=precision,
        )
        diagnostics["unique_decision_count"] = unique_vector_count(
            decision_vectors,
            precision=precision,
        )
    diagnostics.update(crowding_distance_summary(objective_vectors, directions))
    return diagnostics
