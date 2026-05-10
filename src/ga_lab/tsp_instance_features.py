from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TSPInstanceFeatures:
    instance_id: str
    num_cities: int
    bbox_aspect_ratio: float
    pca_anisotropy_ratio: float
    nn_distance_mean: float
    nn_distance_std: float
    nn_distance_cv: float
    mst_total_length: float
    bridge_score: float
    edge_length_cv: float
    radial_distance_mean: float
    radial_distance_std: float
    radial_distance_cv: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "instance_id": self.instance_id,
            "num_cities": self.num_cities,
            "bbox_aspect_ratio": self.bbox_aspect_ratio,
            "pca_anisotropy_ratio": self.pca_anisotropy_ratio,
            "nn_distance_mean": self.nn_distance_mean,
            "nn_distance_std": self.nn_distance_std,
            "nn_distance_cv": self.nn_distance_cv,
            "mst_total_length": self.mst_total_length,
            "bridge_score": self.bridge_score,
            "edge_length_cv": self.edge_length_cv,
            "radial_distance_mean": self.radial_distance_mean,
            "radial_distance_std": self.radial_distance_std,
            "radial_distance_cv": self.radial_distance_cv,
        }


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return mean_value, math.sqrt(max(0.0, variance))


def _distance_matrix(coordinates: list[tuple[float, float]]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for x1, y1 in coordinates:
        row = [math.hypot(x2 - x1, y2 - y1) for x2, y2 in coordinates]
        matrix.append(row)
    return matrix


def _nearest_neighbor_distances(matrix: list[list[float]]) -> list[float]:
    distances: list[float] = []
    for row_index, row in enumerate(matrix):
        nearest = min(distance for idx, distance in enumerate(row) if idx != row_index)
        distances.append(nearest)
    return distances


def _upper_triangle_distances(matrix: list[list[float]]) -> list[float]:
    values: list[float] = []
    for row_index, row in enumerate(matrix):
        for column_index in range(row_index + 1, len(row)):
            values.append(row[column_index])
    return values


def _mst_edge_lengths(matrix: list[list[float]]) -> list[float]:
    size = len(matrix)
    in_tree = [False] * size
    in_tree[0] = True
    edges: list[float] = []
    for _ in range(size - 1):
        best_distance: float | None = None
        best_target: int | None = None
        for source_index, source_used in enumerate(in_tree):
            if not source_used:
                continue
            for target_index, target_used in enumerate(in_tree):
                if target_used:
                    continue
                distance = matrix[source_index][target_index]
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_target = target_index
        if best_distance is None or best_target is None:
            break
        in_tree[best_target] = True
        edges.append(best_distance)
    return edges


def _pca_anisotropy_ratio(coordinates: list[tuple[float, float]]) -> float:
    if not coordinates:
        return 1.0
    mean_x = sum(x for x, _ in coordinates) / len(coordinates)
    mean_y = sum(y for _, y in coordinates) / len(coordinates)
    covariance_xx = sum((x - mean_x) ** 2 for x, _ in coordinates) / len(coordinates)
    covariance_yy = sum((y - mean_y) ** 2 for _, y in coordinates) / len(coordinates)
    covariance_xy = sum((x - mean_x) * (y - mean_y) for x, y in coordinates) / len(
        coordinates
    )
    trace = covariance_xx + covariance_yy
    determinant = (covariance_xx * covariance_yy) - (covariance_xy * covariance_xy)
    discriminant = math.sqrt(max(0.0, (trace * trace) - (4.0 * determinant)))
    largest = (trace + discriminant) / 2.0
    smallest = max((trace - discriminant) / 2.0, 1e-9)
    return max(1.0, largest / smallest)


def _radial_distances(coordinates: list[tuple[float, float]]) -> list[float]:
    if not coordinates:
        return []
    mean_x = sum(x for x, _ in coordinates) / len(coordinates)
    mean_y = sum(y for _, y in coordinates) / len(coordinates)
    return [math.hypot(x - mean_x, y - mean_y) for x, y in coordinates]


def extract_tsp_instance_features(
    instance_id: str,
    coordinates: list[tuple[float, float]] | list[list[float]],
) -> TSPInstanceFeatures:
    normalized_coordinates = [tuple(map(float, point)) for point in coordinates]
    if len(normalized_coordinates) <= 1:
        raise ValueError("TSP feature extraction requires at least two coordinates")

    xs = [point[0] for point in normalized_coordinates]
    ys = [point[1] for point in normalized_coordinates]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    bbox_aspect_ratio = max(x_span, y_span) / max(min(x_span, y_span), 1e-9)

    matrix = _distance_matrix(normalized_coordinates)
    nearest_neighbor_distances = _nearest_neighbor_distances(matrix)
    edge_lengths = _upper_triangle_distances(matrix)
    mst_edges = _mst_edge_lengths(matrix)
    radial_distances = _radial_distances(normalized_coordinates)

    nn_mean, nn_std = _mean_std(nearest_neighbor_distances)
    edge_mean, edge_std = _mean_std(edge_lengths)
    radial_mean, radial_std = _mean_std(radial_distances)
    bridge_baseline = sorted(mst_edges)[len(mst_edges) // 2] if mst_edges else 0.0

    return TSPInstanceFeatures(
        instance_id=instance_id,
        num_cities=len(normalized_coordinates),
        bbox_aspect_ratio=bbox_aspect_ratio,
        pca_anisotropy_ratio=_pca_anisotropy_ratio(normalized_coordinates),
        nn_distance_mean=nn_mean,
        nn_distance_std=nn_std,
        nn_distance_cv=_safe_div(nn_std, nn_mean),
        mst_total_length=sum(mst_edges),
        bridge_score=_safe_div(max(mst_edges, default=0.0), max(bridge_baseline, 1e-9)),
        edge_length_cv=_safe_div(edge_std, edge_mean),
        radial_distance_mean=radial_mean,
        radial_distance_std=radial_std,
        radial_distance_cv=_safe_div(radial_std, radial_mean),
    )


def extract_tsp_instance_features_from_problem_options(
    instance_id: str,
    problem_options: dict[str, Any],
) -> TSPInstanceFeatures:
    coordinates = problem_options.get("coordinates")
    if isinstance(coordinates, list) and coordinates:
        return extract_tsp_instance_features(instance_id, coordinates)

    from ga_lab.problems.tsp import TSPProblem

    problem = TSPProblem(**problem_options)
    if problem.coordinates is None:
        raise ValueError("TSP feature extraction requires coordinates, not distance matrix only")
    return extract_tsp_instance_features(instance_id, list(problem.coordinates))
