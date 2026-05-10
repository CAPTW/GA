from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence


def validate_objective_vector(
    values: Sequence[float],
    *,
    label: str,
) -> None:
    for objective_index, value in enumerate(values):
        if math.isfinite(float(value)):
            continue
        raise ValueError(
            f"Non-finite objective detected in {label} "
            f"(objective_index={objective_index}, value={value!r})"
        )


def _to_minimization(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> list[list[float]]:
    converted: list[list[float]] = []
    for vector_index, vector in enumerate(objective_vectors):
        if len(vector) != len(directions):
            raise ValueError("objective vector length must match directions length")
        converted_vector = [
            -float(value) if directions[idx] else float(value)
            for idx, value in enumerate(vector)
        ]
        validate_objective_vector(converted_vector, label=f"vector[{vector_index}]")
        converted.append(converted_vector)
    return converted


def dominates(
    left: Sequence[float],
    right: Sequence[float],
    directions: Sequence[bool],
) -> bool:
    left_min = _to_minimization([left], directions)[0]
    right_min = _to_minimization([right], directions)[0]
    strictly_better = False
    for left_value, right_value in zip(left_min, right_min, strict=True):
        if left_value > right_value:
            return False
        if left_value < right_value:
            strictly_better = True
    return strictly_better


def nondominated_vectors(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> list[list[float]]:
    if not objective_vectors:
        return []
    nondominated: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for index, candidate in enumerate(objective_vectors):
        dominated = False
        for other_index, other in enumerate(objective_vectors):
            if index == other_index:
                continue
            if dominates(other, candidate, directions):
                dominated = True
                break
        if not dominated:
            normalized = tuple(float(value) for value in candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            nondominated.append(list(normalized))
    return nondominated


def hypervolume_2d(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    reference_point: Sequence[float],
) -> float:
    if len(directions) != 2 or len(reference_point) != 2:
        raise ValueError("hypervolume_2d requires exactly two objectives")
    if not objective_vectors:
        return 0.0

    front = nondominated_vectors(objective_vectors, directions)
    points = _to_minimization(front, directions)
    reference = _to_minimization([reference_point], directions)[0]
    bounded = [
        point
        for point in points
        if point[0] <= reference[0] and point[1] <= reference[1]
    ]
    if not bounded:
        return 0.0

    bounded.sort(key=lambda point: (point[0], point[1]))
    filtered: list[list[float]] = []
    best_y = math.inf
    for x, y in bounded:
        if y < best_y:
            filtered.append([x, y])
            best_y = y

    hv = 0.0
    previous_y = reference[1]
    for x, y in filtered:
        hv += max(0.0, reference[0] - x) * max(0.0, previous_y - y)
        previous_y = min(previous_y, y)
    return hv


def zdt1_reference_front(point_count: int = 201) -> list[list[float]]:
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    points: list[list[float]] = []
    for index in range(point_count):
        f1 = index / float(point_count - 1)
        f2 = 1.0 - math.sqrt(f1)
        points.append([f1, f2])
    return points


def zdt2_reference_front(point_count: int = 201) -> list[list[float]]:
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    points: list[list[float]] = []
    for index in range(point_count):
        f1 = index / float(point_count - 1)
        f2 = 1.0 - (f1**2)
        points.append([f1, f2])
    return points


def zdt3_reference_front(point_count: int = 201) -> list[list[float]]:
    if point_count < 5:
        raise ValueError("point_count must be at least 5")
    segments = [
        (0.0, 0.0830015349),
        (0.1822287280, 0.2577623634),
        (0.4093136748, 0.4538821041),
        (0.6183967944, 0.6525117038),
        (0.8233317983, 0.8518328654),
    ]
    base_count = point_count // len(segments)
    remainder = point_count % len(segments)
    points: list[list[float]] = []
    for segment_index, (left, right) in enumerate(segments):
        local_count = base_count + (1 if segment_index < remainder else 0)
        local_count = max(2, local_count)
        for local_index in range(local_count):
            if segment_index > 0 and local_index == 0:
                continue
            ratio = local_index / float(local_count - 1)
            f1 = left + (right - left) * ratio
            f2 = 1.0 - math.sqrt(f1) - f1 * math.sin(10.0 * math.pi * f1)
            points.append([f1, f2])
    return points


def dtlz2_reference_front(
    point_count: int = 201,
    *,
    objective_count: int = 2,
) -> list[list[float]]:
    if objective_count != 2:
        raise ValueError("dtlz2_reference_front currently supports objective_count=2 only")
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    points: list[list[float]] = []
    for index in range(point_count):
        theta = (index / float(point_count - 1)) * (math.pi / 2.0)
        points.append([math.cos(theta), math.sin(theta)])
    return points


def dtlz3_reference_front(
    point_count: int = 201,
    *,
    objective_count: int = 2,
) -> list[list[float]]:
    return dtlz2_reference_front(point_count, objective_count=objective_count)


def dtlz4_reference_front(
    point_count: int = 201,
    *,
    objective_count: int = 2,
) -> list[list[float]]:
    return dtlz2_reference_front(point_count, objective_count=objective_count)


def pymoo_reference_front(
    problem_name: str,
    *,
    point_count: int = 201,
    objective_count: int = 2,
    variable_count: int = 6,
) -> list[list[float]]:
    if importlib.util.find_spec("pymoo") is None:
        raise ImportError("pymoo is required for pymoo-backed reference fronts")
    from pymoo.problems import get_problem

    problem = get_problem(problem_name, n_var=variable_count, n_obj=objective_count)
    front = problem.pareto_front(n_pareto_points=point_count)
    if front is None:
        raise ValueError(f"Pareto front is unavailable for problem: {problem_name}")
    rows = front.tolist() if hasattr(front, "tolist") else list(front)
    normalized = [[float(value) for value in row] for row in rows]
    for index, row in enumerate(normalized):
        validate_objective_vector(row, label=f"{problem_name}_reference_front[{index}]")
    return normalized


def reference_front_for_problem(
    problem_name: str,
    *,
    point_count: int = 201,
    objective_count: int = 2,
    variable_count: int = 6,
) -> list[list[float]]:
    normalized = problem_name.strip().lower().replace("_", "-")
    if normalized == "zdt1":
        return zdt1_reference_front(point_count)
    if normalized == "zdt2":
        return zdt2_reference_front(point_count)
    if normalized == "zdt3":
        return zdt3_reference_front(point_count)
    if normalized == "dtlz2":
        return dtlz2_reference_front(point_count, objective_count=objective_count)
    if normalized == "dtlz3":
        return dtlz3_reference_front(point_count, objective_count=objective_count)
    if normalized == "dtlz4":
        return dtlz4_reference_front(point_count, objective_count=objective_count)
    if normalized == "wfg1":
        return pymoo_reference_front(
            "wfg1",
            point_count=point_count,
            objective_count=objective_count,
            variable_count=variable_count,
        )
    if normalized == "wfg2":
        return pymoo_reference_front(
            "wfg2",
            point_count=point_count,
            objective_count=objective_count,
            variable_count=variable_count,
        )
    raise ValueError(f"Unsupported reference front problem: {problem_name}")


def _nearest_distance(
    point: Sequence[float],
    candidates: Sequence[Sequence[float]],
) -> float:
    return min(math.dist(point, candidate) for candidate in candidates)


def generational_distance(
    objective_vectors: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> float:
    if not objective_vectors or not reference_front:
        return math.nan
    front = _to_minimization(nondominated_vectors(objective_vectors, directions), directions)
    reference = _to_minimization(reference_front, directions)
    distances = [_nearest_distance(point, reference) for point in front]
    return sum(distances) / len(distances) if distances else math.nan


def inverted_generational_distance(
    objective_vectors: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> float:
    if not objective_vectors or not reference_front:
        return math.nan
    front = _to_minimization(nondominated_vectors(objective_vectors, directions), directions)
    reference = _to_minimization(reference_front, directions)
    distances = [_nearest_distance(point, front) for point in reference]
    return sum(distances) / len(distances) if distances else math.nan


def spacing_metric(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> float:
    front = _to_minimization(nondominated_vectors(objective_vectors, directions), directions)
    if len(front) < 2:
        return math.nan
    nearest_distances: list[float] = []
    for index, point in enumerate(front):
        others = front[:index] + front[index + 1 :]
        nearest_distances.append(_nearest_distance(point, others))
    if not nearest_distances:
        return math.nan
    mean_distance = sum(nearest_distances) / len(nearest_distances)
    if mean_distance == 0.0:
        return 0.0
    variance = sum((distance - mean_distance) ** 2 for distance in nearest_distances) / len(
        nearest_distances
    )
    return math.sqrt(variance)


def objective_min_max(objective_vectors: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    if not objective_vectors:
        return {"min": [], "max": []}
    objective_count = len(objective_vectors[0])
    mins = [math.inf] * objective_count
    maxs = [-math.inf] * objective_count
    for vector in objective_vectors:
        if len(vector) != objective_count:
            raise ValueError("objective vectors must all have the same length")
        validate_objective_vector(vector, label="objective_min_max")
        for index, value in enumerate(vector):
            value = float(value)
            mins[index] = min(mins[index], value)
            maxs[index] = max(maxs[index], value)
    return {
        "min": [float(value) for value in mins],
        "max": [float(value) for value in maxs],
    }


def coverage_indicator(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    directions: Sequence[bool],
) -> float:
    if not right:
        return math.nan
    dominated_count = 0
    for candidate in right:
        if any(dominates(front_point, candidate, directions) for front_point in left):
            dominated_count += 1
    return dominated_count / len(right)


def evaluate_front_metrics(
    objective_vectors: Sequence[Sequence[float]],
    *,
    directions: Sequence[bool],
    reference_front: Sequence[Sequence[float]],
    reference_point: Sequence[float],
) -> dict[str, float | int | dict[str, list[float]]]:
    for index, vector in enumerate(objective_vectors):
        validate_objective_vector(vector, label=f"front[{index}]")
    front = nondominated_vectors(objective_vectors, directions)
    gd = generational_distance(front, reference_front, directions)
    return {
        "archive_size": len(objective_vectors),
        "nondominated_count": len(front),
        "hypervolume_2d": hypervolume_2d(front, directions, reference_point),
        "reference_front_distance": gd,
        "generational_distance": gd,
        "inverted_generational_distance": inverted_generational_distance(
            front, reference_front, directions
        ),
        "spacing": spacing_metric(front, directions),
        "objective_min_max": objective_min_max(front),
    }
