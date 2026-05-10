from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any, Mapping, Sequence

from ga_lab.experiment.mo_metrics import spacing_metric
from ga_lab.experiment.nsga2_diagnostics import (
    compute_objective_bins,
    compute_segment_distribution,
    compute_zdt1_components,
    hash_solution_or_objective,
)


_LOWER_IS_BETTER_METRICS = {
    "g_mean",
    "segment0_g_mean",
    "segment0_distance_mean",
    "spacing",
    "reference_front_distance",
    "inverted_generational_distance",
    "runtime_seconds",
}

_HIGHER_IS_BETTER_METRICS = {
    "segment0_low_g_count",
    "occupied_bins",
    "segment0_coverage",
    "nondominated_count",
    "hypervolume_2d",
    "coverage_indicator",
}


@dataclass(slots=True)
class ExternalOperatorParityConfig:
    external_parity_trace_enabled: bool = False
    segment_count: int = 6
    tail_low_threshold: float = 0.2
    low_g_threshold: float = 1.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_vector_rows(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list | tuple):
            continue
        try:
            rows.append([float(item) for item in row])
        except (TypeError, ValueError):
            continue
    return rows


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


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


def _safe_median(values: Sequence[Any]) -> float | None:
    numeric = [value for item in values if (value := _finite(item)) is not None]
    if not numeric:
        return None
    return float(median(numeric))


def _safe_min(values: Sequence[Any]) -> float | None:
    numeric = [value for item in values if (value := _finite(item)) is not None]
    if not numeric:
        return None
    return float(min(numeric))


def _safe_max(values: Sequence[Any]) -> float | None:
    numeric = [value for item in values if (value := _finite(item)) is not None]
    if not numeric:
        return None
    return float(max(numeric))


def _safe_std(values: Sequence[Any]) -> float | None:
    numeric = [value for item in values if (value := _finite(item)) is not None]
    if not numeric:
        return None
    baseline = float(mean(numeric))
    variance = sum((value - baseline) ** 2 for value in numeric) / len(numeric)
    return float(math.sqrt(variance))


def _candidate_vectors(row: Mapping[str, Any], keys: Sequence[str]) -> list[list[float]]:
    metadata = _metadata(row)
    for key in keys:
        rows = _clean_vector_rows(metadata.get(key))
        if rows:
            return rows
        rows = _clean_vector_rows(row.get(key))
        if rows:
            return rows
    return []


def extract_internal_final_decisions(row: Mapping[str, Any]) -> list[list[float]]:
    return _candidate_vectors(
        row,
        (
            "population_decision_vectors",
            "decision_vectors",
            "front_decision_vectors",
        ),
    )


def extract_pymoo_final_decisions(row: Mapping[str, Any]) -> list[list[float]]:
    return _candidate_vectors(
        row,
        (
            "population_decision_vectors",
            "decision_vectors",
            "front_decision_vectors",
        ),
    )


def extract_internal_final_objectives(row: Mapping[str, Any]) -> list[list[float]]:
    return _candidate_vectors(
        row,
        (
            "population_objective_vectors",
            "front_objective_vectors",
            "objective_vectors",
        ),
    )


def extract_pymoo_final_objectives(row: Mapping[str, Any]) -> list[list[float]]:
    return _candidate_vectors(
        row,
        (
            "population_objective_vectors",
            "front_objective_vectors",
            "objective_vectors",
        ),
    )


def summarize_final_decision_distribution(
    decision_vectors: Sequence[Sequence[float | int]],
    *,
    tail_low_threshold: float = 0.2,
) -> dict[str, Any]:
    if not decision_vectors:
        return {
            "decision_count": 0,
            "x0_min": None,
            "x0_mean": None,
            "x0_max": None,
            "x0_std": None,
            "tail_mean_min": None,
            "tail_mean_mean": None,
            "tail_mean_max": None,
            "tail_std_mean": None,
            "tail_low_rate": None,
            "unique_decision_count": 0,
            "warnings": ["decision_vectors_unavailable"],
        }

    normalized = [[float(value) for value in row] for row in decision_vectors if row]
    x0_values = [row[0] for row in normalized if row]
    tail_means = [
        float(sum(row[1:]) / len(row[1:]))
        for row in normalized
        if len(row) > 1
    ]
    tail_stds = [
        _safe_std(row[1:])
        for row in normalized
        if len(row) > 1
    ]
    tail_low_count = sum(
        1 for value in tail_means if value is not None and float(value) <= float(tail_low_threshold)
    )
    unique_decisions = {hash_solution_or_objective(row) for row in normalized}
    warnings: list[str] = []
    if len(tail_means) != len(normalized):
        warnings.append("short_decision_vectors_without_tail")

    return {
        "decision_count": len(normalized),
        "x0_min": _safe_min(x0_values),
        "x0_mean": _safe_mean(x0_values),
        "x0_max": _safe_max(x0_values),
        "x0_std": _safe_std(x0_values),
        "tail_mean_min": _safe_min(tail_means),
        "tail_mean_mean": _safe_mean(tail_means),
        "tail_mean_max": _safe_max(tail_means),
        "tail_std_mean": _safe_mean(tail_stds),
        "tail_low_rate": (
            float(tail_low_count / len(tail_means)) if tail_means else None
        ),
        "unique_decision_count": len(unique_decisions),
        "warnings": warnings,
    }


def compute_zdt1_component_summary_for_decisions(
    decision_vectors: Sequence[Sequence[float | int]],
    *,
    objective_vectors: Sequence[Sequence[float]] | None = None,
    bins: int = 6,
    low_g_threshold: float = 1.1,
) -> dict[str, Any]:
    warnings: list[str] = []
    objective_vectors = objective_vectors or []
    components = [
        compute_zdt1_components(
            decision,
            objective_vectors[index] if index < len(objective_vectors) else None,
            bins=bins,
        )
        for index, decision in enumerate(decision_vectors)
    ]
    if not components:
        warnings.append("decision_vectors_unavailable")

    segment0 = [component for component in components if bool(component.get("segment0_flag"))]
    segment0_low_g = [
        component
        for component in segment0
        if isinstance(component.get("g"), int | float)
        and float(component["g"]) <= float(low_g_threshold)
    ]
    warnings.extend(
        warning
        for component in components
        for warning in list(component.get("warnings", []))
        if isinstance(warning, str)
    )
    return {
        "component_count": len(components),
        "g_min": _safe_min([component.get("g") for component in components]),
        "g_mean": _safe_mean([component.get("g") for component in components]),
        "g_max": _safe_max([component.get("g") for component in components]),
        "f1_min": _safe_min([component.get("f1") for component in components]),
        "f1_mean": _safe_mean([component.get("f1") for component in components]),
        "f1_max": _safe_max([component.get("f1") for component in components]),
        "f2_min": _safe_min([component.get("f2") for component in components]),
        "f2_mean": _safe_mean([component.get("f2") for component in components]),
        "f2_max": _safe_max([component.get("f2") for component in components]),
        "distance_mean": _safe_mean(
            [component.get("distance_to_zdt1_front") for component in components]
        ),
        "distance_median": _safe_median(
            [component.get("distance_to_zdt1_front") for component in components]
        ),
        "segment0_count": len(segment0),
        "segment0_g_mean": _safe_mean([component.get("g") for component in segment0]),
        "segment0_distance_mean": _safe_mean(
            [component.get("distance_to_zdt1_front") for component in segment0]
        ),
        "segment0_low_g_count": len(segment0_low_g),
        "warnings": sorted(set(warnings)),
    }


def summarize_final_zdt1_component_distribution(
    decision_vectors: Sequence[Sequence[float | int]],
    objective_vectors: Sequence[Sequence[float]] | None = None,
    *,
    bins: int = 6,
    low_g_threshold: float = 1.1,
) -> dict[str, Any]:
    return compute_zdt1_component_summary_for_decisions(
        decision_vectors,
        objective_vectors=objective_vectors,
        bins=bins,
        low_g_threshold=low_g_threshold,
    )


def summarize_final_objective_segment_distribution(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[bool],
    *,
    bins: int = 6,
) -> dict[str, Any]:
    if not objective_vectors:
        return {
            "occupied_bins": 0,
            "segment0_coverage": None,
            "boundary_adjacent_count": 0,
            "empty_segment_count": bins,
            "spacing": None,
            "nondominated_count": 0,
            "warnings": ["objective_vectors_unavailable"],
        }

    occupancy = compute_objective_bins(objective_vectors, directions, bins=bins)
    segments = compute_segment_distribution(objective_vectors, directions, bins=bins)
    segment0_count = int(segments.get("segment_counts", {}).get("0", 0))
    spacing_value = _finite(spacing_metric(objective_vectors, directions))
    warnings = list(occupancy.get("warnings", [])) + list(segments.get("warnings", []))
    return {
        "occupied_bins": int(occupancy.get("occupied_bins", 0)),
        "segment0_coverage": (
            float(segment0_count / len(objective_vectors)) if objective_vectors else None
        ),
        "boundary_adjacent_count": int(segments.get("boundary_adjacent_count", 0)),
        "empty_segment_count": int(segments.get("empty_segment_count", bins)),
        "spacing": spacing_value,
        "nondominated_count": len(objective_vectors),
        "warnings": warnings,
    }


def summarize_operator_parameter_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _metadata(row)
    algorithm = str(row.get("algorithm", "unknown"))
    initialization = str(
        metadata.get("initialization")
        or metadata.get("sampling")
        or "random_uniform_bounds"
    )
    crossover = str(
        metadata.get("crossover_type")
        or metadata.get("crossover")
        or "unknown"
    )
    mutation = str(
        metadata.get("mutation_type")
        or metadata.get("mutation")
        or "unknown"
    )
    duplicate_handling = str(metadata.get("duplicate_handling") or "unknown")
    survival = str(metadata.get("survival") or "rank_crowding")
    crowding = str(metadata.get("crowding") or "nsga2_crowding_distance")
    selection = str(metadata.get("selection") or "tournament")

    if algorithm == "internal_nsga2":
        key_difference = "baseline arithmetic crossover + gaussian mutation"
    elif algorithm == "candidate_j_h_lite_retry2":
        key_difference = "uniform crossover + decision dedup + retry2_lite"
    elif algorithm == "candidate_n_low_g_tail_mutation_light":
        key_difference = "candidate_j path + weak tail-only low-g perturbation"
    elif algorithm == "pymoo_nsga2":
        key_difference = "SBX/PM operator family with binary tournament and rank_crowding"
    elif algorithm == "deap_nsga2":
        key_difference = "selTournamentDCD + selNSGA2 with SBX/polynomial bounded operators"
    else:
        key_difference = str(metadata.get("operator_family") or "algorithm_specific")

    return {
        "algorithm": algorithm,
        "population_size": metadata.get("population_size"),
        "generations": metadata.get("configured_generations")
        or metadata.get("actual_generations_hint"),
        "requested_budget": row.get("requested_budget"),
        "actual_evaluations": row.get("actual_evaluations"),
        "initialization": initialization,
        "crossover": crossover,
        "mutation": mutation,
        "duplicate_handling": duplicate_handling,
        "repair_or_bounds_handling": str(
            metadata.get("repair_or_bounds_handling") or metadata.get("repair_bounds") or "bounds_only"
        ),
        "survival": survival,
        "crowding": crowding,
        "selection": selection,
        "operator_family": str(metadata.get("operator_family") or "unknown"),
        "key_difference": key_difference,
    }


def summarize_parity_gap(
    algorithm_summaries: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidate_j = dict(algorithm_summaries.get("candidate_j_h_lite_retry2", {}))
    candidate_n = dict(algorithm_summaries.get("candidate_n_low_g_tail_mutation_light", {}))
    pymoo = dict(algorithm_summaries.get("pymoo_nsga2", {}))

    metrics = [
        "g_mean",
        "segment0_g_mean",
        "segment0_distance_mean",
        "segment0_low_g_count",
        "occupied_bins",
        "spacing",
        "nondominated_count",
        "hypervolume_2d",
        "inverted_generational_distance",
        "runtime_seconds",
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
        if candidate_n_value is None:
            interpretation = "candidate_n value unavailable"
        elif metric in _LOWER_IS_BETTER_METRICS:
            if candidate_j_value is not None and candidate_n_value < candidate_j_value:
                interpretation = "candidate_n improved versus candidate_j on a lower-is-better metric"
            elif candidate_j_value is not None and math.isclose(
                candidate_n_value,
                candidate_j_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                interpretation = "candidate_n stayed effectively tied with candidate_j"
            else:
                interpretation = "candidate_n did not improve over candidate_j on this lower-is-better metric"
            if pymoo_value is not None and candidate_n_value > pymoo_value:
                interpretation += "; pymoo remains stronger"
        elif metric in _HIGHER_IS_BETTER_METRICS:
            if candidate_j_value is not None and candidate_n_value > candidate_j_value:
                interpretation = "candidate_n improved versus candidate_j on a higher-is-better metric"
            elif candidate_j_value is not None and math.isclose(
                candidate_n_value,
                candidate_j_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                interpretation = "candidate_n stayed effectively tied with candidate_j"
            else:
                interpretation = "candidate_n did not improve over candidate_j on this higher-is-better metric"
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
                "interpretation": interpretation,
            }
        )
    return rows
