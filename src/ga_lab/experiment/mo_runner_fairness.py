from __future__ import annotations

from typing import Any

from ga_lab.config import GAConfig
from ga_lab.experiment.nsga2_candidate_suite import MOBenchmarkSpec
from ga_lab.experiment.nsga2_candidate_variants import (
    NSGA2CandidateVariant,
    candidate_variant_metadata,
)
from ga_lab.experiment.parameter_fairness import METRIC_POSTPROCESSING_ID


_CANDIDATE_DUPLICATE_HANDLING: dict[str, str] = {
    "candidate_d_uniform_crossover": "none",
    "candidate_e_uniform_decision_dedup": "decision_dedup + retry1 + reinitialize_fallback",
    "candidate_f_uniform_objective_dedup": "partial_front_objective_dedup",
    "candidate_g_uniform_crowding_survival": "novelty_crowding_survival",
    "candidate_h_uniform_dedup_mutation_boost": "decision_dedup + retry2 + novelty_survival",
    "candidate_i_h_lite_retry1": "decision_dedup + retry1_lite",
    "candidate_j_h_lite_retry2": "decision_dedup + retry2_lite",
    "candidate_k_dedup_only_lite": "decision_dedup + reinitialize_fallback",
    "candidate_l_sparse_parent_bias_light": "decision_dedup + retry2_lite + sparse_parent_bias_light",
    "candidate_m_boundary_preservation_light": "decision_dedup + retry2_lite + boundary_preservation_light",
}


def build_mo_benchmark_rows(specs: list[MOBenchmarkSpec]) -> list[dict[str, Any]]:
    return [
        {
            "problem": spec.problem,
            "variables": spec.variables,
            "objectives": spec.objectives,
            "bounds": list(spec.bounds),
            "reference_front": spec.reference_front_name,
            "hv_reference_point": list(spec.hv_reference_point),
            "notes": spec.notes,
        }
        for spec in specs
    ]


def build_candidate_rows(variants: list[NSGA2CandidateVariant]) -> list[dict[str, Any]]:
    return [candidate_variant_metadata(variant) for variant in variants]


def fairness_summary_rows(fairness_payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary_counts = dict(fairness_payload.get("summary_counts", {}))
    return [
        {
            "status": str(fairness_payload.get("status", "unknown")),
            "pass": int(summary_counts.get("pass", 0)),
            "warning": int(summary_counts.get("warning", 0)),
            "fail": int(summary_counts.get("fail", 0)),
        }
    ]


def algorithm_contract(
    *,
    base_config: GAConfig,
    spec: MOBenchmarkSpec,
    algorithm: str,
    variant: NSGA2CandidateVariant | None = None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "objective_count": spec.objectives,
        "variable_count": spec.variables,
        "bounds": list(spec.bounds),
        "reference_front_source": spec.reference_front_name,
        "hypervolume_reference_point": list(spec.hv_reference_point),
        "metric_postprocessing": METRIC_POSTPROCESSING_ID,
        "population_size": base_config.population_size,
        "configured_generations": base_config.generations,
        "repair_bounds": "representation_bounds",
    }
    if algorithm == "internal_nsga2":
        return contract | {
            "algorithm_family": "internal_nsga2",
            "operator_family": "internal_nsga2_arithmetic_gaussian",
            "crossover_type": base_config.crossover,
            "mutation_type": base_config.mutation,
            "mutation_probability": base_config.mutation_rate,
            "duplicate_handling": "none",
        }
    if variant is not None:
        return contract | candidate_variant_metadata(variant) | {
            "algorithm_family": "internal_nsga2_candidate",
            "operator_family": variant.candidate_id,
            "crossover_type": "uniform",
            "mutation_type": base_config.mutation,
            "mutation_probability": base_config.mutation_rate,
            "duplicate_handling": _CANDIDATE_DUPLICATE_HANDLING.get(
                variant.candidate_id,
                "candidate_specific",
            ),
        }
    if algorithm == "pymoo_nsga2":
        mutation_probability = (
            base_config.mutation_rate
            if base_config.mutation_rate > 0
            else 1.0 / max(1, spec.variables)
        )
        return contract | {
            "algorithm_family": "external_nsga2",
            "operator_family": "pymoo_standard_sbx_pm",
            "crossover_type": "sbx",
            "mutation_type": "polynomial",
            "mutation_probability": mutation_probability,
            "duplicate_handling": "none",
        }
    if algorithm == "deap_nsga2":
        return contract | {
            "algorithm_family": "external_nsga2",
            "operator_family": "deap_selNSGA2_sbx_poly",
            "crossover_type": "sbx",
            "mutation_type": "polynomial",
            "mutation_probability": base_config.mutation_rate,
            "duplicate_handling": "none",
        }
    if algorithm == "random_pareto_archive":
        return contract | {
            "algorithm_family": "random_archive_baseline",
            "operator_family": "random_archive_baseline",
            "population_size": None,
            "configured_generations": None,
            "crossover_type": None,
            "mutation_type": None,
            "mutation_probability": None,
            "duplicate_handling": "archive_nondominated_only",
        }
    return contract | {
        "algorithm_family": "unknown",
        "operator_family": "unknown",
        "crossover_type": None,
        "mutation_type": None,
        "mutation_probability": None,
        "duplicate_handling": "unknown",
    }


def decorate_fairness_row(
    row: dict[str, Any],
    *,
    spec: MOBenchmarkSpec,
    base_config: GAConfig,
    requested_budget: int,
    variant_map: dict[str, NSGA2CandidateVariant],
) -> dict[str, Any]:
    algorithm = str(row["algorithm"])
    variant = variant_map.get(algorithm)
    metadata = dict(row.get("metadata", {}))
    metadata.update(
        algorithm_contract(
            base_config=base_config,
            spec=spec,
            algorithm=algorithm,
            variant=variant,
        )
    )
    row["metadata"] = metadata
    row["metric_postprocessing"] = METRIC_POSTPROCESSING_ID
    row["reference_front_source"] = spec.reference_front_name
    row["hv_reference_point"] = list(spec.hv_reference_point)
    row["problem_objectives"] = spec.objectives
    row["problem_variables"] = spec.variables
    row["problem_bounds"] = list(spec.bounds)
    row["requested_budget"] = requested_budget
    row["benchmark_notes"] = spec.notes
    return row
