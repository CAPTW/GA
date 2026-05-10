from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ga_lab.config import GAConfig


@dataclass(frozen=True, slots=True)
class NSGA2CandidateVariant:
    candidate_id: str
    operator_change: str
    source_diagnosis_report: str
    default_changed: bool
    promotion_status: str
    notes: str
    base_candidate_id: str | None = None
    source_artifacts: tuple[str, ...] = ()
    expected_gain: str = ""
    risk: str = ""
    mechanism: str = ""
    source_hypothesis: str = ""
    allowed_use: str = ""
    disallowed_use: str = ""
    zdt1_specific_warning: str = ""
    candidate_parameters: dict[str, Any] | None = None


def candidate_d_uniform_crossover() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_d_uniform_crossover",
        operator_change="crossover arithmetic -> uniform",
        source_diagnosis_report="artifacts/nsga2_parity_diagnosis_report.md",
        default_changed=False,
        promotion_status="under_validation",
        notes=(
            "ZDT1 parity diagnosis candidate. Intended to test the arithmetic crossover "
            "front-contraction hypothesis without changing the frozen internal default."
        ),
        expected_gain="HV / distance / GD / IGD / coverage",
        risk="spacing and nondominated_count may still trail external comparators",
    )


def candidate_e_uniform_decision_dedup() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_e_uniform_decision_dedup",
        operator_change="candidate_d + offspring decision dedup / reinitialize fallback",
        source_diagnosis_report="artifacts/change_requests/nsga2_uniform_crossover_candidate_d_review.md",
        default_changed=False,
        promotion_status="under_validation",
        notes=(
            "Follow-up diversity candidate that keeps uniform crossover and tries to reduce "
            "decision-vector duplicates before survivor selection."
        ),
        base_candidate_id="candidate_d_uniform_crossover",
        source_artifacts=(
            "artifacts/change_requests/nsga2_uniform_crossover_candidate_d_cr.md",
            "artifacts/nsga2_candidate_suite_validation_report_candidate_d_cr.md",
        ),
        expected_gain="spacing / nondominated_count / decision_duplicate_rate",
        risk="offspring dedup may add exploration noise and weaken convergence on DTLZ-like fronts",
    )


def candidate_f_uniform_objective_dedup() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_f_uniform_objective_dedup",
        operator_change="candidate_d + partial-front objective dedup",
        source_diagnosis_report="artifacts/change_requests/nsga2_uniform_crossover_candidate_d_review.md",
        default_changed=False,
        promotion_status="under_validation",
        notes=(
            "Partial-front survivor selection prefers unique objective vectors before allowing "
            "duplicate points from the overflow front."
        ),
        base_candidate_id="candidate_d_uniform_crossover",
        source_artifacts=(
            "artifacts/change_requests/nsga2_uniform_crossover_candidate_d_cr.md",
            "artifacts/nsga2_candidate_suite_validation_report_candidate_d_cr.md",
        ),
        expected_gain="spacing / nondominated_count / archive_duplicate_rate",
        risk="partial-front dedup can trade some convergence pressure for spread",
    )


def candidate_g_uniform_crowding_survival() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_g_uniform_crowding_survival",
        operator_change="candidate_d + novelty-aware partial-front survival",
        source_diagnosis_report="artifacts/change_requests/nsga2_uniform_crossover_candidate_d_review.md",
        default_changed=False,
        promotion_status="under_validation",
        notes=(
            "Uses novelty-aware partial-front survivor selection so that spacing pressure is "
            "not driven only by plain crowding-distance sorting."
        ),
        base_candidate_id="candidate_d_uniform_crossover",
        source_artifacts=(
            "artifacts/change_requests/nsga2_uniform_crossover_candidate_d_cr.md",
            "artifacts/nsga2_candidate_suite_validation_report_candidate_d_cr.md",
        ),
        expected_gain="spacing / coverage / boundary preservation",
        risk="novelty-biased survival may sacrifice IGD or HV on some benchmarks",
    )


def candidate_h_uniform_dedup_mutation_boost() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_h_uniform_dedup_mutation_boost",
        operator_change="candidate_d + decision dedup + duplicate-triggered mutation boost + novelty survival",
        source_diagnosis_report="artifacts/change_requests/nsga2_uniform_crossover_candidate_d_review.md",
        default_changed=False,
        promotion_status="under_validation",
        notes=(
            "Compound diversity candidate that retries duplicate offspring with higher mutation "
            "pressure and uses novelty-aware partial-front survivor selection."
        ),
        base_candidate_id="candidate_d_uniform_crossover",
        source_artifacts=(
            "artifacts/change_requests/nsga2_uniform_crossover_candidate_d_cr.md",
            "artifacts/nsga2_candidate_suite_validation_report_candidate_d_cr.md",
        ),
        expected_gain="spacing / nondominated_count with reduced duplicate pressure",
        risk="too much mutation pressure can erase candidate_d convergence gains",
    )


def candidate_i_h_lite_retry1() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_i_h_lite_retry1",
        operator_change="candidate_d + decision dedup + duplicate-triggered mutation retry x1 (lighter than candidate_h)",
        source_diagnosis_report="artifacts/nsga2_diversity_candidate_report_diversity_run1.md",
        default_changed=False,
        promotion_status="under_validation",
        notes=(
            "First h-lite follow-up that keeps candidate_d uniform crossover, enables decision "
            "dedup, and uses a single duplicate-triggered mutation retry without novelty survival."
        ),
        base_candidate_id="candidate_h_uniform_dedup_mutation_boost",
        source_artifacts=(
            "artifacts/change_requests/nsga2_uniform_crossover_candidate_d_cr.md",
            "artifacts/nsga2_diversity_candidate_report_diversity_run1.md",
        ),
        expected_gain="spacing / nondominated_count with less HV-distance regression than candidate_h",
        risk="single retry may leave too many duplicates to close the external diversity gap",
    )


def candidate_j_h_lite_retry2() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_j_h_lite_retry2",
        operator_change="candidate_d + decision dedup + duplicate-triggered mutation retry x2 at lighter scale",
        source_diagnosis_report="artifacts/nsga2_diversity_candidate_report_diversity_run1.md",
        default_changed=False,
        promotion_status="under_validation",
        notes=(
            "Second h-lite follow-up that retries duplicates twice, but with a milder mutation "
            "boost than candidate_h and no novelty survivor override."
        ),
        base_candidate_id="candidate_h_uniform_dedup_mutation_boost",
        source_artifacts=(
            "artifacts/change_requests/nsga2_uniform_crossover_candidate_d_cr.md",
            "artifacts/nsga2_diversity_candidate_report_diversity_run1.md",
        ),
        expected_gain="duplicate reduction with smaller convergence penalty than candidate_h",
        risk="still may over-correct on DTLZ-like fronts while not fully restoring spacing",
    )


def candidate_k_dedup_only_lite() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_k_dedup_only_lite",
        operator_change="candidate_d + decision dedup / reinitialize fallback without mutation boost",
        source_diagnosis_report="artifacts/nsga2_diversity_candidate_report_diversity_run1.md",
        default_changed=False,
        promotion_status="under_validation",
        notes=(
            "Dedup-only h-lite follow-up that reuses the cleanest part of candidate_e without "
            "adding extra duplicate-triggered mutation pressure."
        ),
        base_candidate_id="candidate_d_uniform_crossover",
        source_artifacts=(
            "artifacts/change_requests/nsga2_uniform_crossover_candidate_d_cr.md",
            "artifacts/nsga2_diversity_candidate_report_diversity_run1.md",
        ),
        expected_gain="DTLZ-friendly diversity cleanup with minimal convergence regression",
        risk="dedup alone may be too weak to materially improve spacing on the ZDT suite",
    )


def candidate_l_sparse_parent_bias_light() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_l_sparse_parent_bias_light",
        operator_change=(
            "candidate_j + light sparse-region parent bias during parent selection only"
        ),
        source_diagnosis_report=(
            "artifacts/hypotheses/nsga2_survivor_pressure_controlled_ablation_plan.md"
        ),
        default_changed=False,
        promotion_status="phase0_sanity",
        notes=(
            "Phase 0 survivor-pressure candidate that keeps candidate_j duplicate handling and "
            "uniform crossover, but adds a very light sparse-region bias only when selecting "
            "the second parent in a mating pair."
        ),
        base_candidate_id="candidate_j_h_lite_retry2",
        source_artifacts=(
            "artifacts/hypotheses/nsga2_survivor_pressure_controlled_ablation_plan.md",
            "artifacts/nsga2_survivor_pressure_planning_report.md",
            "artifacts/nsga2_candidate_governance_closure_report.md",
        ),
        expected_gain="spacing / nondominated_count",
        risk=(
            "even a light parent-selection bias may still regress HV / distance / IGD or show "
            "no measurable diversity benefit"
        ),
        mechanism="sparse_region_parent_bias_light",
        source_hypothesis="nsga2_survivor_pressure_controlled_ablation_plan",
        allowed_use="phase0_sanity_only",
        disallowed_use="default_replacement",
    )


def candidate_m_boundary_preservation_light() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_m_boundary_preservation_light",
        operator_change=(
            "candidate_j + light boundary-preservation preference inside partial-front overflow ordering"
        ),
        source_diagnosis_report=(
            "artifacts/hypotheses/nsga2_survivor_pressure_redesign_note.md"
        ),
        default_changed=False,
        promotion_status="phase0_sanity",
        notes=(
            "Phase 0 survivor-pressure candidate that keeps candidate_j duplicate handling and "
            "uniform crossover, but adds a very light same-front boundary preference without "
            "rewriting crowding distance or the full survival path."
        ),
        base_candidate_id="candidate_j_h_lite_retry2",
        source_artifacts=(
            "artifacts/hypotheses/nsga2_survivor_pressure_redesign_note.md",
            "artifacts/hypotheses/nsga2_sparse_parent_bias_phase1_postmortem.md",
            "artifacts/nsga2_candidate_l_closure_report.md",
        ),
        expected_gain="spacing / nondominated_count / hypervolume_2d / coverage_indicator",
        risk=(
            "boundary preservation may overlap with existing crowding logic, regress HV / distance "
            "/ IGD if it over-preserves extremes, or be too weak to show measurable effect"
        ),
        mechanism="boundary_preservation_light",
        source_hypothesis="nsga2_survivor_pressure_redesign_note",
        allowed_use="phase0_sanity_only",
        disallowed_use="default_replacement",
    )


def candidate_n_low_g_tail_mutation_light() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_n_low_g_tail_mutation_light",
        operator_change=(
            "candidate_j + very light tail-variable low-g perturbation after mutation / duplicate retry"
        ),
        source_diagnosis_report=(
            "artifacts/hypotheses/nsga2_operator_quality_phase0_plan.md"
        ),
        default_changed=False,
        promotion_status="phase0_sanity",
        notes=(
            "Phase 0 operator-quality candidate that keeps candidate_j duplicate handling and "
            "uniform crossover, while applying a very light stochastic downward perturbation only "
            "to tail variables and only for ZDT1-focused low-g sanity checks."
        ),
        base_candidate_id="candidate_j_h_lite_retry2",
        source_artifacts=(
            "artifacts/hypotheses/nsga2_zdt1_low_g_bottleneck_postmortem.md",
            "artifacts/hypotheses/nsga2_operator_quality_phase0_plan.md",
            "artifacts/nsga2_operator_quality_planning_report.md",
            "artifacts/nsga2_zdt1_component_diagnostics_report_zdtcomp1.md",
        ),
        expected_gain=(
            "segment0_g_mean / segment0_distance_mean without catastrophic HV-distance-IGD regression"
        ),
        risk=(
            "ZDT1-specific overfitting, diversity loss, weak effect size, or mutation noise without "
            "real low-g conversion"
        ),
        mechanism="low_g_tail_mutation_light",
        source_hypothesis="nsga2_operator_quality_phase0_plan",
        allowed_use="phase0_sanity_only",
        disallowed_use="default_replacement",
        zdt1_specific_warning=(
            "ZDT1-only sanity evidence; do not generalize as a broad MOEA improvement signal."
        ),
        candidate_parameters={
            "low_g_tail_probability": 0.4,
            "low_g_tail_gene_probability": 0.4,
            "low_g_tail_step_scale": 0.1,
            "zdt1_only": True,
        },
    )


def candidate_o_spread_preserving_variation_light() -> NSGA2CandidateVariant:
    return NSGA2CandidateVariant(
        candidate_id="candidate_o_spread_preserving_variation_light",
        operator_change=(
            "candidate_n + very light bounded spread-preserving jitter after mutation / duplicate retry"
        ),
        source_diagnosis_report=(
            "artifacts/hypotheses/nsga2_spread_preserving_phase0_plan.md"
        ),
        default_changed=False,
        promotion_status="phase0_sanity",
        notes=(
            "Phase 0 spread-preserving candidate that keeps candidate_n low-g tail "
            "mutation behavior intact, then adds a very light stochastic bounded "
            "decision-space jitter intended only to reduce premature spread collapse "
            "without steering directly on objective segments."
        ),
        base_candidate_id="candidate_n_low_g_tail_mutation_light",
        source_artifacts=(
            "artifacts/hypotheses/nsga2_spread_preservation_bottleneck_postmortem.md",
            "artifacts/hypotheses/nsga2_spread_preserving_phase0_plan.md",
            "artifacts/nsga2_spread_preserving_planning_report.md",
            "artifacts/nsga2_spread_parity_report_spread_parity1.md",
            "artifacts/nsga2_candidate_n_closure_report.md",
        ),
        expected_gain=(
            "occupied_bins / segment_entropy / spacing / nondominated_count without "
            "materially degrading candidate_n segment0_g_mean or segment0_distance_mean"
        ),
        risk=(
            "ZDT1-specific overfitting, objective-space steering appearance, low-g signal "
            "dilution, or added jitter without meaningful spread gain"
        ),
        mechanism="spread_preserving_variation_light",
        source_hypothesis="nsga2_spread_preserving_phase0_plan",
        allowed_use="phase0_sanity_only",
        disallowed_use="default_replacement",
        zdt1_specific_warning=(
            "ZDT1-only Phase 0 sanity evidence; do not treat this as a broad MOEA spread "
            "improvement claim."
        ),
        candidate_parameters={
            "low_g_tail_probability": 0.4,
            "low_g_tail_gene_probability": 0.4,
            "low_g_tail_step_scale": 0.1,
            "spread_preserving_probability": 0.25,
            "spread_preserving_x0_probability": 0.7,
            "spread_preserving_x0_step_scale": 0.04,
            "spread_preserving_tail_step_scale": 0.02,
            "zdt1_only": True,
        },
    )


def apply_candidate_variant(
    config: GAConfig,
    variant: NSGA2CandidateVariant,
) -> GAConfig:
    clone = GAConfig.from_dict(config.to_dict())
    clone.run_name = f"{config.run_name}_{variant.candidate_id}"
    clone.crossover_options = deepcopy(clone.crossover_options)
    clone.algorithm_options = deepcopy(clone.algorithm_options)
    clone.selection_options = deepcopy(clone.selection_options)
    if variant.candidate_id == "candidate_d_uniform_crossover":
        clone.crossover = "uniform"
        return clone
    if variant.candidate_id == "candidate_e_uniform_decision_dedup":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_offspring_decision_dedup"] = True
        clone.algorithm_options["nsga2_duplicate_retry_count"] = 1
        clone.algorithm_options["nsga2_duplicate_retry_mutation_scale"] = 1.0
        clone.algorithm_options["nsga2_duplicate_reinitialize_fallback"] = True
        return clone
    if variant.candidate_id == "candidate_f_uniform_objective_dedup":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_partial_front_dedup_mode"] = "objective"
        return clone
    if variant.candidate_id == "candidate_g_uniform_crowding_survival":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_partial_front_strategy"] = "novelty_crowding"
        return clone
    if variant.candidate_id == "candidate_h_uniform_dedup_mutation_boost":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_offspring_decision_dedup"] = True
        clone.algorithm_options["nsga2_duplicate_retry_count"] = 2
        clone.algorithm_options["nsga2_duplicate_retry_mutation_scale"] = 2.0
        clone.algorithm_options["nsga2_duplicate_reinitialize_fallback"] = True
        clone.algorithm_options["nsga2_partial_front_strategy"] = "novelty_crowding"
        clone.algorithm_options["nsga2_partial_front_dedup_mode"] = "decision"
        return clone
    if variant.candidate_id == "candidate_i_h_lite_retry1":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_offspring_decision_dedup"] = True
        clone.algorithm_options["nsga2_duplicate_retry_count"] = 1
        clone.algorithm_options["nsga2_duplicate_retry_mutation_scale"] = 1.25
        return clone
    if variant.candidate_id == "candidate_j_h_lite_retry2":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_offspring_decision_dedup"] = True
        clone.algorithm_options["nsga2_duplicate_retry_count"] = 2
        clone.algorithm_options["nsga2_duplicate_retry_mutation_scale"] = 1.5
        return clone
    if variant.candidate_id == "candidate_k_dedup_only_lite":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_offspring_decision_dedup"] = True
        clone.algorithm_options["nsga2_duplicate_reinitialize_fallback"] = True
        return clone
    if variant.candidate_id == "candidate_l_sparse_parent_bias_light":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_offspring_decision_dedup"] = True
        clone.algorithm_options["nsga2_duplicate_retry_count"] = 2
        clone.algorithm_options["nsga2_duplicate_retry_mutation_scale"] = 1.5
        clone.selection_options["nsga2_sparse_parent_bias_light"] = True
        clone.selection_options["nsga2_sparse_parent_bias_probability"] = 0.15
        clone.selection_options["nsga2_sparse_parent_bias_pool"] = max(
            int(clone.selection_options.get("tournament_size", clone.tournament_size)) + 1,
            3,
        )
        return clone
    if variant.candidate_id == "candidate_m_boundary_preservation_light":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_offspring_decision_dedup"] = True
        clone.algorithm_options["nsga2_duplicate_retry_count"] = 2
        clone.algorithm_options["nsga2_duplicate_retry_mutation_scale"] = 1.5
        clone.algorithm_options["nsga2_partial_front_strategy"] = "boundary_preservation_light"
        return clone
    if variant.candidate_id == "candidate_n_low_g_tail_mutation_light":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_offspring_decision_dedup"] = True
        clone.algorithm_options["nsga2_duplicate_retry_count"] = 2
        clone.algorithm_options["nsga2_duplicate_retry_mutation_scale"] = 1.5
        clone.algorithm_options["nsga2_low_g_tail_mutation_light"] = True
        clone.algorithm_options["nsga2_low_g_tail_probability"] = 0.4
        clone.algorithm_options["nsga2_low_g_tail_gene_probability"] = 0.4
        clone.algorithm_options["nsga2_low_g_tail_step_scale"] = 0.1
        clone.algorithm_options["nsga2_low_g_tail_zdt1_only"] = True
        return clone
    if variant.candidate_id == "candidate_o_spread_preserving_variation_light":
        clone.crossover = "uniform"
        clone.algorithm_options["nsga2_offspring_decision_dedup"] = True
        clone.algorithm_options["nsga2_duplicate_retry_count"] = 2
        clone.algorithm_options["nsga2_duplicate_retry_mutation_scale"] = 1.5
        clone.algorithm_options["nsga2_low_g_tail_mutation_light"] = True
        clone.algorithm_options["nsga2_low_g_tail_probability"] = 0.4
        clone.algorithm_options["nsga2_low_g_tail_gene_probability"] = 0.4
        clone.algorithm_options["nsga2_low_g_tail_step_scale"] = 0.1
        clone.algorithm_options["nsga2_low_g_tail_zdt1_only"] = True
        clone.algorithm_options["nsga2_spread_preserving_variation_light"] = True
        clone.algorithm_options["nsga2_spread_preserving_probability"] = 0.25
        clone.algorithm_options["nsga2_spread_preserving_x0_probability"] = 0.7
        clone.algorithm_options["nsga2_spread_preserving_x0_step_scale"] = 0.04
        clone.algorithm_options["nsga2_spread_preserving_tail_step_scale"] = 0.02
        clone.algorithm_options["nsga2_spread_preserving_zdt1_only"] = True
        return clone
    raise ValueError(f"Unsupported NSGA-II candidate variant: {variant.candidate_id}")


def candidate_variant_metadata(variant: NSGA2CandidateVariant) -> dict[str, object]:
    return {
        "candidate_id": variant.candidate_id,
        "base_candidate_id": variant.base_candidate_id,
        "base_candidate": variant.base_candidate_id,
        "operator_change": variant.operator_change,
        "source_diagnosis_report": variant.source_diagnosis_report,
        "source_artifacts": list(variant.source_artifacts),
        "default_changed": variant.default_changed,
        "promotion_status": variant.promotion_status,
        "expected_gain": variant.expected_gain,
        "risk": variant.risk,
        "mechanism": variant.mechanism,
        "source_hypothesis": variant.source_hypothesis,
        "allowed_use": variant.allowed_use,
        "disallowed_use": variant.disallowed_use,
        "zdt1_specific_warning": variant.zdt1_specific_warning or None,
        "candidate_parameters": dict(variant.candidate_parameters or {}),
        "variant_notes": variant.notes,
    }
