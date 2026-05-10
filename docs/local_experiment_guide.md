# Local Experiment Guide

This guide is for the main use case of this repo: quick local GA experiments on a laptop.

You do not need external benchmarks, release packaging, or claim governance for the workflow below.

## Install For Local Experimentation

Use the repo-dev path:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .[dev]
```

## Two Supported Paths

This local workflow intentionally stays small:

1. quick single run
2. small parameter sweep

## Fastest Single Runs

```bash
python scripts/run_local_experiment.py --preset onemax_small
python scripts/run_local_experiment.py --demo hybrid
python scripts/run_local_experiment.py --demo nsga2
```

What you get:

- `raw_result.json`
- `summary.csv`
- `summary.md`
- `plot_*.png` when a plot makes sense

The output root defaults to `outputs/local_runs/`.

## Most Useful Small Sweeps

```bash
python scripts/run_local_sweep.py --study onemax_mutation_study
python scripts/run_local_sweep.py --study tsp_quality_study
python scripts/run_local_sweep.py --study zdt1_nsga2_mutation_study
python scripts/run_local_sweep.py --study onemax_adaptive_mutation_study
python scripts/run_local_sweep.py --study knapsack_feasibility_study
python scripts/run_local_sweep.py --study tsp_stagnation_study
python scripts/run_local_sweep.py --study zdt1_diversity_study
```

The output root defaults to `outputs/local_studies/`.

Each study writes:

- `raw_results.csv`
- `raw_results.json`
- `summary.csv`
- `summary.md`
- `history_summary.csv`
- `ranking_detail.csv` for candidate-level fast-versus-canonical rows when ranking-fidelity
  analysis is enabled
- `ranking_fidelity.csv` for Spearman / Kendall / top-k recall summaries when ranking-fidelity is
  enabled
- `triage_workflow_summary.csv` for `always_fast`, `always_canonical`, and top-k canonical-confirm
  workflow comparisons
- `tolerance_table.csv` for quality-first versus budget-first loss-envelope summaries
- `seed_budget_table.csv` for seed-count-versus-confidence summaries on the current local defaults
- `sequential_decision_table.csv` for paired-seed `3 -> 5 -> 8 -> 10` escalation decisions and
  stop labels
- `tsp_fast_tail_summary.csv` for TSP fast-profile anti-case tail summaries against the Q/F comparator pair
- `stress_case_catalog.csv` and `stress_case_catalog.md` for pinned local failure cases and
  borderline rows that should anchor the next optimization pass
- `tail_risk_summary.csv` for mean / p90 / p95 / max stress-tail summaries on the current local
  defaults
- `stress_suite_notes.md` for short protocol-facing notes that connect the stress suite back to the
  frozen local operating rules
- `plot_convergence.png`
- `plot_primary_metric.png`
- `plot_diversity.png`
- `plot_stagnation.png`
- `plot_parameter_sweep.png` for one-axis numeric sweeps
- `plot_feasibility.png` for knapsack studies
- `plot_violation.png` for knapsack feasibility diagnostics
- `plot_route_distance.png` for TSP studies
- `plot_diversity_vs_distance.png` for TSP diversity-collapse checks
- `plot_trigger_events.png` for TSP, ZDT1, and knapsack trigger timing
- `plot_post_trigger_gain.png` for TSP trigger-gain checks
- `plot_refresh_schedule_vs_gain.png` for TSP delayed-trigger versus periodic-refresh comparisons
- `plot_refresh_volume_vs_gain.png` for TSP trigger refresh-volume checks
- `plot_collapse_onset_vs_trigger.png` for TSP collapse timing versus first trigger timing
- `plot_budget_band_vs_gap.png` for TSP reduced/current budget comparisons
- `plot_instance_feature_map.png` for TSP hard-case geometry feature maps
- `plot_feature_vs_policy_gap.png` for TSP feature-to-policy-margin inspection
- `plot_bridge_score_vs_trigger_value.png` for TSP trigger-value versus bridge-score checks
- `plot_anisotropy_vs_decay_value.png` for TSP decay-value versus anisotropy checks
- `plot_router_regret.png` for router regret against fixed policies
- `plot_budget_band_vs_policy_win.png` for policy wins by TSP budget band
- `plot_mode_switch_timeline.png` for TSP online regime-switching timelines
- `plot_diversity_vs_mode.png` for TSP diversity traces colored by active mode
- `plot_collapse_onset_vs_switch.png` for TSP collapse timing versus first mode switch
- `plot_regret_vs_policy.png` for TSP regret against the best fixed policy
- `plot_threshold_vs_hv.png` for ZDT1 threshold anatomy checks
- `plot_refresh_vs_hv.png` for ZDT1 refresh anatomy checks
- `plot_cooldown_vs_hv.png` for ZDT1 cooldown anatomy checks
- `plot_budget_band_vs_regret.png` for TSP switching regret by budget band
- `plot_rescue_target_vs_anticase_gap.png` for TSP rescue-target gain versus anti-case damage
- `plot_seed_fraction_vs_gap.png` for TSP seeded fixed-stack gap checks
- `plot_seed_source_vs_gap.png` for TSP seed-source anatomy checks
- `plot_mutation_operator_vs_gap.png` for TSP mutation-operator simplification checks
- `plot_initial_quality_vs_final_gap.png` for TSP seeded head-start versus final gap
- `plot_family_vs_regret.png` for knapsack family-conditioned regret against greedy / GA baselines
- `plot_seeded_vs_repair_gap.png` for knapsack seeding versus repair anatomy checks
- `plot_repair_vs_greedy_gap.png` for repair-only versus greedy gaps on knapsack rerun-boundary studies
- `plot_init_feasible_vs_final_gain.png` for knapsack feasible-start versus final-gain checks
- `plot_initial_feasible_fraction_vs_gain.png` for knapsack feasible-start versus gain checks
- `plot_capacity_tightness_vs_gain.png` for knapsack capacity-tightness versus gain-over-plain-GA checks
- `plot_budget_vs_feasible_gain.png` for knapsack budget-to-feasible-gain checks
- `plot_hypervolume.png` for ZDT studies
- `plot_hv_vs_spread.png` for ZDT tradeoff checks
- `plot_budget_vs_hv.png` for ZDT threshold-versus-budget checks
- `plot_budget_vs_runtime.png` for TSP budget-versus-runtime checks
- `plot_budget_vs_regret.png` for TSP quality-loss versus budget checks
- `plot_early_stop_vs_quality.png` for TSP plateau-stop quality tradeoffs
- `plot_fast_vs_canonical_rank.png` for TSP fast-versus-canonical rank fidelity checks
- `plot_topk_recall_vs_budget.png` for triage top-k recall versus budget
- `plot_triage_cost_vs_regret.png` for TSP triage workflow cost-versus-regret checks
- `plot_rescue_target_vs_anticase_rank_fidelity.png` for TSP rank-fidelity split by hard-case
  group
- `plot_q_vs_f_loss_distribution.png` for TSP route-distance loss envelopes
- `plot_rescue_vs_anticase_loss.png` for TSP rescue-target versus anti-case Q/F loss tails
- `plot_budget_savings_vs_quality_loss.png` for TSP budget savings versus route-distance loss
- `plot_budget_vs_spread.png` for ZDT spread changes across budget cuts
- `plot_early_stop_vs_hv.png` for ZDT early-stop HV checks
- `plot_fast_vs_canonical_hv_rank.png` for ZDT1 fast-versus-canonical HV ranking checks
- `plot_triage_cost_vs_hv_regret.png` for ZDT1 triage workflow cost-versus-HV regret
- `plot_spread_safety_failures.png` for ZDT1 spread / pareto safety failures during triage
- `plot_q_vs_f_loss_distribution_recalibrated.png` for the post-hardening TSP Q/F recalibration pass
- `plot_rescue_vs_anticase_loss_recalibrated.png` for recalibrated rescue-target versus anti-case TSP tails
- `plot_budget_savings_vs_quality_loss_recalibrated.png` for recalibrated TSP budget-savings versus quality-loss checks
- `plot_tolerance_accept_rate_recalibrated.png` for recalibrated TSP tolerance-bin acceptance rates
- `plot_q_vs_f_hv_loss_distribution.png` for ZDT1 HV loss envelopes
- `plot_q_vs_f_hv_loss_distribution_recheck.png` for the ZDT1 tiny Q/F freeze recheck
- `plot_population_generation_tradeoff_vs_tail.png` for TSP same-budget population/generation tradeoff versus anti-case tail
- `plot_spread_vs_joint_fail_split.png` for ZDT1 spread-fail versus joint-fail split probes
- `plot_budget_savings_vs_hv_loss.png` for ZDT1 budget savings versus HV loss
- `plot_seed_count_vs_loss_ci.png` for TSP route-distance loss CI width versus seed count
- `plot_seed_stage_vs_ci_width.png` for TSP paired-seed CI width across sequential stages
- `plot_seed_stage_vs_decision_flip.png` for TSP / ZDT1 sequential decision flips versus added
  paired seeds
- `plot_rescue_vs_anticase_escalation_rate.png` for TSP rescue-target versus anti-case escalation
  pressure in the sequential pass
- `plot_cost_savings_vs_false_decision.png` for TSP seed savings versus early wrong-call risk
- `plot_seed_count_vs_accept_rate.png` for TSP fast-profile accept rate versus seed count
- `plot_seed_count_vs_hv_ci.png` for ZDT1 HV CI width versus seed count
- `plot_seed_count_vs_safety_fail_rate.png` for ZDT1 spread / Pareto safety failure rate versus seed count
- `plot_seed_stage_vs_hv_ci.png` for ZDT1 sequential HV CI width across paired-seed stages
- `plot_seed_stage_vs_safety_fail_rate.png` for ZDT1 sequential spread / Pareto safety failures
- `plot_seed_stage_vs_repair_note_stability.png` for knapsack repair-note stability across `3 -> 5`
  paired-seed stages
- `plot_seed_count_vs_repair_note_stability.png` for knapsack repair-note stability versus seed count
- `plot_seed_count_vs_control_stability.png` for OneMax control stability versus seed count
- `plot_q_vs_f_tail_distribution.png` for TSP fast-hardening loss distributions against the quality-first comparator
- `plot_candidate_vs_anti_case_p90.png` for TSP anti-case p90 checks across fast fixed-stack variants
- `plot_candidate_vs_rescue_mean.png` for TSP rescue-target mean-loss preservation across fast variants
- `plot_seed_fraction_vs_tail.png` for TSP seed-fraction versus anti-case tail checks
- `plot_operator_vs_tail.png` for TSP mutation-operator versus anti-case tail checks
- `plot_old_fast_vs_new_fast_tail.png` for the legacy-fast versus hardened-fast TSP anti-case tail comparison
- `tsp_protocol_limitation_freeze_summary.md` for the TSP freeze readout when the remaining anti-case
  `p95/max` tail is treated as a protocol limitation rather than another same-budget contour target
- `zdt1_spread_candidate_boundary_table.csv` for slice-by-slice boundary validation of the strongest
  ZDT1 spread candidate against the current fast default
- `zdt1_spread_candidate_boundary_notes.md` for the spread-candidate boundary decision readout
- `plot_tsp_tail_freeze_summary.png` for the frozen TSP anti-case / rescue tail readout
- `plot_anticase_q_vs_f_tail.png` for the frozen TSP anti-case Q-versus-F tail overlay
- `plot_spread_candidate_vs_currentF.png` for the ZDT1 spread candidate versus current-fast
  validation slices
- `plot_spread_tail_validation.png` for ZDT1 spread-tail validation across stress, holdout, and
  stable slices
- `plot_hv_preservation_vs_spread_gain.png` for the ZDT1 spread-gain versus HV-preservation tradeoff
- `plot_joint_non_regression.png` for the ZDT1 joint-fail non-regression check during spread-candidate validation
- `plot_stable_normal_non_regression.png` for the ZDT1 stable/normal slice non-regression check
- `plot_rerun_gate_vs_regret.png` for knapsack pilot+rereun efficiency checks
- `plot_initial_feasible_fraction_vs_rerun_value.png` for knapsack rerun-value versus pilot feasibility
- `plot_final_pareto_front.png` for ZDT studies
- `plot_mutation_rate.png` for knapsack feasibility-aware mutation studies

Re-render plots from an existing study directory:

```bash
python scripts/plot_local_results.py --study-dir outputs/local_studies/<timestamp>_<study_name>
```

## Included Study Manifests

The repo includes the main local study presets below.

Core sweeps:

- `onemax_mutation_study`
- `onemax_population_study`
- `onemax_adaptive_mutation_study`
- `onemax_restart_study`
- `knapsack_penalty_study`
- `knapsack_population_study`
- `knapsack_adaptive_mutation_study`
- `knapsack_feasibility_study`
- `tsp_mutation_study`
- `tsp_quality_study`
- `tsp_adaptive_mutation_study`
- `tsp_stagnation_study`
- `zdt1_nsga2_population_study`
- `zdt1_nsga2_mutation_study`
- `zdt1_adaptive_mutation_study`
- `zdt1_diversity_study`

Focused adaptive tuning:

- `tsp_diversity_threshold_study`
- `tsp_refresh_fraction_study`
- `tsp_stagnation_window_study`
- `tsp_mutation_boost_study`
- `tsp_targeted_confirm_study`
- `zdt1_diversity_threshold_study`
- `zdt1_refresh_fraction_study`
- `zdt1_targeted_confirm_study`
- `knapsack_restart_window_study`
- `knapsack_restart_confirm_study`
- `onemax_control_adaptive_check`

Mechanism / holdout / budget passes:

- `tsp_mechanism_isolation_study`
- `tsp_mechanism_confirm_seedblock_b`
- `tsp_budget_sensitivity_study`
- `tsp_delayed_trigger_study`
- `tsp_periodic_refresh_control_study`
- `tsp_delayed_trigger_confirm_holdout`
- `tsp_budget_sensitivity_delayed_trigger`
- `zdt1_profile_holdout_study`
- `zdt1_budget_sensitivity_study`
- `zdt1_threshold_budget_rule_study`
- `zdt1_threshold_holdout_confirm_study`
- `knapsack_feasibility_mutation_study`
- `knapsack_feasibility_confirm_study`
- `knapsack_feasibility_tiny_confirm`
- `onemax_control_mechanism_check`
- `onemax_control_delayed_trigger_check`

Hard-case / note-freeze pass:

- `tsp_hardcase_trigger_suite`
- `tsp_hardcase_holdout_suite`
- `zdt1_budget_note_freeze`
- `onemax_control_hardcase_check`

Instance-aware router pass:

- `tsp_instance_feature_labeling_study`
- `tsp_policy_router_train_study`
- `tsp_policy_router_holdout_study`
- `tsp_policy_router_budget_band_study`
- `tsp_regime_switching_coarse`
- `tsp_regime_switching_holdout`
- `tsp_regime_switching_budget_band`
- `tsp_underbudget_rescue_suite`
- `tsp_underbudget_rescue_holdout`
- `tsp_underbudget_anticase_check`
- `tsp_seed_fraction_study`
- `tsp_mutation_operator_study`
- `tsp_fixed_stack_coarse`
- `tsp_fixed_stack_holdout`
- `knapsack_family_suite`
- `knapsack_seeded_repair_anatomy_study`
- `knapsack_feasibility_control_study`
- `knapsack_canonical_experimental_confirm`
- `knapsack_rerun_boundary_suite`
- `knapsack_rerun_boundary_holdout`
- `knapsack_repair_vs_restart_confirm`
- `zdt1_default_freeze_recheck`
- `zdt1_threshold_anatomy_study`
- `zdt1_refresh_anatomy_study`
- `zdt1_cooldown_anatomy_study`
- `zdt1_canonical_default_confirm`
- `tsp_ranking_fidelity_study`
- `tsp_triage_confirm`
- `zdt1_ranking_fidelity_study`
- `zdt1_triage_confirm`
- `knapsack_triage_sanity_study`
- `onemax_control_ranking_check`
- `tsp_qf_tolerance_study`
- `tsp_qf_tolerance_confirm`
- `tsp_fast_tail_hardening_study`
- `tsp_fast_tail_confirm`
- `tsp_qf_recalibration_study`
- `tsp_qf_recalibration_confirm`
- `tsp_fast_legacy_reference_check`
- `tsp_sequential_compare_study`
- `tsp_stress_suite`
- `tsp_fast_stress_hardening_study`
- `tsp_fast_stress_hardening_confirm`
- `zdt1_qf_tolerance_study`
- `zdt1_qf_tolerance_confirm`
- `zdt1_qf_tiny_freeze_check`
- `zdt1_qf_tiny_freeze_recheck`
- `zdt1_sequential_compare_study`
- `zdt1_stress_suite`
- `zdt1_fast_stress_hardening_study`
- `zdt1_fast_stress_hardening_confirm`
- `knapsack_note_freeze_check`
- `knapsack_sequential_sanity`
- `knapsack_stress_suite`
- `knapsack_stress_note_freeze_check`
- `onemax_control_freeze_check`
- `onemax_control_sequential_check`
- `onemax_control_stress_check`
- `onemax_control_stress_freeze_check`
- `tsp_seed_budget_calibration`
- `zdt1_seed_budget_calibration`
- `knapsack_seed_budget_sanity`
- `onemax_seed_budget_control`
- `tsp_default_tiny_freeze_check`
- `onemax_control_tiny_check`
- `onemax_control_fixedstack_check`
- `onemax_control_router_check`
- `onemax_control_switching_check`

They are intentionally small enough for local iteration rather than full benchmark evidence.

## Adaptive Studies Worth Running First

If you want to debug local failure modes rather than just compare final scores, start here:

```bash
python scripts/run_local_sweep.py --study onemax_adaptive_mutation_study
python scripts/run_local_sweep.py --study knapsack_feasibility_study
python scripts/run_local_sweep.py --study tsp_stagnation_study
python scripts/run_local_sweep.py --study zdt1_diversity_study
```

These studies keep the configured evaluation budget matched across variants and add:

- convergence diagnostics per generation
- diversity signals per representation
- stagnation tracking
- adaptive mutation / restart / diversity injection comparisons

Current local read from the broader adaptive pass:

- onemax:
  - fixed mutation stayed as good as or better than adaptive schedules at the tested local budget
- knapsack:
  - restart only showed a weak exploratory signal
- tsp:
  - low-diversity injection and light mutation schedules reduced route-distance stagnation
- zdt1:
  - low-diversity injection improved hypervolume on the tested local study

Treat these as local study directions, not broad defaults.

## Hard-Case TSP / Tiny ZDT1 Note Pass: Start Here

This pass keeps the adaptive family set fixed and asks narrower questions:

1. on small local TSP hard cases, is the current trigger ever better than `decay_mutation` or `none`?
2. when the final route distance ties, does the trigger still show selective utility after collapse?
3. for ZDT1, should `0.55` stay the local default while `0.60` stays only a nearby threshold note?
4. should knapsack adaptive stay parked?
5. does onemax still behave like a control problem?

Run these first:

```bash
python scripts/run_local_sweep.py --study tsp_hardcase_trigger_suite
python scripts/run_local_sweep.py --study tsp_hardcase_holdout_suite
python scripts/run_local_sweep.py --study zdt1_budget_note_freeze
python scripts/run_local_sweep.py --study onemax_control_hardcase_check
python scripts/run_local_experiment.py --config configs/local_profiles/tsp_diversity_injection.json
python scripts/run_local_experiment.py --config configs/local_profiles/zdt1_diversity_injection.json
python scripts/run_local_experiment.py --config configs/local_profiles/knapsack_restart_experimental.json
```

All of these stayed on matched configured evaluation budgets, and this pass still kept
`extra_evaluations_from_adaptation = 0`. Triggered refresh or mutation boosts only reused the normal
next-generation budget; they did not add extra objective calls on top of the configured budget.

### Current problem-specific read

- tsp:
  - keep `configs/local_profiles/tsp_diversity_injection.json` as the focused-budget reusable local
    profile, but read it narrowly
  - the hard-case suite showed that `low_diversity_injection` is not a broad local TSP default:
    it won the bridge-like holdout at `60` generations, but lost the corridor-style holdout to
    `decay_mutation` and stayed close to `none`
  - the trigger still tends to fire well before the measured collapse onset, so it reads more like
    a focused low-budget heuristic than a clean late rescue policy
  - use `plot_collapse_onset_vs_trigger.png`, `plot_post_trigger_gain.png`, and
    `plot_refresh_volume_vs_gain.png` together; they tell you whether the trigger was selective or
    just another way to keep exploration alive
  - `decay_mutation` remains the clean schedule-only alternative, especially on corridor-like hard
    cases and reduced budgets
  - `periodic_refresh_control` stays a mechanism probe only; it helps interpret refresh volume, not
    define a reusable rule
  - in the router pass, cheap feature maps did explain part of the split, but the train-selected
    simple rule still matched `always_low_diversity_injection` on holdout and lost to
    `always_decay_mutation`
  - read `plot_instance_feature_map.png`, `plot_bridge_score_vs_trigger_value.png`, and
    `plot_anisotropy_vs_decay_value.png` together before inventing a router rule of your own
  - the underbudget rescue-specialization pass then asked one last narrow question:
    can switching survive as a rescue-target-only local profile once bridge/ring-like cases and
    corridor anti-cases are read separately?
  - the answer stayed no:
    rescue-target rows still preferred `low_diversity_injection` at `36` generations and
    `decay_mutation` at `60`, while `switch_controller_v1` only won one bridge-like holdout
    (`twin_bridge_holdout_15`) and lost the other (`bridge_spoke_holdout_18`)
  - the dedicated anti-case check kept the downside visible:
    at `36` generations `switch_controller_v1` trailed `decay_mutation` by about `3.77` route
    units on average, and at `45` generations by about `4.69`
  - use `case_group_summary.csv` plus `plot_rescue_target_vs_anticase_gap.png` to read this pass:
    if a switching rule becomes worth keeping later, it needs to move left and down on that plot
  - keep switching as an experimental mechanism probe only; do not create a separate underbudget
    switching profile from this pass
- zdt1:
  - keep `configs/local_profiles/zdt1_diversity_injection.json` as the reusable local default:
    `diversity_threshold=0.55`, `refresh_fraction=0.10`, `adaptation_cooldown=4`
  - the anatomy pass split the current profile into threshold / refresh / cooldown pieces:
    `threshold=0.55` stayed the clean center, `refresh_fraction=0.10` matched the old `0.20`
    profile closely enough to simplify the default, and `cooldown=4` stayed the safest middle
    ground over `2` or `6`
  - `threshold=0.45` and `cooldown=6` each produced coarse wins, but they did not survive cleanly
    enough across the holdout budget bands to replace the simpler `0.55 / 0.10 / 4` default
  - keep `0.60` only as a nearby exploratory threshold, not as a promoted replacement or a budget
    rule
- knapsack:
  - keep adaptive work parked for now
  - `configs/local_profiles/knapsack_restart_experimental.json` remains an experimental rerun
    recipe, not a reusable profile
- onemax:
  - the control rerun kept `none` ahead on evaluations-to-target
  - `switch_controller_v1` fired occasionally but still finished slower than the fixed baseline
  - keep onemax as a fixed-baseline control problem unless you are debugging instrumentation

## TSP Instance-Aware Router Pass

The next narrow question after the hard-case trigger pass is not "which policy wins overall?"
but "when should you even try `low_diversity_injection` instead of `decay_mutation` or `none`?"

This pass keeps the candidate set deliberately small:

- `none`
- `low_diversity_injection`
- `decay_mutation`

Run it like this:

```bash
python scripts/run_local_sweep.py --study tsp_policy_router_train_study
python scripts/run_local_sweep.py --study tsp_policy_router_holdout_study
python scripts/run_local_sweep.py --study tsp_policy_router_budget_band_study
python scripts/run_tsp_router_analysis.py --train-study-dir outputs/local_studies/<train_dir> --holdout-study-dir outputs/local_studies/<holdout_dir> --budget-study-dir outputs/local_studies/<budget_dir>
python scripts/run_local_sweep.py --study zdt1_default_freeze_recheck
python scripts/run_local_sweep.py --study onemax_control_router_check
```

The router analysis bundle writes:

- `instance_features.csv`
- `policy_labels_train.csv`
- `policy_labels_holdout.csv`
- `summary.csv`
- `summary.md`
- `router_decision_table.md`
- `plot_instance_feature_map.png`
- `plot_feature_vs_policy_gap.png`
- `plot_bridge_score_vs_trigger_value.png`
- `plot_anisotropy_vs_decay_value.png`
- `plot_router_regret.png`
- `plot_budget_band_vs_policy_win.png`

### Cheap TSP geometry features used

The router pass uses deterministic, geometry-only features with no extra objective calls:

- `bridge_score`
  - MST longest-edge / median-edge ratio
- `pca_anisotropy_ratio`
  - cheap corridor-like elongation proxy
- `nn_distance_cv`
  - nearest-neighbor distance variation
- supporting context:
  - `num_cities`
  - `bbox_aspect_ratio`
  - `radial_distance_cv`

### Current read from the router pass

- bridge/ring-like hard cases at the focused budget still align with
  `low_diversity_injection`
- corridor-like hard cases continue to align with `decay_mutation`
- the features above explain part of that split, but the best simple threshold rule did not beat
  the best fixed holdout policy
- keep the router as a local diagnostic tool for now; do not promote it to a reusable local rule

In practice:

- if the case looks bridge-like or ring-like and you are staying on the focused budget, try
  `configs/local_profiles/tsp_diversity_injection.json`
- if the case is strongly corridor-like or elongated, keep `decay_mutation` as the cleaner local
  comparator
- if you want a broad automatic rule, the honest answer is still: not yet

## TSP Online Regime-Switching Pass

After the cheap geometry router failed to beat the best fixed holdout policy, the next narrow
question became: can runtime diagnostics switch between `decay_mutation` and
`low_diversity_injection` more effectively than either fixed policy?

This pass keeps the candidate set tight:

- `none`
- `decay_mutation`
- `low_diversity_injection`
- `switch_controller_v1`
- `switch_controller_v2`

Run it like this:

```bash
python scripts/run_local_sweep.py --study tsp_regime_switching_coarse
python scripts/run_local_sweep.py --study tsp_regime_switching_holdout
python scripts/run_local_sweep.py --study tsp_regime_switching_budget_band
python scripts/run_local_sweep.py --study tsp_underbudget_rescue_suite
python scripts/run_local_sweep.py --study tsp_underbudget_rescue_holdout
python scripts/run_local_sweep.py --study tsp_underbudget_anticase_check
python scripts/run_local_sweep.py --study zdt1_default_freeze_recheck
python scripts/run_local_sweep.py --study onemax_control_switching_check
```

### What the switching pass showed

- reduced-budget holdout:
  - `switch_controller_v1` was the best average regret path at `36` generations
  - average regret was about `0.46`, versus `1.47` for `low_diversity_injection` and `1.66` for
    `decay_mutation`
  - this is the one place where online switching looked genuinely useful
- current focused budget:
  - the switching controllers did not hold up
  - `none` actually had the best average regret on the holdout set, with
    `low_diversity_injection` next and both switching variants behind
- budget-band pass:
  - `decay_mutation` had the best overall average regret
  - `switch_controller_v2` only matched the oracle in a couple of underbudget cases mostly because
    it never switched out of decay mode

### How to read the switching plots

- `plot_mode_switch_timeline.png`
  - check whether switches happen as late rescue events or just early mode flips
- `plot_diversity_vs_mode.png`
  - check whether trigger mode only engages after diversity collapse
- `plot_collapse_onset_vs_switch.png`
  - check whether the first switch happens after collapse onset or long before it
- `plot_regret_vs_policy.png`
  - compare switching against the oracle best fixed policy, not just raw distance
- `plot_budget_band_vs_regret.png`
  - verify whether switching only helps in reduced-budget hard cases
- `plot_rescue_target_vs_anticase_gap.png`
  - check whether rescue-target gain is large enough to justify the anti-case damage

### Current local read after switching

- keep `configs/local_profiles/tsp_diversity_injection.json` as the narrow reusable local profile
  for focused-budget bridge/ring-like hard cases
- keep `decay_mutation` as the clean schedule-only comparator, especially on corridor-like cases
  and when you want the lowest-regret broad fixed policy on the switching holdout suite
- keep `switch_controller_v1` only as an experimental reduced-budget mechanism probe
  - it is more interpretable than the earlier early-fire trigger because it usually switches later
    in the run
  - but it did not beat the best fixed policy across the focused-budget holdout
- keep `switch_controller_v2` as a rescue-window probe only
  - when it looked good, it often did so by effectively staying in decay mode
- the underbudget rescue-specialization pass did not justify a separate switching profile either
  - `switch_controller_v1` still won the `twin_bridge_holdout_15` rescue-target case at `36`,
    `45`, and `60` generations
  - but `bridge_spoke_holdout_18` still preferred fixed policies, and the corridor anti-case check
    showed meaningful damage against `decay_mutation`
- the honest answer is still: there is no simple broad online switching rule yet, and there is no
  clean underbudget-only switching profile yet either

## How To Read The New Diagnostics

Across all problems, the history output now includes:

- `best_fitness`, `mean_fitness`, `median_fitness`
- `improvement_delta`
- `generations_since_last_improvement`
- `recent_window_improvement`
- `recent_window_slope`

Representation-specific diversity is also tracked:

- bitstring:
  - `allele_entropy`
  - `sampled_mean_hamming_distance`
- permutation / TSP:
  - `positional_diversity`
  - `edge_diversity_ratio`
- real / ZDT:
  - `mean_coordinate_variance`
  - `population_spread`

TSP hard-case studies also summarize trigger selectivity:

- `collapse_onset_generation`
- `trigger_delay_from_collapse`
- `time_to_first_nontrivial_improvement_after_trigger`
- `useless_trigger_rate`
- `realized_refresh_volume`

Use them like this:

- flat convergence plus falling diversity:
  - premature convergence
- long `generations_since_last_improvement` with stable diversity:
  - search is wandering without finding useful steps
- knapsack feasible ratio collapsing:
  - mutation pressure is too destructive for the current budget
- ZDT hypervolume improving while spread worsens:
  - the policy is trading diversity for front concentration

## How To Read The Results

### onemax

Look at:

- `target_hit_rate`
- `generations_to_target_mean`
- convergence plot

If settings tie on hit rate, prefer the one that reaches the target in fewer generations.

Adaptive read:

- compare `evaluations_to_target_mean` first
- use `plot_stagnation.png` to see whether a mutation boost only moves later generations around
- use `plot_diversity.png` to see whether the run is already collapsing before target reach

### knapsack

Look at:

- `best_feasible_fitness_mean`
- `feasible_rate`
- `mean_violation_mean`

Do not judge a setup only by penalized `best_fitness` when feasibility is unstable.

Adaptive read:

- use `plot_feasibility.png` together with `plot_stagnation.png`
- use `plot_violation.png` when restart seems to help only because feasibility recovered late
- use `plot_mutation_rate.png` to see whether a feasibility-aware boost actually engaged
- use `plot_trigger_events.png` to see whether restart or mutation-boost triggers are firing too
  late to matter
- if best feasible fitness rises while `feasible_ratio` collapses, the policy is too aggressive
- if mutation-rate boosts do not fire or only edge `none` in coarse runs but lose to restart in
  confirm, keep them experimental
- if restart only ties `none`, park the adaptive branch instead of promoting it

### tsp

Look at:

- `best_route_distance_mean`
- `best_route_distance_std`
- convergence plot

`best_fitness` is only the internal sign-flipped objective. Distance is the user-facing metric.

Adaptive read:

- use `plot_route_distance.png` as the main quality plot
- use `plot_diversity_vs_distance.png` to see whether distance only improves after diversity
  collapse
- use `plot_trigger_events.png` to check whether a trigger fires too early, too often, or only
  after the run is already effectively done
- use `plot_post_trigger_gain.png` to see whether injection actually buys recovery after the first
  trigger or only mirrors a decay schedule
- use `plot_refresh_schedule_vs_gain.png` when comparing delayed trigger semantics against periodic
  refresh control
- use `plot_collapse_onset_vs_trigger.png` to check whether first fire happens after measured
  collapse or long before it
- use `plot_refresh_volume_vs_gain.png` and `plot_budget_band_vs_gap.png` to see whether refresh
  volume only helps on reduced/current budget hard cases
- if `trigger_delay_from_collapse_mean` stays strongly negative, the trigger is acting like an
  early exploration schedule, not a late rescue
- if `useless_trigger_rate_mean` stays high while final score only ties `decay_mutation`, prefer
  `decay_mutation` as the cleaner local alternative
- if bridge-like hard cases improve but corridor-like hard cases do not, keep
  `tsp_diversity_injection.json` only as a focused hard-case heuristic instead of a broad local
  default

### zdt1

Look at:

- `hypervolume_mean`
- `pareto_ratio_mean`
- `spread_mean`
- final Pareto scatter

Do not collapse ZDT into one scalar `best_fitness` reading.

Adaptive read:

- use `plot_hypervolume.png` and `plot_diversity.png` together
- use `plot_hv_vs_spread.png` to check whether an HV gain is only coming from front compression
- use `plot_trigger_events.png` to see whether the diversity profile is firing almost every few
  generations
- use `plot_budget_vs_hv.png` before inventing a new threshold rule
- if the holdout block still leaves `0.55 / 0.10 / cooldown=4` on top, keep the current profile
- if a nearby threshold wins only on one seed block or one budget tier, keep it exploratory
- if a budget-up run does not consistently keep `0.6` on top across seed blocks, do not promote a
  budget-conditioned threshold rule

## TSP Fixed-Stack Simplification Pass

The current local TSP question is no longer "can a smarter trigger fire later?" but
"does the trigger do anything that a simpler fixed stack cannot already do?"

Run this pass like this:

```bash
python scripts/run_local_sweep.py --study tsp_seed_fraction_ablation_study
python scripts/run_local_sweep.py --study tsp_seed_source_ablation_study
python scripts/run_local_sweep.py --study tsp_mutation_operator_ablation_study
python scripts/run_local_sweep.py --study tsp_canonical_default_confirm
python scripts/run_local_sweep.py --study zdt1_threshold_anatomy_study
python scripts/run_local_sweep.py --study zdt1_refresh_anatomy_study
python scripts/run_local_sweep.py --study zdt1_cooldown_anatomy_study
python scripts/run_local_sweep.py --study zdt1_canonical_default_confirm
python scripts/run_local_sweep.py --study knapsack_family_suite
python scripts/run_local_sweep.py --study knapsack_seeded_repair_anatomy_study
python scripts/run_local_sweep.py --study knapsack_feasibility_control_study
python scripts/run_local_sweep.py --study knapsack_canonical_experimental_confirm
python scripts/run_local_sweep.py --study tsp_default_tiny_freeze_check
python scripts/run_local_sweep.py --study onemax_control_tiny_check
python scripts/run_local_experiment.py --config configs/local_profiles/tsp_seeded_swap_local.json
python scripts/run_local_experiment.py --config configs/local_profiles/zdt1_diversity_injection.json
python scripts/run_local_experiment.py --config configs/local_profiles/knapsack_repair_local_experimental.json
```

What this pass isolates:

- whether `seed_fraction` alone recovers most of the old trigger gain
- whether the seeding source itself matters more than the nominal `hybrid_ga` wrapper
- whether `swap` versus `inversion` changes the need for a trigger
- whether rescue-target bridge/ring-like cases still need trigger intervention once the initial
  stack is strengthened
- whether corridor-like anti-cases are safer with a simple fixed stack

### Current read from the fixed-stack pass

- `configs/local_profiles/tsp_seeded_swap_local.json` is now the simplest reusable local TSP
  profile for the current hard-case suite
- the profile is:
  - `hybrid_ga`
  - `swap` mutation
  - `tsp_nearest_neighbor_mix`
  - `seed_fraction=0.5`
  - `adaptive_policy=none`
  - `local_search_strategy=none`
- the seeding-anatomy pass tightened that read:
  - nearest-neighbor mix seeding is the dominant contributor
  - `hybrid_ga` without that seed source falls back toward `none`, so the wrapper alone is not the
    main gain source
  - `seed_fraction=0.25` stays close on rescue-target cases, but `seed_fraction=0.5` remains safer
    on anti-cases and on the mixed holdout average
  - `seed_fraction=0.75` looked strong in the coarse pass, but it increases complexity rather than
    simplifying the local default, so it stays exploratory
  - `seeded_inversion_seed25` stayed competitive, but `seeded_swap_seed50` still won the mixed
    holdout summary and keeps the current mutation operator unchanged
- read `plot_seed_source_vs_gap.png` first:
  if the no-seed hybrid rows sit near `none`, the dominant effect is the seeding source itself
- read `plot_seed_fraction_vs_gap.png` next:
  if the curve still slopes between `0.25` and `0.5` on anti-cases or the mixed holdout summary,
  keep `0.5` as the canonical default
- read `plot_mutation_operator_vs_gap.png` after that:
  if inversion only wins on one subset while `swap` keeps the better mixed holdout average, keep
  swap as the default mutation
- use `plot_initial_quality_vs_final_gap.png` to see whether better final tours are mostly coming
  from a stronger starting population rather than later trigger-like rescue behavior

## ZDT1 Profile Anatomy / Canonical Freeze

Run these when you want to simplify the ZDT1 default rather than widen the adaptive family:

```bash
python scripts/run_local_sweep.py --study zdt1_threshold_anatomy_study
python scripts/run_local_sweep.py --study zdt1_refresh_anatomy_study
python scripts/run_local_sweep.py --study zdt1_cooldown_anatomy_study
python scripts/run_local_sweep.py --study zdt1_canonical_default_confirm
python scripts/run_local_sweep.py --study tsp_default_tiny_freeze_check
python scripts/run_local_sweep.py --study onemax_control_tiny_check
python scripts/run_local_experiment.py --config configs/local_profiles/zdt1_diversity_injection.json
```

What this pass isolates:

- whether `diversity_threshold=0.55` is still the center of the local ZDT1 profile
- whether `refresh_fraction=0.10` is enough to keep the hypervolume gains that used to need `0.20`
- whether `adaptation_cooldown=4` is actually doing useful work versus `2` or `6`
- whether a simpler default can survive the reduced/current/+10% local budget bands on a holdout
  seed block
- whether TSP, knapsack, and onemax should simply stay frozen while ZDT1 is simplified

### Current read from the ZDT1 anatomy pass

- `configs/local_profiles/zdt1_diversity_injection.json` is now the canonical reusable local ZDT1
  profile at:
  - `diversity_threshold=0.55`
  - `refresh_fraction=0.10`
  - `adaptation_cooldown=4`
- threshold anatomy:
  - `0.45` produced the strongest coarse hypervolume at the `88`-generation band, but it did not
    stay clean enough across the confirm holdout budgets to replace `0.55`
  - `0.60` remains a nearby note-only threshold, not a promoted default
- refresh anatomy:
  - `refresh_fraction=0.10` was the clean simplification winner
  - it reduced intervention volume, stayed competitive with the old `0.20` profile on the holdout
    block, and slightly improved the mixed-budget average hypervolume
- cooldown anatomy:
  - `cooldown=6` produced a coarse win, but it did not beat the simpler `4` cleanly enough in the
    confirm holdout
  - `cooldown=2` triggered too often and did not keep a better mixed-budget result
- read `plot_threshold_vs_hv.png`, `plot_refresh_vs_hv.png`, and `plot_cooldown_vs_hv.png`
  together:
  if the threshold curve stays flat near `0.55`, the refresh curve favors `0.10`, and the cooldown
  curve stays shallow around `4`, keep the simplified `0.55 / 0.10 / 4` default frozen
- use `plot_hv_vs_spread.png` and `plot_budget_vs_hv.png` before changing the default again:
  if a candidate only wins at one budget band while making spread noisier elsewhere, keep it as an
  exploratory note instead of a promoted replacement

## Knapsack Local Structure / Park-Or-Promote Pass

This pass asks one narrow question only: is there any knapsack local profile worth keeping once
you separate seeding, repair, and feasibility-control effects under the same configured budget?

Run it like this:

```bash
python scripts/run_local_sweep.py --study knapsack_family_suite
python scripts/run_local_sweep.py --study knapsack_seeded_repair_anatomy_study
python scripts/run_local_sweep.py --study knapsack_feasibility_control_study
python scripts/run_local_sweep.py --study knapsack_canonical_experimental_confirm
python scripts/run_local_sweep.py --study knapsack_rerun_boundary_suite
python scripts/run_local_sweep.py --study knapsack_rerun_boundary_holdout
python scripts/run_local_sweep.py --study knapsack_repair_vs_restart_confirm
python scripts/run_local_sweep.py --study tsp_default_tiny_freeze_check
python scripts/run_local_sweep.py --study zdt1_default_tiny_freeze_check
python scripts/run_local_sweep.py --study zdt1_qf_tiny_freeze_check
python scripts/run_local_sweep.py --study onemax_control_tiny_check
python scripts/run_local_experiment.py --config configs/local_profiles/knapsack_repair_local_experimental.json
```

What this pass isolates:

- whether the local gain comes from cheap heuristic seeding, simple capacity repair, or both
- whether feasibility-aware mutation is more useful than the older restart probe
- whether any GA-ish local path is worth keeping even if it does not replace
  `greedy_local_search`
- whether the honest outcome is still "park knapsack" or a narrower family-conditioned
  experimental profile

### Current read from the knapsack anatomy pass

- broad knapsack adaptive work still stays parked
- one narrow local profile is worth keeping as an experimental rerun recipe:
  `configs/local_profiles/knapsack_repair_local_experimental.json`
- the dominant anatomy result was:
  - `repair_only` matched `seeded_repair`
  - `seeded_only` was usually weaker and often only recovered the greedy head start
  - so the main gain source was the repair step, not the seed source
- use `plot_seeded_vs_repair_gap.png` first:
  if `repair_only` and `seeded_repair` sit on top of each other while `seeded_only` trails them,
  repair is doing the real work
- the confirm holdout kept that read:
  - on `subset_sum_like_small_b`, `repair_only` and `seeded_repair` both reached `80.4`
    while `none` stayed at `79.87` and restart at `79.93`
  - on `tight_capacity_small_b`, the same repair paths reached `87.0` while `none` and restart
    stayed at `85.6`
  - on `weakly_correlated_small_b`, repair tied greedy at `200.0` instead of clearly beating it
- the rerun-boundary pass tightened that into a narrow note:
  - `repair_only` is still the only knapsack rerun path worth retrying
  - `seeded_repair` no longer buys a second slot; it matched `repair_only` again on the holdout
    rerun suite while costing more runtime
  - `restart` and `feasibility_aware_mutation_v1/v3` stayed behind `repair_only` on the mixed
    holdout confirm and did not create a cleaner family boundary
  - the simplest usable local note is:
    if a small instance looks subset-sum-like or tight-capacity-like, especially when a plain-GA
    pilot has very low `initial_feasible_fraction` (around `<= 0.05` on the tested local suite),
    one `repair_only` rerun is worth trying
  - weakly correlated holdout cases still only tied greedy rather than beating it, so there is
    still no broad knapsack default and no strong rerun rule outside that narrow family slice
- that means the honest local interpretation is:
  - keep `greedy_local_search` as the practical comparator
  - keep `knapsack_repair_local_experimental.json` only for subset-sum-like and tight-capacity
    small local families where fast feasibility recovery is the point
  - keep `knapsack_restart_experimental.json` as a legacy experimental note rather than a promoted
    profile
  - do not read repair-only as a broad knapsack default
- feasibility-control anatomy did not justify promotion:
  - feasibility-aware mutation was more interpretable than restart, but it did not beat restart
    cleanly enough on the mixed family pass
  - use `plot_feasibility.png`, `plot_violation.png`, and `plot_mutation_rate.png` together:
    if mutation boosts barely fire or only match `none` while repair jumps straight to feasibility,
    keep the mutation-control branch experimental
- read `plot_family_vs_regret.png` and `case_group_summary.csv` together:
  if a candidate only wins on subset-sum-like / tight-capacity families and merely ties greedy on
  weakly correlated ones, keep it narrow and family-conditioned
- read `plot_repair_vs_greedy_gap.png`, `plot_initial_feasible_fraction_vs_gain.png`, and
  `plot_capacity_tightness_vs_gain.png` together:
  if the gain-over-greedy points only separate from zero at low `capacity_ratio` and very low
  initial feasible fraction, keep the rerun note narrow and do not generalize it into a default

### Current freeze for the other problems

- tsp:
  - keep `configs/local_profiles/tsp_seeded_swap_local.json` as the canonical local default
- zdt1:
  - keep `configs/local_profiles/zdt1_diversity_injection.json` frozen at
    `0.55 / 0.10 / cooldown=4`
- onemax:
  - keep the fixed baseline as the control problem

### What remains experimental

- `configs/local_profiles/tsp_diversity_injection.json` stays as a narrow mechanism-comparison
  profile, not as the preferred reusable local TSP profile
- `decay_mutation` remains the strong fixed comparator when you want a schedule-only baseline
- delayed / periodic / switching variants stay historical mechanism probes unless a later local
  pass shows they beat the seeded fixed stack on the same hard-case suite

## Budget Frontier / Fast Profiles

This pass asks one narrow local question only: once the canonical profiles are already chosen, how
much budget can you cut before the quality loss stops being worth it?

Run it like this:

```bash
python scripts/run_local_sweep.py --study tsp_budget_frontier_study
python scripts/run_local_sweep.py --study tsp_fast_profile_confirm
python scripts/run_local_sweep.py --study zdt1_budget_frontier_study
python scripts/run_local_sweep.py --study zdt1_fast_profile_confirm
python scripts/run_local_sweep.py --study knapsack_rerun_gate_efficiency_study
python scripts/run_local_sweep.py --study knapsack_rerun_gate_confirm
python scripts/run_local_sweep.py --study onemax_control_budget_check
python scripts/run_local_sweep.py --study tsp_two_stage_gate_study
python scripts/run_local_sweep.py --study tsp_two_stage_gate_confirm
python scripts/run_local_sweep.py --study zdt1_two_stage_gate_study
python scripts/run_local_sweep.py --study zdt1_two_stage_gate_confirm
python scripts/run_local_sweep.py --study knapsack_rerun_gate_sanity_study
python scripts/run_local_sweep.py --study tsp_restart_portfolio_study
python scripts/run_local_sweep.py --study tsp_restart_portfolio_confirm
python scripts/run_local_sweep.py --study zdt1_restart_portfolio_study
python scripts/run_local_sweep.py --study zdt1_restart_portfolio_confirm
python scripts/run_local_sweep.py --study knapsack_multistart_sanity_study
python scripts/run_local_sweep.py --study onemax_control_portfolio_check
python scripts/run_local_experiment.py --config configs/local_profiles/tsp_seeded_swap_local.json
python scripts/run_local_experiment.py --config configs/local_profiles/tsp_seeded_swap_local_fast.json
python scripts/run_local_experiment.py --config configs/local_profiles/zdt1_diversity_injection.json
python scripts/run_local_experiment.py --config configs/local_profiles/zdt1_diversity_injection_fast.json
```

What this pass isolates:

- for TSP, whether cheaper generation cuts, population cuts, or one simple plateau-stop wrapper can
  stay close to the canonical seeded-swap stack
- for ZDT1, whether cheaper generation or population cuts can preserve HV / Pareto-ratio / spread
  under the current `0.55 / 0.10 / 4` diversity rule
- for knapsack, whether a cheap pilot plus conditional `repair_only` rerun is more budget-efficient
  than simply keeping the narrow repair rerun note
- for onemax, whether the fixed baseline still behaves like the control even when a budget check is
  present

### Current read from the budget frontier pass

- tsp:
  - quality-first stays `configs/local_profiles/tsp_seeded_swap_local.json`
  - budget-first now has a cleaner fast profile:
    `configs/local_profiles/tsp_seeded_swap_local_fast.json`
  - the fast profile keeps the same seeded-swap stack but now uses the hardened fixed cut:
    `population_size=40`, `generations=33`
  - on the post-hardening recalibration it kept configured budget near 25% below Q while holding
    overall mean route-distance loss near `0.63%`, rescue-target mean loss near `0.59%`, and
    anti-case `p90` loss near `2.60%`
  - read that as the current freeze result:
    `pop40/gen33` is now the least-bad mixed holdout tradeoff at the fast budget, but it still
    does not close the anti-case tail enough to promote a global numeric tolerance rule
  - the more aggressive `generations_050` squeeze stayed viable at roughly 50% budget, but it
    damaged anti-cases more and stays a budget-squeeze note rather than the canonical fast default
  - the plateau early-stop wrapper did save budget, but it was both noisier and weaker than the
    simpler fixed population cut, so it stays experimental
- zdt1:
  - quality-first stays `configs/local_profiles/zdt1_diversity_injection.json`
  - budget-first now has a cleaner fast profile:
    `configs/local_profiles/zdt1_diversity_injection_fast.json`
  - the fast profile keeps the same `0.55 / 0.10` rule, cuts population from `60` to `45`, and
    now uses `cooldown=3`
  - on the post-hardening recalibration it cut configured budget by about 25% with mean HV loss
    near `0.09%`, `p90` near `0.31%`, flat `pareto_ratio`, and about `20%` spread / joint safety
    failures
  - `population_050` stayed surprisingly strong as a half-budget squeeze, but it enlarged HV
    variance enough that it is better kept as an aggressive note than as the canonical fast default
  - generation cuts were consistently less clean than population cuts because spread degraded much
    faster
  - the HV plateau early-stop wrapper did not trigger usefully on the tested seeds, so there is no
    promoted ZDT1 early-stop rule
- knapsack:
  - there is still no broad knapsack default
  - `configs/local_profiles/knapsack_repair_local_experimental.json` remains the only narrow rerun
    note worth keeping
  - the cheap pilot + conditional rerun gate did become more selective after tightening the pilot
    signals: it reran on boundary-like subset-sum / tight-capacity families and skipped the
    anti-boundary weakly correlated holdout
  - but as a budget-efficiency rule it still failed the broad usefulness test:
    when the rerun fired it simply spent the full repair budget again, and when it did not fire the
    pilot-only path gave back too much quality on anti-boundary families
  - keep the gate experimental, and keep the practical note narrow:
    if the family looks subset-sum-like or tight-capacity-like and the pilot starts with very low
    `initial_feasible_fraction` (around `<= 0.05` on the tested suite), a full `repair_only` rerun
    is still worth trying once
- onemax:
  - keep the fixed `none` baseline as both quality-first and budget-first control
  - the 25-generation control check already hit target on every seed, but that is still a control
    sanity note rather than a new promoted fast profile

### Quality-First vs Budget-First

- onemax:
  - quality-first: `none`
  - budget-first: `none`
- tsp:
  - quality-first: `configs/local_profiles/tsp_seeded_swap_local.json`
  - budget-first: `configs/local_profiles/tsp_seeded_swap_local_fast.json`
- zdt1:
  - quality-first: `configs/local_profiles/zdt1_diversity_injection.json`
  - budget-first: `configs/local_profiles/zdt1_diversity_injection_fast.json`
- knapsack:
  - quality-first: `greedy_local_search` as the practical comparator plus the narrow
    `repair_only` rerun note
  - budget-first: no promoted fast profile and no promoted rerun gate

Read the new plots like this:

- for TSP, check `plot_budget_vs_regret.png`, `plot_budget_vs_runtime.png`, and
  `plot_early_stop_vs_quality.png` together:
  if the fixed cut is below the plateau-stop curve on both runtime and regret, prefer the fixed cut
- for ZDT1, check `plot_budget_vs_hv.png`, `plot_budget_vs_spread.png`, and `plot_hv_vs_spread.png`
  together:
  if population cuts keep the HV line near-flat while generation cuts bend spread upward, prefer
  the population cut
- for knapsack, check `plot_rerun_gate_vs_regret.png` and
  `plot_initial_feasible_fraction_vs_rerun_value.png` together:
  if the gate only helps when it fully reruns the repair path and hurts when it skips, keep the
  note narrow and do not turn it into a default

### Two-Stage Pilot / Escalation

This pass asked one narrow follow-up question only: once a quality-first canonical profile and a
budget-first fast profile already exist, can a simple "run fast first, then escalate if needed"
rule beat that manual split?

Current read:

- tsp:
  - no promoted smart gate
  - `pilot_then_canonical_v1` could recover canonical-quality routes, but only by escalating so
    often that average total evaluations rose above `always_canonical`
  - `pilot_then_canonical_v2` spent a little less than canonical on the holdout confirm, but the
    savings were too small to justify the added rule complexity
- zdt1:
  - no promoted smart gate
  - the tested gates either spent more evaluations than `always_canonical` or stayed too close to
    `always_fast` to justify a separate local rule
- knapsack:
  - keep the narrow manual `repair_only` rerun note
  - the pilot+rereun sanity probe did not beat that simpler note once full rerun cost was counted
- onemax:
  - keep `none` as a fixed control; the gate logic still adds complexity without practical value

Read the two-stage plots like this:

- tsp:
  - `plot_pilot_fraction_vs_regret.png` shows whether a larger pilot fraction actually buys better
    regret
  - `plot_escalation_rate_vs_budget.png` shows whether the gate is drifting back toward canonical
    cost
  - `plot_false_keep_vs_false_escalate.png` is the cleanest summary of gate selectivity
  - `plot_actual_eval_vs_quality.png` tells you whether the gate found a genuinely better
    quality/budget corner
- zdt1:
  - `plot_pilot_fraction_vs_hv.png` shows whether longer pilots preserve more HV
  - `plot_escalation_rate_vs_hv.png` shows whether extra escalations are really paying for
    themselves
  - `plot_actual_eval_vs_hv.png` is the fastest way to see whether a gate beats the current
    quality-first vs budget-first split
- knapsack:
  - `plot_rerun_gate_vs_regret.png` and `plot_actual_eval_vs_feasible_gain.png` together show
    whether a pilot+rereun gate is actually more efficient than the existing narrow manual note

Practical recommendation:

- when quality matters most, run the canonical profile directly
- when you are in a quick local tuning loop, run the fast profile directly
- keep the explicit split for now; do not add a promoted `*_smart.json` gate until a later pass
  cuts average actual evaluations by a clear margin without giving back the quality the current
  confirm runs protected

### Multi-Start Portfolio Allocation

This pass asked one narrow follow-up question only: with the same total evaluation budget, should
you run one longer quality-first/canonical pass or several shorter budget-first/fast passes?

Run it like this:

```bash
python scripts/run_local_sweep.py --study tsp_restart_portfolio_study
python scripts/run_local_sweep.py --study tsp_restart_portfolio_confirm
python scripts/run_local_sweep.py --study zdt1_restart_portfolio_study
python scripts/run_local_sweep.py --study zdt1_restart_portfolio_confirm
python scripts/run_local_sweep.py --study knapsack_multistart_sanity_study
python scripts/run_local_sweep.py --study onemax_control_portfolio_check
```

Current read:

- tsp:
  - keep `configs/local_profiles/tsp_seeded_swap_local.json` as the quality-first rule
  - keep `configs/local_profiles/tsp_seeded_swap_local_fast.json` as the budget-first rule
  - `fast_x3_equal_split` beat `fast_once` on rescue-target holdout rows, but it still lost to
    `canonical_once` overall and gave back too much on corridor-like anti-cases
  - there is no promoted TSP multistart rule yet; if you are probing bridge/ring-like hard cases,
    multiple fast restarts can be a diagnostic option, not the default
- zdt1:
  - keep `configs/local_profiles/zdt1_diversity_injection.json` as the quality-first rule
  - keep `configs/local_profiles/zdt1_diversity_injection_fast.json` as the budget-first rule
  - merged fast archives (`fast_x2_equal_split`, `fast_x2_budget075`) did not beat either
    `canonical_once` or `fast_once` on holdout hypervolume, so there is no promoted multistart
    archive note
- knapsack:
  - keep `configs/local_profiles/knapsack_repair_local_experimental.json` as a narrow repair rerun
    note only
  - `repair_only` still beat the tiny `none_x2_equal_split` sanity comparator on the tested family
    rows, so broad multistart remains parked
- onemax:
  - keep `none` as the control; restart portfolios add no practical value here

Read the portfolio plots like this:

- tsp:
  - `plot_total_budget_vs_best_of_k.png` shows whether more restarts really improve the best route
    under a matched total budget
  - `plot_restart_count_vs_regret.png` is the quickest way to see whether extra restarts are
    reducing regret relative to `canonical_once`
  - `plot_multistart_vs_single_gap.png` tells you whether best-of-k is actually better than simply
    taking `fast_once`
- zdt1:
  - `plot_total_budget_vs_merged_hv.png` shows whether merged fast archives are buying HV at the
    same total budget
  - `plot_restart_count_vs_hv.png` shows whether more restarts preserve or dilute the archive
  - `plot_merged_archive_vs_single_run.png` is the quickest summary of merged-archive value versus
    `fast_once` and `canonical_once`
- knapsack:
  - `plot_multistart_vs_repair_gap.png` shows whether cheap multistart `none` closes any meaningful
    part of the gap to `repair_only`
  - `plot_budget_vs_feasible_gain.png` shows whether extra starts are buying feasible-quality gain
    or just spending budget to retread the same region

Practical recommendation:

- when quality matters most, run the canonical profile directly
- when you are in a quick local tuning loop, run the fast profile directly
- do not promote a multistart restart note yet; on the current local suite, single-run canonical vs
  single-run fast is still the cleaner rule

### Ranking-Fidelity / Cheap Screening

This pass asks one narrow question only: can the budget-first fast profile act as a cheap ranking
proxy before canonical confirmation?

Run it like this:

```bash
python scripts/run_local_sweep.py --study tsp_ranking_fidelity_study
python scripts/run_local_sweep.py --study tsp_triage_confirm
python scripts/run_local_sweep.py --study zdt1_ranking_fidelity_study
python scripts/run_local_sweep.py --study zdt1_triage_confirm
python scripts/run_local_sweep.py --study knapsack_triage_sanity_study
python scripts/run_local_sweep.py --study onemax_control_ranking_check
```

How to read the outputs:

- `ranking_fidelity.csv`:
  - rank correlation (`Spearman`, `Kendall`)
  - `top_1_match_rate`
  - `top_2_recall`, `top_3_recall`
  - fast-top1 regret against the canonical best candidate
- `triage_workflow_summary.csv`:
  - `always_canonical_all_candidates`
  - `always_fast_pick_top1`
  - `fast_screen_then_confirm_top2`
  - `fast_screen_then_confirm_top3`
  - each row keeps `total_actual_evaluations_used`, final regret, and false-negative rate

Current local read:

- TSP:
  - fast-budget ranking is useful as a rough hint, but not as a promoted screening rule
  - on the 10-seed confirm pass, overall rank fidelity stayed only moderate
    (`Spearman ~= 0.79`, `Kendall ~= 0.69`, `top_1_match ~= 0.33`)
  - `always_fast_pick_top1` saved about 25% of evaluations versus canonical-all, but still missed
    the canonical best on both rescue-target and anti-case rows
  - `fast_screen_then_confirm_top2` removed most of the regret, but because fast runs still cost
    about 75% of canonical, the combined screen+confirm path was more expensive than running all
    candidates canonically once
  - keep `tsp_seeded_swap_local_fast.json` as a budget-first final run, not as a promoted
    screening proxy
- ZDT1:
  - fast-budget HV ranking was weaker (`Spearman ~= 0.26`, `top_1_match = 0`)
  - top-3 canonical confirm recovered the oracle pick, but again cost more than the full
    canonical sweep
  - `zdt1_diversity_injection_fast.json` stays useful as a budget-first final profile, not as a
    cheap screening stage
- knapsack:
  - broad triage is still not worth it
  - `repair_only` remains only a narrow rerun note for subset-sum-like / tight-capacity cases
- onemax:
  - keep it as a control problem; do not build a triage workflow around it

Practical rule:

- quality-first:
  - run the canonical profile directly
- budget-first:
  - run the fast profile directly
- candidate screening:
  - keep it manual / exploratory only until fast ranking becomes cheaper and more rank-preserving
    than it is in the current local suite

### Quality-Tolerance Operating Envelope

This pass asks one narrower follow-up question only: once the Q/F split is already frozen, can you
say more concretely how much quality loss the budget-first fast profile is likely to cost?

Run it like this:

```bash
python scripts/run_local_sweep.py --study tsp_qf_tolerance_study
python scripts/run_local_sweep.py --study tsp_qf_tolerance_confirm
python scripts/run_local_sweep.py --study tsp_fast_tail_hardening_study
python scripts/run_local_sweep.py --study tsp_fast_tail_confirm
python scripts/run_local_sweep.py --study tsp_qf_recalibration_study
python scripts/run_local_sweep.py --study tsp_qf_recalibration_confirm
python scripts/run_local_sweep.py --study tsp_fast_legacy_reference_check
python scripts/run_local_sweep.py --study tsp_qf_recalibration_after_hardening
python scripts/run_local_sweep.py --study tsp_seed_budget_recheck_after_hardening
python scripts/run_local_sweep.py --study zdt1_qf_tolerance_study
python scripts/run_local_sweep.py --study zdt1_qf_tolerance_confirm
python scripts/run_local_sweep.py --study zdt1_qf_tiny_freeze_check
python scripts/run_local_sweep.py --study zdt1_qf_tiny_freeze_recheck
python scripts/run_local_sweep.py --study zdt1_qf_recalibration_after_hardening
python scripts/run_local_sweep.py --study zdt1_seed_budget_recheck_after_hardening
python scripts/run_local_sweep.py --study knapsack_note_freeze_check
python scripts/run_local_sweep.py --study knapsack_note_freeze_recheck
python scripts/run_local_sweep.py --study onemax_control_freeze_check
python scripts/run_local_sweep.py --study onemax_control_freeze_recheck
```

How to read the outputs:

- `tolerance_table.csv`:
  - for TSP: `acceptable_rate` at each route-distance loss bin, split into `overall`,
    `rescue_target`, and `anti_case`
  - for ZDT1: both `hv_only_accept_rate` and safety-gated `acceptable_rate`, plus
    `pareto_ratio_fail_rate`, `spread_fail_rate`, and `joint_safety_fail_rate`
- `plot_q_vs_f_loss_distribution.png`:
  - TSP route-distance loss distribution for fast versus canonical
- `plot_rescue_vs_anticase_loss.png`:
  - the quickest TSP read for whether rescue-target and anti-case tails behave differently
- `plot_q_vs_f_hv_loss_distribution.png`:
  - ZDT1 hypervolume loss distribution for fast versus canonical
- `plot_budget_savings_vs_quality_loss.png` and `plot_budget_savings_vs_hv_loss.png`:
  - whether the saved budget is buying an acceptable quality tradeoff

Current local read:

- TSP:
  - keep the current Q/F split, but do not promote a single explicit numeric tolerance rule yet
  - after fast-default hardening, `F = configs/local_profiles/tsp_seeded_swap_local_fast.json`
    now uses the `pop40/gen33` fixed stack
  - on the 10-seed post-hardening recalibration, that hardened F still saved about `25.5%` actual
    evaluations and about `24.8%` runtime against Q
  - the same recalibration still showed a nontrivial loss tail:
    `mean ~= 0.63%`, `median = 0`, `p90 ~= 2.52%`, `p95 ~= 2.90%`, `max ~= 5.59%`
  - rescue-target rows stayed somewhat cleaner in the middle of the distribution
    (`rescue mean ~= 0.59%`, `rescue p90 ~= 1.39%`) than anti-case rows
    (`anti-case mean ~= 0.67%`, `anti-case p90 ~= 2.60%`), but the global tail still moved enough
    that one numeric rule would over-claim certainty
  - compared against the legacy fast reference on the same holdout suite, the hardened F helped
    rescue-target mean distance a little but still did not close the anti-case side enough to
    relax the protocol
  - this still freezes Option B only: a descriptive split, not a numeric tolerance rule
  - practical read:
    if route quality is sensitive, a corridor-like anti-case is plausible, or you are closing a
    hard-case final, stay on Q; if you want cheaper local iteration or a budget-first final run,
    use the hardened F, but still without a promoted global loss-tolerance rule
- ZDT1:
  - after fast-default hardening, `F = configs/local_profiles/zdt1_diversity_injection_fast.json`
    now uses `adaptation_cooldown=3`
  - on the 10-seed post-hardening recalibration, the hardened F saved about `25%` actual
    evaluations and about `44.2%` runtime against Q
  - the HV envelope tightened further:
    `mean loss ~= 0.094%`, `median ~= 0.081%`, `p90 ~= 0.31%`, `max ~= 0.47%`
  - `pareto_ratio` stayed flat, but spread / joint safety still failed on `20%` of the holdout
    runs, so the final safety read still belongs to Q
  - practical read:
    if roughly `0.25%` HV loss is acceptable and an occasional spread miss is tolerable, F is a
    reasonable budget-first final run; if Pareto-shape protection matters more, stay on Q
- knapsack:
  - keep `configs/local_profiles/knapsack_repair_local_experimental.json` only as a narrow rerun
    note
  - the tiny freeze recheck still favored `repair_only` on the tested subset-sum-like /
    tight-capacity holdout rows, but there is still no broad default and no Q/F envelope
- onemax:
  - keep `none` as the control
  - the tiny freeze recheck again showed no practical need for a separate quality/budget split

Practical recommendation:

- TSP:
  - quality-sensitive run -> Q
  - quick local loop -> F
  - no promoted explicit tolerance note yet
- ZDT1:
  - quality-sensitive run -> Q
  - budget-first run -> F is acceptable when a small HV drop and rare spread miss are okay
- knapsack:
  - keep the narrow repair rerun note only
- onemax:
  - keep the fixed control baseline only

### Seed-Budget Calibration / Confidence-Aware Workflow

This pass asks a narrower operational question: once Q and F are already frozen, how many seeds do
you need before a local decision stops wobbling?

Run it like this:

```bash
python scripts/run_local_sweep.py --study tsp_seed_budget_calibration
python scripts/run_local_sweep.py --study zdt1_seed_budget_calibration
python scripts/run_local_sweep.py --study knapsack_seed_budget_sanity
python scripts/run_local_sweep.py --study onemax_seed_budget_control
python scripts/run_local_sweep.py --study tsp_seed_budget_recheck_after_hardening
python scripts/run_local_sweep.py --study zdt1_seed_budget_recheck_after_hardening
python scripts/run_local_sweep.py --study knapsack_note_freeze_recheck
python scripts/run_local_sweep.py --study onemax_control_freeze_recheck
```

How to read the outputs:

- `seed_budget_table.csv`:
  - for TSP: `mean_loss_pct`, `p90_loss_pct`, tolerance-bin accept rates, and
    `decision_flip_rate_vs_full` across `n = 1 / 3 / 5 / 8 / 10`
  - for ZDT1: HV loss plus spread / Pareto safety fail rates across the same seed ladder
  - for knapsack: whether the narrow `repair_only` note stays directionally stable against `none`
    and `greedy_local_search`
  - for OneMax: whether the control baseline changes at all once a few seeds are added
- `plot_seed_count_vs_loss_ci.png` and `plot_seed_count_vs_hv_ci.png`:
  - the quickest read for how fast the uncertainty band narrows on TSP and ZDT1
- `plot_seed_count_vs_decision_flip.png`:
  - whether the early `n=1` / `n=3` call still flips after the full seed ladder is included
- `plot_rescue_vs_anticase_seed_stability.png`:
  - whether TSP rescue-target and anti-case slices need different seed budgets

Current local read:

- TSP:
  - `n=1` is still too noisy; on the current holdout pool it can make the overall and anti-case
    slices look temporarily F-acceptable
  - `n=3` stays fine for exploratory F, but the overall read is still ambiguous there
  - by `n=5`, the overall read already snaps back toward Q, and the full `n=10` ladder still ends
    on the Q side
  - practical seed budget:
    - exploratory local loop: F on `1-3` seeds
    - comparative Q/F check: at least `3-5` seeds
    - quality-sensitive / anti-case-suspected final call: Q on `8-10` seeds
- ZDT1:
  - `n=1` is too optimistic because HV can look fine before spread safety failures appear
  - `n=3` remains enough for exploratory HV-first use of F
  - `n=5` and `n=8` still stay ambiguous once spread / Pareto safety is included
  - practical seed budget:
    - exploratory budget-first read: F on `3` seeds
    - comparative Q/F check: `5` seeds
    - final confirm with spread / Pareto safety: `8-10` seeds
- knapsack:
  - broad default still stays parked
  - the narrow `repair_only` rerun note stayed stable through `n=1 / 3 / 5`, so the honest local
    rule is still "sanity-check on `3` seeds, use `5` if the family looks borderline"
- onemax:
  - the control comparison stayed flat from `n=1`, so there is no reason to spend a larger seed
    ladder here

### Paired-Seed Sequential Compare / Confidence-Aware Stop Rule

This pass narrows the workflow one step further: instead of choosing a fixed seed count up front,
start with paired `n=3` seeds and only escalate when the comparison stays ambiguous.

Run it like this:

```bash
python scripts/run_local_sweep.py --study tsp_sequential_compare_study
python scripts/run_local_sweep.py --study zdt1_sequential_compare_study
python scripts/run_local_sweep.py --study knapsack_sequential_sanity
python scripts/run_local_sweep.py --study onemax_control_sequential_check
```

How to read the outputs:

- `sequential_decision_table.csv`:
  - the paired-seed stage-by-stage summary for `n = 3 / 5 / 8 / 10`
  - includes `paired_delta_mean`, `paired_delta_median`, bootstrap CI bounds, sign / win counts,
    `decision_label`, `stage_action`, `seeds_used`, and evaluation savings versus the full ladder
- TSP plots:
  - `plot_seed_stage_vs_ci_width.png` for how quickly the Q-versus-F uncertainty shrinks
  - `plot_seed_stage_vs_decision_flip.png` for how often the stage decision still changes after
    more paired seeds are added
  - `plot_rescue_vs_anticase_escalation_rate.png` for whether rescue-target and anti-case slices
    need different escalation depth
  - `plot_cost_savings_vs_false_decision.png` for the tradeoff between stopping early and making
    the wrong Q/F call
- ZDT1 plots:
  - `plot_seed_stage_vs_hv_ci.png` for paired HV uncertainty by stage
  - `plot_seed_stage_vs_safety_fail_rate.png` for spread / Pareto safety failures by stage
  - `plot_seed_stage_vs_decision_flip.png` for whether the early final-safety call still flips
    after more paired seeds are added

Current local read:

- TSP:
  - exploratory loop:
    - `F` on paired `n=3` is already enough for the overall slice and the anti-case slice on the
      current holdout pool
    - rescue-target-only rows are the one place where `n=3` can still be ambiguous, but they
      settled cleanly by `n=5`
  - quality-sensitive comparison:
    - overall and anti-case rows still reject `F` immediately at paired `n=3`
    - rescue-target-only rows briefly look acceptable around `n=8`, but the full `n=10` ladder
      still snaps back to Q-favoring caution
  - practical sequential rule:
    - quick exploratory loop: run `F` on paired `3` seeds and stop unless the run is explicitly
      rescue-target-only and still ambiguous, then extend to `5`
    - comparative / anti-case-aware read: if anti-case suspicion or quality sensitivity matters,
      go straight to `Q` and confirm on paired `8-10` seeds instead of trying to rescue `F`
- ZDT1:
  - exploratory comparison:
    - paired `n=3` is already enough to accept `F` for a cheap HV-first local read
  - safety-sensitive final read:
    - HV can still look fine at `n=3-5`, but spread / Pareto safety misses remain common enough
      that the honest operating rule is still "use `Q` for the final safety-aware call"
  - practical sequential rule:
    - exploratory budget-first loop: `F` on paired `3` seeds
    - final read with spread / Pareto safety: skip the sequential hope and use `Q` on paired
      `8-10` seeds
- knapsack:
  - keep the narrow repair note only
  - `repair_only` sanity is still a paired `3`-seed question, and only tight-capacity /
    borderline-looking rows earned the extra step to paired `5`
- onemax:
  - the sequential comparison is overkill; the control stayed flat immediately, so there is still
    no reason to spend more than the minimal check here

### Frozen Local Operating Protocol / Assisted Compare Runner

The current local findings are now also frozen as a small operating-protocol matrix plus a helper
script. This does not add a new adaptive family or a new search rule. It simply turns the current
Q/F split, seed-budget notes, and narrow knapsack repair-note sanity rule into a direct local
workflow.

Run it like this:

```bash
python scripts/run_local_protocol.py --problem tsp --mode explore
python scripts/run_local_protocol.py --problem tsp --mode compare --case-group rescue_target
python scripts/run_local_protocol.py --problem tsp --mode final --anti-case-suspected
python scripts/run_local_protocol.py --problem zdt1 --mode explore
python scripts/run_local_protocol.py --problem zdt1 --mode compare --final-safety
python scripts/run_local_protocol.py --problem knapsack --mode sanity --borderline
python scripts/run_local_protocol.py --problem onemax --mode control
```

The helper writes `protocol_decision.json` and `protocol_decision.md` to
`outputs/local_protocols/<timestamp>_<problem>_<mode>/`. With `--execute`, it also runs the
current study slice that matches the frozen rule and returns the resulting study paths.

Current protocol matrix:

- TSP:
  - explore: `F` on paired `3` seeds
  - compare: paired `Q` vs `F` at `3` seeds, but rescue-target-only ambiguity earns the `5`-seed
    step
  - final: if anti-case suspicion or quality sensitivity matters, skip the hopeful ladder and go
    straight to `Q` on paired `8-10` seeds
- ZDT1:
  - explore: `F` on paired `3` seeds
  - compare: the same `3`-seed paired read is enough for exploratory HV-first work
  - final: when spread / Pareto safety matters, skip the hopeful ladder and use `Q` on paired
    `8-10` seeds
- knapsack:
  - default baseline still stays `greedy_local_search`
  - `repair_only` remains only a narrow sanity note: `3` seeds for the ordinary check, `5` only
    for tight-capacity or otherwise borderline rows
- onemax:
  - keep `none` as the control and spend just `1` seed

For the shorter matrix and helper-specific examples, see
[Local protocol guide](local_protocol_guide.md).

### Local Stress-Suite Mining / Tail-Risk Reduction

The current defaults and operating protocol are intentionally frozen here. The next local question
is not "what new profile should I add?" but "where do the current defaults still wobble, and which
cases should anchor the next optimization pass?"

Run the small stress suite like this:

```bash
python scripts/run_local_sweep.py --study tsp_stress_suite
python scripts/run_local_sweep.py --study zdt1_stress_suite
python scripts/run_local_sweep.py --study knapsack_stress_suite
python scripts/run_local_sweep.py --study onemax_control_stress_check
```

What to read:

- `stress_case_catalog.csv` / `stress_case_catalog.md`
  - pinned local failure cases, decision-flip rows, and borderline families
- `tail_risk_summary.csv`
  - mean / median / p75 / p90 / p95 / max tail summaries on the current defaults
- `stress_suite_notes.md`
  - short operating notes that point the next optimization pass back to the frozen protocol matrix

Current read from this pass:

- TSP:
  - keep `Q = configs/local_profiles/tsp_seeded_swap_local.json`
  - keep `F = configs/local_profiles/tsp_seeded_swap_local_fast.json`
  - use the stress suite to keep corridor-like anti-cases and rescue-target ambiguity rows fixed as
    future optimization targets
- ZDT1:
  - keep the current `Q/F` split
  - use the stress suite to keep spread / Pareto safety failures visible even when mean HV looks
    acceptable
- knapsack:
  - keep the broad default parked
  - use the stress suite only to remember where the narrow `repair_only` note still overlaps with
    subset-sum-like / tight-capacity families and where weakly correlated rows stay borderline
- onemax:
  - keep `none` as the control
  - the stress check is only there to keep the instrumentation honest

### Stress-Suite-Driven Micro-Hardening For Budget-First Defaults

The next pass stayed inside the existing stress suite and asked a narrower question: can the
budget-first defaults be made less fragile without changing the Q defaults, adding a new adaptive
family, or spending more budget?

Run the micro-hardening pass like this:

```bash
python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_study
python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_confirm
python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_study
python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_confirm
python scripts/run_local_sweep.py --study knapsack_stress_note_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
```

Current read from this pass:

- TSP:
  - keep `Q = configs/local_profiles/tsp_seeded_swap_local.json`
  - keep the same `F` path, but update `configs/local_profiles/tsp_seeded_swap_local_fast.json`
    to the stress-hardened `population_size=40`, `generations=33` split
  - why it changed:
    - same configured budget as the old fast default
    - lower overall mean loss on the pinned stress suite
    - lower anti-case `p90` and lower anti-case max tail
    - rescue-target mean loss did not get worse
  - what did not change:
    - anti-case suspicion and quality-sensitive finals still go straight to `Q`
    - TSP still closes on a descriptive `Q/F` split rather than a promoted global tolerance rule
- ZDT1:
  - keep `Q = configs/local_profiles/zdt1_diversity_injection.json`
  - keep the same `F` path, but update `configs/local_profiles/zdt1_diversity_injection_fast.json`
    to `adaptation_cooldown=3`
  - why it changed:
    - same configured budget as the old fast default
    - lower mean HV loss versus `Q`
    - lower spread / joint safety-fail rate on the pinned stress seeds
    - no Pareto-ratio regression was introduced
  - what did not change:
    - final safety-aware decisions still belong to `Q`
    - the fast default remains for budget-first exploratory or non-final local work
- knapsack:
  - keep the broad default parked
  - keep only the narrow `repair_only` note; the freeze check still does not justify a broad rule
- onemax:
  - keep `none` as the control
  - the freeze check is just a control-path sanity pass

What to read:

- `tail_risk_summary.csv`
  - direct before/after read on the current fast default versus the tested micro-tweaks
- `plot_currentF_vs_candidate_tail.png`
  - TSP anti-case tail comparison on the pinned stress suite
  - `plot_spread_safety_failures.png`
  - ZDT1 spread / joint safety fail read for the current fast default and micro-tweaks

### Protocol-Aware Stress Refresh / Future-Target Registry

Once the current `Q/F` split and protocol matrix are already frozen, the next local-only question is
not "what new profile should I try?" but "which weak points should the next optimization pass be
forced to hit?".

Run the refresh like this:

```bash
python scripts/run_local_sweep.py --study tsp_stress_refresh_suite
python scripts/run_local_sweep.py --study zdt1_stress_refresh_suite
python scripts/run_local_sweep.py --study knapsack_stress_refresh_suite
python scripts/run_local_sweep.py --study onemax_control_refresh_check
python scripts/build_stress_refresh_registry.py --study-name tsp_stress_refresh_suite --study-name zdt1_stress_refresh_suite --study-name knapsack_stress_refresh_suite --study-name onemax_control_refresh_check
```

What this writes under `outputs/local_studies/`:

- `current_stress_case_catalog.csv` / `current_stress_case_catalog.md`
  - the refreshed worst-loss, safety-fail, ambiguity, and borderline rows against the current
    hardened defaults
- `tail_risk_refresh_summary.csv`
  - mean / median / `p75` / `p90` / `p95` / `max` summaries for the current `Q/F/default` pairs
- `future_optimization_targets.csv` / `.json` / `.md`
  - the pinned target registry for the next optimization pass
- `stress_refresh_notes.md`
  - short protocol-facing notes that say where the current defaults still wobble

Current local read from the refresh:

- TSP:
  - keep `Q = configs/local_profiles/tsp_seeded_swap_local.json`
  - keep `F = configs/local_profiles/tsp_seeded_swap_local_fast.json`
  - the weakest slice is still corridor-like anti-case tail, with rescue-target ambiguity second
  - treat the next optimization pass as `tsp_fast_anti_case_tail` first, then
    `tsp_rescue_target_ambiguity`
- ZDT1:
  - keep `Q = configs/local_profiles/zdt1_diversity_injection.json`
  - keep `F = configs/local_profiles/zdt1_diversity_injection_fast.json`
  - the weakest slice is still spread / joint safety failure, not plain mean HV
  - treat the next optimization pass as `zdt1_fast_spread_safety_fail` and
    `zdt1_fast_joint_safety_fail`
- knapsack:
  - keep the broad default parked
  - keep only the narrow `repair_only` note on subset-sum-like / tight-capacity rows, with weakly
    correlated ties as a low-priority borderline target
- onemax:
  - keep `none` as the control and treat it as having no active optimization target

Practical rule:

- do not add a new adaptive family just because one mean metric moved a little
- first check whether the new idea reduces one of the pinned targets in
  `future_optimization_targets.*`
- if it does not move a pinned target, it probably is not the right next local pass

### Stress-Target Reduction Pass

Once the target registry is frozen, the next local-only pass should not reopen broad discovery. It
should ask a much narrower question: do the current top targets actually go down on the pinned
stress suite at the same fast budget?

Run the pass like this:

```bash
python scripts/run_local_sweep.py --study tsp_stress_target_reduction_study
python scripts/run_local_sweep.py --study tsp_stress_target_reduction_confirm
python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_study
python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_confirm
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_stress_target_reduction_registry.py
```

What this writes:

- `tail_risk_reduction_summary.csv`
  - TSP / ZDT1 confirm-side tail summaries focused on whether the pinned stress targets actually
    moved at the same configured fast budget
- `future_optimization_targets.csv` / `.json` / `.md`
  - the refreshed registry with `previous_severity`, `current_severity`, `changed_or_not`, and the
    next honest action
- `stress_reduction_notes.md`
  - short protocol-facing wording that says whether the defaults changed and which targets remain

Current read from the stress-target reduction pass:

- TSP:
  - keep `Q = configs/local_profiles/tsp_seeded_swap_local.json`
  - keep `F = configs/local_profiles/tsp_seeded_swap_local_fast.json`
  - the tested `inversion` micro-tweak improved overall mean loss and anti-case max tail, but it
    did not lower anti-case `p90/p95` enough to replace the current budget-first default
  - protocol implication: keep the anti-case caution wording strong and keep
    `tsp_fast_anti_case_tail` as the top regression target
- ZDT1:
  - keep `Q = configs/local_profiles/zdt1_diversity_injection.json`
  - keep `F = configs/local_profiles/zdt1_diversity_injection_fast.json`
  - nearby `refresh_fraction` / `cooldown` tweaks did not beat the current fast default once HV
    preservation and joint safety were both counted
  - protocol implication: keep `final safety -> Q` explicit and keep
    `zdt1_fast_spread_safety_fail` / `zdt1_fast_joint_safety_fail` at the top of the queue
- knapsack:
  - keep the broad default parked
  - keep only the narrow `repair_only` note
- onemax:
  - keep `none` as the control
  - there is still no active optimization target here

### Failure-Trace Anatomy / Target-Specific Hypothesis Pass

Once the stress-target reduction pass says “keep the current defaults,” the next honest question is
not “what other profile should we try?” but “what mechanism keeps the pinned target alive under the
current default?”

```bash
python scripts/run_local_sweep.py --study tsp_failure_trace_suite
python scripts/run_local_sweep.py --study zdt1_failure_trace_suite
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_failure_hypotheses_registry.py
```

These write:

- `failure_trace_table.csv`
  - per-row trace anatomy for the hardened fast default versus `Q`, including onset/plateau/collapse
    generations plus final loss or safety deltas
- `failure_hypotheses.csv` / `.json` / `.md`
  - the narrowed mechanism hypotheses that the next reduction pass should test directly
- refreshed `future_optimization_targets.csv` / `.json` / `.md`
  - the same target registry, but now with `current_mechanism_hypothesis` and a more concrete
    `recommended_next_action`
- `failure_trace_notes.md`
  - short protocol-facing wording that explains why TSP still needs anti-case caution and why ZDT1
    final safety still belongs to `Q`

Current read from the failure-trace pass:

- TSP:
  - keep `Q = configs/local_profiles/tsp_seeded_swap_local.json`
  - keep `F = configs/local_profiles/tsp_seeded_swap_local_fast.json`
  - the important question is no longer “can we find another cheap profile?” but whether the
    anti-case tail is driven more by seeded head-start lock-in plus diversity collapse, or by late
    refinement deficit after a decent start
  - protocol implication: keep the anti-case/corridor caution wording explicit, because the current
    fast path is still for budget-first iteration, not for the anti-case-sensitive final answer
- ZDT1:
  - keep `Q = configs/local_profiles/zdt1_diversity_injection.json`
  - keep `F = configs/local_profiles/zdt1_diversity_injection_fast.json`
  - the important question is whether the remaining safety failures come from early front plateau,
    late spread collapse, or refresh/cooldown timing mismatch
  - protocol implication: keep `final safety -> Q` explicit even if `F` still looks good on HV
- knapsack:
  - keep the broad default parked
  - keep only the narrow `repair_only` note
- onemax:
  - keep `none` as the control
  - still no active optimization target here

### Target-Specific Hypothesis Probe Pass

Once the failure-trace pass narrows the mechanism list, the next honest question is not "what new
profile should we search?" but "does one tiny same-budget probe actually validate the strongest
hypothesis?"

```bash
python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_study
python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_confirm
python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_study
python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_confirm
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_target_hypothesis_probe_confirm --zdt1-study-name zdt1_target_hypothesis_probe_confirm
```

These write:

- `failure_trace_table.csv`
  - the same trace anatomy table, but now with probe-specific rows for
    `late_refinement_score`, `collapse_to_last_improvement_gap`, safety-fail onset, and the
    current same-budget probe variants
- `failure_hypotheses.csv` / `.json` / `.md`
  - updated rows that mark each hypothesis as `strengthened`, `weakened`, `secondary_only`, or
    `still_unisolated`
- refreshed `future_optimization_targets.csv` / `.json` / `.md`
  - the same target registry, but now with `current_best_hypothesis` and the next recommended
    probe narrowed to the current mechanism story
- `failure_trace_notes.md`
  - protocol-facing wording that explains why TSP still keeps anti-case caution and why ZDT1 final
    safety still belongs to `Q`
- trace plots such as:
  - `plot_late_refinement_gap.png`
  - `plot_collapse_onset_vs_last_improvement.png`
  - `plot_safety_fail_onset.png`

Current read from the target-specific hypothesis probe pass:

- TSP:
  - keep `Q = configs/local_profiles/tsp_seeded_swap_local.json`
  - keep `F = configs/local_profiles/tsp_seeded_swap_local_fast.json`
  - the same-budget late-refinement probe improved some anti-case means, but it made anti-case
    `p90/p95` worse and nudged rescue-target mean loss up, so it did not justify a default rewrite
  - the best current mechanism read is still `late_refinement_deficit`, but only as a weakened
    primary hypothesis; `seed_lockin_and_diversity_collapse` stays secondary
  - protocol implication: keep anti-case/corridor suspicion on `Q`, and describe `F` as the fast
    loop that can start well but still miss late corridor cleanup
- ZDT1:
  - keep `Q = configs/local_profiles/zdt1_diversity_injection.json`
  - keep `F = configs/local_profiles/zdt1_diversity_injection_fast.json`
  - the cooldown-only timing probe reduced safety-fail rate, but it paid too much HV; the
    refresh-only timing probe increased spread/joint failures instead of closing them
  - `refresh_timing_mismatch` is therefore weakened rather than confirmed, and the spread failure
    mechanism is still not isolated enough to promote a new fast default
  - protocol implication: keep `F` for exploratory HV reads, but keep final spread / Pareto safety
    on `Q`
- knapsack:
  - keep the broad default parked
  - keep only the narrow `repair_only` note
- onemax:
  - keep `none` as the control
  - still no active optimization target here

### Population-Generation Tradeoff / Spread-vs-Joint Split Pass

Once the target-specific hypothesis probe pass says "keep the defaults," the next honest question
is not "what broader family should we search?" but "which same-budget knob actually carries
information: population/generation tradeoff or timing?"

```bash
python scripts/run_local_sweep.py --study tsp_population_generation_probe_study
python scripts/run_local_sweep.py --study tsp_population_generation_probe_confirm
python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_study
python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_confirm
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_population_generation_probe_confirm --zdt1-study-name zdt1_timing_vs_pg_probe_confirm
```

These write:

- `failure_trace_table.csv`
  - the same trace anatomy table, now focused on whether the current targets move more under
    population/generation probes or timing probes
- `failure_hypotheses.csv` / `.json` / `.md`
  - refreshed rows that mark whether the current strongest mechanism is strengthened, weakened, or
    still split across targets
- refreshed `future_optimization_targets.csv` / `.json` / `.md`
  - the same target registry, but now with the next recommended probe narrowed to the current
    tradeoff story
- `failure_trace_notes.md`
  - protocol-facing wording that explains why the defaults still stand and what the next
    reduction-oriented pass should test
- focused trace plots such as:
  - `plot_population_generation_tradeoff_vs_tail.png`
  - `plot_spread_vs_joint_fail_split.png`

Current read from the same-budget tradeoff pass:

- TSP:
  - keep `Q = configs/local_profiles/tsp_seeded_swap_local.json`
  - keep `F = configs/local_profiles/tsp_seeded_swap_local_fast.json`
  - the generation-up / population-down probe improved anti-case mean, but it still made anti-case
    `p90/p95` worse on confirm, so it did not justify a fast-default rewrite
  - `late_refinement_deficit` stays the best current explanation, but only in a weakened form;
    `seed_lockin_and_diversity_collapse` stays secondary because the seed-fraction-only probe was
    clearly worse
  - protocol implication: keep anti-case/corridor suspicion on `Q`, and treat rescue-target-only
    ambiguity as the secondary slice only
- ZDT1:
  - keep `Q = configs/local_profiles/zdt1_diversity_injection.json`
  - keep `F = configs/local_profiles/zdt1_diversity_injection_fast.json`
  - spread fail and joint fail should stay separated: the population/generation probe gave the
    strongest spread-side signal, while timing-only probes did not close the joint-fail story
  - `probe_pg_pop41_gen88` is now the most informative same-budget reference candidate, but the
    evidence is still too narrow to rewrite the fast default from this pass alone
  - protocol implication: keep `F` for exploratory HV reads, but keep final safety on `Q 8-10`
- knapsack:
  - keep the broad default parked
  - keep only the narrow `repair_only` note
- onemax:
  - keep `none` as the control
  - still no active optimization target here

### Extreme-Tail Closeout / Split-Target Confirmation

The next question stayed deliberately narrow: not "what broader family should we search?" but
"can the current fast defaults be closed with one more same-budget contour pass that prioritizes
the tail slices directly?"

```bash
python scripts/run_local_sweep.py --study tsp_extreme_tail_pg_contour_study
python scripts/run_local_sweep.py --study tsp_extreme_tail_pg_contour_confirm
python scripts/run_local_sweep.py --study zdt1_spread_pg_probe_study
python scripts/run_local_sweep.py --study zdt1_spread_pg_probe_confirm
python scripts/run_local_sweep.py --study zdt1_joint_timing_probe_study
python scripts/run_local_sweep.py --study zdt1_joint_timing_probe_confirm
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_extreme_tail_pg_contour_confirm --zdt1-spread-study-name zdt1_spread_pg_probe_confirm --zdt1-joint-study-name zdt1_joint_timing_probe_confirm
```

These write:

- `failure_trace_table.csv`
  - the same mechanism table, but now with extreme-tail TSP contour rows and fully split ZDT1
    spread-vs-joint rows
- `failure_hypotheses.csv` / `.json` / `.md`
  - refreshed rows that label the current mechanisms as `weakened`, `secondary_only`,
    `strengthened`, or still `unisolated` under the current fast defaults
- refreshed `future_optimization_targets.csv` / `.json` / `.md`
  - the same target registry, but now with TSP anti-case tail and ZDT1 spread/joint targets
    refreshed separately after the closeout pass
- `failure_trace_notes.md`
  - protocol-facing wording that explains why the defaults still stand and what the next
    reduction-oriented pass should test
- focused plots such as:
  - `plot_anticase_p95_max_reduction.png`
  - `plot_population_generation_vs_spread_tail.png`
  - `plot_timing_vs_joint_fail_rate.png`

Current read from the extreme-tail closeout pass:

- TSP:
  - keep `Q = configs/local_profiles/tsp_seeded_swap_local.json`
  - keep `F = configs/local_profiles/tsp_seeded_swap_local_fast.json`
  - the closest same-budget contour point, `contour_pg_gen32_pop41`, nudged anti-case `p95`
    slightly down (`2.8977 -> 2.8741`) but still made anti-case `max` worse (`3.0665 -> 3.5291`)
    and slightly hurt rescue-target mean (`0.5863 -> 0.6744`), so it still did not justify a
    same-name default rewrite
  - `late_refinement_deficit` stays the strongest current TSP explanation, but only as
    `weakened`; `seed_lockin_and_diversity_collapse` is now narrow enough to keep as
    `secondary_only`
  - protocol implication: anti-case/corridor suspicion still belongs on `Q`, while rescue-target
    ambiguity stays secondary
- ZDT1 spread target:
  - keep `Q = configs/local_profiles/zdt1_diversity_injection.json`
  - keep `F = configs/local_profiles/zdt1_diversity_injection_fast.json`
  - `spread_pg_pop41_gen88` is still the strongest same-budget reference point: on the fixed
    spread-fail slice it cut spread fail rate from `0.20` to `0.10`, lowered `spread_delta_p90`
    from `0.0611` to `0.0387`, lowered `spread_delta_p95` from `0.0916` to `0.0552`, and also
    improved HV tail instead of reopening it
  - that is enough to strengthen the "spread behaves more like a population/generation problem"
    story, but not enough to replace the fast default from this pass alone
- ZDT1 joint target:
  - keep `Q = configs/local_profiles/zdt1_diversity_injection.json`
  - keep `F = configs/local_profiles/zdt1_diversity_injection_fast.json`
  - `joint_timing_cooldown2` still lowered joint fail rate from `0.20` to `0.10`, but it paid too
    much HV tail (`p90` moved to `0.5097%`, `p95` to `0.5827%`), while `joint_timing_refresh012`
    reopened failures instead of closing them
  - timing mismatch therefore stays plausible only in a weakened way, and joint safety should stay
    separated from the spread target
- knapsack:
  - keep the broad default parked
  - keep only the narrow `repair_only` note
- onemax:
  - keep `none` as the control
  - still no active optimization target here

### TSP Tail Freeze / ZDT1 Spread Candidate Validation

The next local-only question is not a new profile search. It is a closeout pass:

1. freeze the remaining TSP anti-case `p95/max` as an operating limitation if the current
   same-budget contour still cannot close it, and
2. validate whether `spread_pg_pop41_gen88` is a real ZDT1 fast-default replacement candidate or
   only a note-level reference.

Run it like this:

```bash
python scripts/run_local_sweep.py --study tsp_tail_freeze_recheck
python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_study
python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_confirm
python scripts/run_local_sweep.py --study zdt1_joint_note_freeze_check
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_tail_freeze_recheck --zdt1-spread-study-name zdt1_spread_candidate_boundary_confirm --zdt1-joint-study-name zdt1_joint_note_freeze_check
```

Read those outputs like this:

- TSP:
  - keep `Q = configs/local_profiles/tsp_seeded_swap_local.json`
  - keep `F = configs/local_profiles/tsp_seeded_swap_local_fast.json`
  - the remaining anti-case `p95/max` tail is now best treated as a protocol limitation under the
    current fixed stack, not as an active same-budget contour target
  - protocol implication: anti-case / corridor suspicion and quality-sensitive finals still go
    straight to `Q 8-10`
- ZDT1 spread target:
  - keep `Q = configs/local_profiles/zdt1_diversity_injection.json`
  - keep `F = configs/local_profiles/zdt1_diversity_injection_fast.json` unless the spread
    candidate wins not just on stress rows, but also on normal / stable rows without hurting HV,
    Pareto ratio, or joint safety
  - `spread_pg_pop41_gen88` is the current information-rich candidate, but it closes as
    `note_only_stress_slice` if its stress gain does not survive the normal / holdout slice
- ZDT1 joint target:
  - timing-only checks remain note-level only
  - final safety still belongs to `Q`
- knapsack:
  - keep the broad default parked
  - keep only the narrow `repair_only` note
- onemax:
  - keep `none` as the control
  - still no active optimization target here

## Local UX Notes

- `run_local_experiment.py` is for quick checks.
- `run_local_sweep.py` is for 1-3 axis local tuning.
- The study manifests reuse existing presets and config normalization.
- The plotting layer stays lightweight: CSV, Markdown, and matplotlib PNG only.

## Related Files

- [Examples](../examples/README.md)
- [Quickstart](quickstart.md)
- [Install](install.md)

## Local Baseline Snapshot

The current local operating baseline is now frozen in:

- `artifacts/local_baseline_snapshot.json`
- `artifacts/local_baseline_snapshot.md`
- `artifacts/local_baseline_check.json`
- `artifacts/local_baseline_check.md`

Baseline read:

- TSP:
  - `F = configs/local_profiles/tsp_seeded_swap_local_fast.json` remains budget-first / exploratory
  - anti-case / corridor suspicion or quality-sensitive final still goes straight to `Q 8-10`
  - `tsp_fast_anti_case_tail is frozen as a protocol limitation`
- ZDT1:
  - `F = configs/local_profiles/zdt1_diversity_injection_fast.json` remains exploratory / budget-first
  - final safety still belongs to `Q = configs/local_profiles/zdt1_diversity_injection.json`
  - `spread_pg_pop41_gen88` remains `note_only_stress_slice`, not a default replacement
  - `zdt1_fast_joint_safety_fail` remains `monitor_only` and closes via `Q` on final safety work
- knapsack:
  - no broad default
  - keep the `repair_only` note narrow on subset-sum / tight-capacity-like rows only
- onemax:
  - `none` control only

Baseline guard commands:

```bash
python scripts/run_local_sweep.py --study local_baseline_guard_tsp
python scripts/run_local_sweep.py --study local_baseline_guard_zdt1
python scripts/run_local_sweep.py --study local_baseline_guard_knapsack
python scripts/run_local_sweep.py --study local_baseline_guard_onemax
python scripts/check_local_baseline.py --write-snapshot
python scripts/check_local_baseline.py
```

This freeze does not promote any new profile. It only locks the current defaults, protocol notes,
and target decisions so future local experiments can be judged as clear improvements or drift
against one machine-readable baseline.

## Regression-Gated Candidate Workflow

Once the baseline is frozen, new local ideas should enter as candidate manifests
instead of immediate profile edits.

Files:

- `configs/local_candidates/candidate_schema.json`
- `configs/local_candidates/example_tsp_candidate.json`
- `configs/local_candidates/example_zdt1_candidate.json`
- `configs/local_candidates/example_knapsack_candidate.json`
- `docs/local_candidate_workflow.md`

Commands:

```bash
python scripts/check_local_baseline.py
python scripts/run_local_candidate.py --candidate configs/local_candidates/example_zdt1_candidate.json
python scripts/run_local_candidate.py --candidate configs/local_candidates/example_tsp_candidate.json --no-execute
python scripts/run_local_candidate.py --candidate configs/local_candidates/example_knapsack_candidate.json --use-existing-output
python scripts/summarize_local_candidates.py
python scripts/summarize_local_optimization_status.py
python scripts/build_local_baseline_change_request.py --candidate-report outputs/local_candidates/.../candidate_report.json
python scripts/build_local_baseline_change_request.py --candidate-report outputs/local_candidates/.../candidate_report.json --force-draft
```

Each candidate report writes:

- `candidate_report.json`
- `candidate_report.md`
- `candidate_summary.csv`
- `candidate_vs_baseline.csv`

Decision labels:

- `reject_regression`
- `reject_no_material_gain`
- `note_only_stress_slice`
- `monitor_only`
- `candidate_promising_needs_confirm`
- `candidate_passes_local_guard`
- `candidate_requires_new_mechanism_hypothesis`
- `baseline_drift_detected`
- `intentional_baseline_change_required`

Lifecycle read:

- `reject_*` -> `rejected`
- `note_only_stress_slice` -> `note_only`
- `monitor_only` -> `monitor`
- `candidate_promising_needs_confirm` -> `promising_needs_confirm`
- `candidate_passes_local_guard` -> `passed_local_guard`
- `candidate_requires_new_mechanism_hypothesis` -> `requires_new_mechanism`
- `baseline_drift_detected` -> `blocked_by_baseline_drift`
- `intentional_baseline_change_required` -> `ready_for_change_request`

Ledger + change-control:

- `python scripts/summarize_local_candidates.py`
  - rebuilds `artifacts/local_candidate_ledger.*`
  - rebuilds `artifacts/local_candidate_summary.*`
- `python scripts/build_local_baseline_change_request.py --candidate-report ...`
  - drafts a manual change-request pack only
  - never rewrites `configs/local_profiles/*`
  - never refreshes the baseline snapshot automatically

Problem-specific read:

- TSP:
  - `F` stays budget-first / exploratory only
  - anti-case / corridor suspicion or quality-sensitive final still goes
    straight to `Q 8-10`
  - because `tsp_fast_anti_case_tail` is frozen as a protocol limitation, a new
    TSP fast candidate must bring a new mechanism hypothesis before it can even
    be considered for replacement
- ZDT1:
  - `F` stays useful for exploratory / budget-first work
  - final safety still belongs to `Q`
  - `spread_pg_pop41_gen88` remains a `note_only_stress_slice` reference until a
    future candidate survives stable/normal non-regression as well
  - timing-only joint candidates stay `monitor_only` unless they stop paying an
    HV tail penalty
- knapsack:
  - broad default promotion is still forbidden
  - keep `repair_only` narrow and family-conditioned
- onemax:
  - still control only

Passing the local candidate guard does not auto-update the baseline snapshot or
replace any profile. It only means the candidate has earned a manual review.

## Local Optimization Cycle Closeout

The first local-only optimization cycle is now closed as a status/reporting
state rather than an open tuning queue.

Artifacts:

- `artifacts/local_optimization_status.json`
- `artifacts/local_optimization_status.md`
- `artifacts/local_reopen_criteria.json`
- `artifacts/local_candidate_backlog_closeout.json`
- `artifacts/local_candidate_backlog_closeout.md`
- `docs/local_reopen_criteria.md`

Commands:

```bash
python scripts/check_local_baseline.py
python scripts/summarize_local_candidates.py
python scripts/summarize_local_optimization_status.py
python scripts/build_local_baseline_change_request.py --candidate-report outputs/local_candidates/.../candidate_report.json --force-draft
```

Closeout read:

- TSP:
  - `F` stays budget-first / exploratory
  - anti-case / corridor suspicion or quality-sensitive final still goes straight to `Q 8-10`
  - `tsp_fast_anti_case_tail` is now a frozen protocol limitation
  - reopen only with a genuinely new mechanism hypothesis
- ZDT1:
  - `F` stays exploratory / budget-first
  - final safety still belongs to `Q`
  - `spread_pg_pop41_gen88` stays `note_only_stress_slice`
  - `zdt1_fast_joint_safety_fail` stays `monitor_only`
  - reopen only if a candidate generalizes beyond the spread-stress slice without stable/normal regression
- knapsack:
  - no broad default
  - keep `repair_only` as a narrow family-conditioned note only
- onemax:
  - no active target
  - reopen only if control drift appears

When not to run more experiments:

- do not rerun the old TSP PG contour family
- do not claim stress-slice-only ZDT1 spread gains as a default replacement
- do not reopen broad knapsack default discovery
- do not treat onemax as an active optimization branch without drift evidence

