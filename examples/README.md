# Examples

These examples are meant to give an external user a first successful run in one command.
After install, the commands below are supported from outside the repo root.

## Commands

```bash
ga-lab-run --preset onemax_small --output-root ./ga-lab-outputs
ga-lab-demo baseline --output-root outputs/demo
ga-lab-demo pure-ga --output-root outputs/demo
ga-lab-demo hybrid --output-root outputs/demo
ga-lab-demo nsga2 --output-root outputs/demo
```

## Local Experiment Commands

```bash
python scripts/run_local_experiment.py --preset onemax_small
python scripts/run_local_experiment.py --demo nsga2
python scripts/run_local_sweep.py --study onemax_mutation_study
python scripts/run_local_sweep.py --study tsp_quality_study
python scripts/run_local_sweep.py --study zdt1_nsga2_mutation_study
python scripts/run_local_sweep.py --study onemax_adaptive_mutation_study
python scripts/run_local_sweep.py --study tsp_stagnation_study
python scripts/run_local_sweep.py --study zdt1_diversity_study
python scripts/run_local_sweep.py --study tsp_hardcase_trigger_suite
python scripts/run_local_sweep.py --study tsp_hardcase_holdout_suite
python scripts/run_local_sweep.py --study tsp_policy_router_train_study
python scripts/run_local_sweep.py --study tsp_policy_router_holdout_study
python scripts/run_local_sweep.py --study tsp_policy_router_budget_band_study
python scripts/run_tsp_router_analysis.py --train-study-dir outputs/local_studies/<train_dir> --holdout-study-dir outputs/local_studies/<holdout_dir> --budget-study-dir outputs/local_studies/<budget_dir>
python scripts/run_local_sweep.py --study tsp_regime_switching_coarse
python scripts/run_local_sweep.py --study tsp_regime_switching_holdout
python scripts/run_local_sweep.py --study tsp_regime_switching_budget_band
python scripts/run_local_sweep.py --study tsp_underbudget_rescue_suite
python scripts/run_local_sweep.py --study tsp_underbudget_rescue_holdout
python scripts/run_local_sweep.py --study tsp_underbudget_anticase_check
python scripts/run_local_sweep.py --study zdt1_budget_note_freeze
python scripts/run_local_sweep.py --study tsp_seed_fraction_ablation_study
python scripts/run_local_sweep.py --study tsp_seed_source_ablation_study
python scripts/run_local_sweep.py --study tsp_mutation_operator_ablation_study
python scripts/run_local_sweep.py --study tsp_canonical_default_confirm
python scripts/run_local_sweep.py --study zdt1_default_freeze_recheck
python scripts/run_local_sweep.py --study zdt1_threshold_anatomy_study
python scripts/run_local_sweep.py --study zdt1_refresh_anatomy_study
python scripts/run_local_sweep.py --study zdt1_cooldown_anatomy_study
python scripts/run_local_sweep.py --study zdt1_canonical_default_confirm
python scripts/run_local_sweep.py --study knapsack_family_suite
python scripts/run_local_sweep.py --study knapsack_seeded_repair_anatomy_study
python scripts/run_local_sweep.py --study knapsack_feasibility_control_study
python scripts/run_local_sweep.py --study knapsack_canonical_experimental_confirm
python scripts/run_local_sweep.py --study knapsack_rerun_boundary_suite
python scripts/run_local_sweep.py --study knapsack_rerun_boundary_holdout
python scripts/run_local_sweep.py --study knapsack_repair_vs_restart_confirm
python scripts/run_local_sweep.py --study tsp_budget_frontier_study
python scripts/run_local_sweep.py --study tsp_fast_profile_confirm
python scripts/run_local_sweep.py --study zdt1_budget_frontier_study
python scripts/run_local_sweep.py --study zdt1_fast_profile_confirm
python scripts/run_local_sweep.py --study knapsack_rerun_gate_efficiency_study
python scripts/run_local_sweep.py --study knapsack_rerun_gate_confirm
python scripts/run_local_sweep.py --study onemax_control_budget_check
python scripts/run_local_sweep.py --study tsp_two_stage_gate_study
python scripts/run_local_sweep.py --study zdt1_two_stage_gate_study
python scripts/run_local_sweep.py --study knapsack_rerun_gate_sanity_study
python scripts/run_local_sweep.py --study tsp_restart_portfolio_study
python scripts/run_local_sweep.py --study zdt1_restart_portfolio_study
python scripts/run_local_sweep.py --study knapsack_multistart_sanity_study
python scripts/run_local_sweep.py --study onemax_control_portfolio_check
python scripts/run_local_sweep.py --study tsp_ranking_fidelity_study
python scripts/run_local_sweep.py --study tsp_triage_confirm
python scripts/run_local_sweep.py --study zdt1_ranking_fidelity_study
python scripts/run_local_sweep.py --study zdt1_triage_confirm
python scripts/run_local_sweep.py --study knapsack_triage_sanity_study
python scripts/run_local_sweep.py --study onemax_control_ranking_check
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
python scripts/run_local_sweep.py --study tsp_seed_budget_calibration
python scripts/run_local_sweep.py --study zdt1_seed_budget_calibration
python scripts/run_local_sweep.py --study knapsack_seed_budget_sanity
python scripts/run_local_sweep.py --study onemax_seed_budget_control
python scripts/run_local_sweep.py --study tsp_sequential_compare_study
python scripts/run_local_sweep.py --study zdt1_sequential_compare_study
python scripts/run_local_sweep.py --study knapsack_sequential_sanity
python scripts/run_local_sweep.py --study onemax_control_sequential_check
python scripts/run_local_sweep.py --study tsp_stress_suite
python scripts/run_local_sweep.py --study zdt1_stress_suite
python scripts/run_local_sweep.py --study knapsack_stress_suite
python scripts/run_local_sweep.py --study onemax_control_stress_check
python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_study
python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_confirm
python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_study
python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_confirm
python scripts/run_local_sweep.py --study knapsack_stress_note_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/run_local_sweep.py --study tsp_default_tiny_freeze_check
python scripts/run_local_sweep.py --study onemax_control_tiny_check
python scripts/run_local_sweep.py --study onemax_control_hardcase_check
python scripts/run_local_sweep.py --study onemax_control_router_check
python scripts/run_local_sweep.py --study onemax_control_switching_check
python scripts/run_local_experiment.py --config configs/local_profiles/tsp_diversity_injection.json
python scripts/run_local_experiment.py --config configs/local_profiles/tsp_seeded_swap_local.json
python scripts/run_local_experiment.py --config configs/local_profiles/tsp_seeded_swap_local_fast.json
python scripts/run_local_experiment.py --config configs/local_profiles/zdt1_diversity_injection.json
python scripts/run_local_experiment.py --config configs/local_profiles/zdt1_diversity_injection_fast.json
python scripts/run_local_experiment.py --config configs/local_profiles/knapsack_restart_experimental.json
python scripts/run_local_experiment.py --config configs/local_profiles/knapsack_repair_local_experimental.json
```

These commands are meant for local iteration inside the repo checkout. They write CSV, Markdown,
and PNG outputs directly under `outputs/local_runs/` and `outputs/local_studies/`.

Adaptive local studies also produce:

- `history_summary.csv`
- `plot_diversity.png`
- `plot_stagnation.png`
- `plot_trigger_events.png` for TSP, ZDT1, and knapsack trigger timing
- `plot_post_trigger_gain.png` for TSP mechanism-isolation runs
- `plot_refresh_schedule_vs_gain.png` for TSP delayed-trigger versus periodic-refresh runs
- `plot_refresh_volume_vs_gain.png` for TSP refresh-volume checks
- `plot_collapse_onset_vs_trigger.png` for TSP collapse timing versus first-trigger timing
- `plot_budget_band_vs_gap.png` for TSP reduced/current budget comparisons
- `plot_instance_feature_map.png` for TSP geometry feature maps
- `plot_feature_vs_policy_gap.png` for feature-to-policy-margin inspection
- `plot_bridge_score_vs_trigger_value.png` for trigger-value versus bridge-score checks
- `plot_anisotropy_vs_decay_value.png` for decay-value versus anisotropy checks
- `plot_router_regret.png` for router regret against fixed policies
- `plot_budget_band_vs_policy_win.png` for policy wins by budget band
- `router_decision_table.md` for the tested threshold rule
- `plot_mode_switch_timeline.png` for online regime-switching timelines
- `plot_diversity_vs_mode.png` for diversity traces by active mode
- `plot_collapse_onset_vs_switch.png` for collapse timing versus first mode switch
- `plot_regret_vs_policy.png` for regret against the best fixed policy
- `plot_budget_band_vs_regret.png` for switching regret by budget band
- `plot_rescue_target_vs_anticase_gap.png` for rescue-target gain versus anti-case damage
- `plot_seed_fraction_vs_gap.png` for TSP seeded fixed-stack gap checks
- `plot_seed_source_vs_gap.png` for TSP seeding-source checks
- `plot_mutation_operator_vs_gap.png` for TSP mutation-operator simplification checks
- `plot_initial_quality_vs_final_gap.png` for TSP seeded head-start versus final gap
- `plot_family_vs_regret.png` for knapsack family-conditioned regret checks
- `plot_seeded_vs_repair_gap.png` for knapsack seeding-versus-repair anatomy
- `plot_repair_vs_greedy_gap.png` for repair-only versus greedy gaps on knapsack boundary runs
- `plot_init_feasible_vs_final_gain.png` for knapsack feasible-start versus final-gain checks
- `plot_initial_feasible_fraction_vs_gain.png` for knapsack feasible-start versus gain checks
- `plot_capacity_tightness_vs_gain.png` for knapsack tightness versus gain-over-plain-GA checks
- `plot_budget_vs_feasible_gain.png` for knapsack budget-to-feasible-gain checks
- `plot_budget_vs_runtime.png` for TSP budget-versus-runtime checks
- `plot_budget_vs_regret.png` for TSP quality-loss versus budget checks
- `plot_early_stop_vs_quality.png` for TSP plateau-stop quality tradeoffs
- `plot_fast_vs_canonical_rank.png` for TSP fast-versus-canonical rank fidelity
- `plot_topk_recall_vs_budget.png` for fast-screening top-k recall versus budget
- `plot_triage_cost_vs_regret.png` for TSP triage workflow cost-versus-regret
- `plot_rescue_target_vs_anticase_rank_fidelity.png` for rescue-target versus anti-case rank fidelity
- `plot_q_vs_f_loss_distribution.png` for TSP Q-versus-F route-distance loss envelopes
- `plot_rescue_vs_anticase_loss.png` for TSP rescue-target versus anti-case Q/F loss tails
- `plot_q_vs_f_loss_distribution_recalibrated.png` for the post-hardening TSP Q/F recalibration pass
- `plot_rescue_vs_anticase_loss_recalibrated.png` for recalibrated rescue-target versus anti-case TSP tails
- `plot_budget_savings_vs_quality_loss_recalibrated.png` for recalibrated TSP budget-savings versus quality-loss checks
- `plot_tolerance_accept_rate_recalibrated.png` for recalibrated TSP tolerance-bin acceptance rates
- `plot_budget_vs_spread.png` for ZDT spread changes across budget cuts
- `plot_early_stop_vs_hv.png` for ZDT early-stop HV checks
- `plot_fast_vs_canonical_hv_rank.png` for ZDT1 fast-versus-canonical HV rank fidelity
- `plot_triage_cost_vs_hv_regret.png` for ZDT1 triage workflow cost-versus-HV regret
- `plot_spread_safety_failures.png` for ZDT1 spread / pareto safety failures during triage
- `plot_q_vs_f_hv_loss_distribution.png` for ZDT1 Q-versus-F hypervolume loss envelopes
- `plot_q_vs_f_hv_loss_distribution_recheck.png` for the ZDT1 tiny Q/F freeze recheck
- `plot_budget_savings_vs_hv_loss.png` for ZDT1 budget savings versus HV loss
- `tolerance_table.csv` for Q/F operating-envelope summaries
- `tsp_fast_tail_summary.csv` for TSP fast-profile anti-case tail summaries
- `plot_q_vs_f_tail_distribution.png` for TSP fast-hardening loss distributions against Q
- `plot_candidate_vs_anti_case_p90.png` for TSP anti-case p90 checks across fixed-stack fast variants
- `plot_candidate_vs_rescue_mean.png` for TSP rescue-target mean-loss preservation across fast variants
- `plot_seed_fraction_vs_tail.png` for TSP seed-fraction versus anti-case tail checks
- `plot_operator_vs_tail.png` for TSP mutation-operator versus anti-case tail checks
- `plot_old_fast_vs_new_fast_tail.png` for the legacy-fast versus hardened-fast TSP anti-case tail comparison
- `plot_rerun_gate_vs_regret.png` for knapsack pilot+rereun efficiency checks
- `plot_initial_feasible_fraction_vs_rerun_value.png` for knapsack rerun-value versus pilot feasibility
- `plot_mutation_rate.png` for knapsack mutation schedules
- `plot_parameter_sweep.png` for numeric single-axis sweeps
- problem-specific plots like `plot_feasibility.png`, `plot_route_distance.png`, and
  `plot_hypervolume.png`
- focused diagnostics like `plot_violation.png`, `plot_diversity_vs_distance.png`, and
  `plot_hv_vs_spread.png`
- `plot_budget_vs_hv.png` for ZDT threshold-versus-budget checks

## Stable Python API

```python
from ga_lab.api import recommend_solver, run_demo, run_preset

recommendation = recommend_solver("onemax", 128, "default")
run_preset("onemax_small", output_root="ga-lab-outputs")
run_demo("nsga2", output_root="ga-lab-outputs")
```

Use `ga_lab.api` for library code. The `scripts/` wrappers remain repo-oriented compatibility
helpers, not the stable import contract.

## Example Folders

- [minimal_baseline](minimal_baseline/README.md)
- [minimal_ga](minimal_ga/README.md)
- [minimal_hybrid](minimal_hybrid/README.md)
- [minimal_nsga2](minimal_nsga2/README.md)

## Notes

- `baseline` is the practical onemax default path.
- `pure-ga` is a smoke run for the GA execution path, not the practical onemax default.
- `hybrid` stays within the current narrow official TSP medium quality-first path.
- `nsga2` should be read through hypervolume and Pareto metrics.
- family-conditioned benchmark guidance still lives in docs rather than the stable Python API.
- local sweep studies are for tuning and inspection, not for claim-making or release governance.
- in the current local-only adaptive pass:
  - TSP now keeps `configs/local_profiles/tsp_seeded_swap_local.json` as the canonical local
    default for the current hard-case suite
  - `configs/local_profiles/tsp_seeded_swap_local_fast.json` is now the budget-first companion:
    same seeded swap stack, but now hardened to `population_size=40`, `generations=33`, about 25%
    lower configured budget, about `0.63%` mean route-distance loss on the recalibration confirm,
    and a clearly better mixed holdout tradeoff than the older legacy fast reference
  - the seeding-anatomy pass showed that nearest-neighbor mix seeding is the main gain source:
    `hybrid_ga` without that seed source falls back toward `none`
  - `seed_fraction=0.25` stays competitive on rescue-target cases, but `seed_fraction=0.5` remains
    the safer mixed holdout choice, so the current default stays frozen
  - `seeded_inversion_seed25` stayed competitive, but not enough to replace the current
    `swap`-based stack as the simplest reusable profile
  - use `plot_seed_source_vs_gap.png`, `plot_seed_fraction_vs_gap.png`, and
    `plot_initial_quality_vs_final_gap.png` together to separate seed-source head start from
    operator-only effects
  - `decay_mutation` stays the strong fixed comparator, and `tsp_diversity_injection.json` stays a
    legacy mechanism-comparison profile rather than the preferred default
  - ZDT1 now keeps the simpler `0.55 / 0.10 / cooldown=4` profile as the reusable quality-first
    default
  - `configs/local_profiles/zdt1_diversity_injection_fast.json` is now the budget-first companion:
    same diversity rule, `population_size=45`, `cooldown=3`, about 25% lower configured budget,
    and about `0.09%` mean HV loss on the recalibration confirm
  - the anatomy pass still keeps `threshold=0.55` at the center and `refresh_fraction=0.10` as the
    clean simplified setting, while the hardened fast companion now uses the shorter cooldown
  - `0.60` remains only a nearby exploratory threshold note, not a promoted replacement
  - knapsack broad adaptive tuning is still parked, but there is now one narrow experimental local
    profile worth keeping:
    `configs/local_profiles/knapsack_repair_local_experimental.json`
  - the seeded/repair anatomy pass showed that repair is the real mechanism:
    `repair_only` matched `seeded_repair`, while `seeded_only` usually trailed
  - the rerun-boundary confirm kept that note narrow instead of broad:
    `repair_only` stayed useful on subset-sum-like / tight-capacity holdout rows, but
    `weakly_correlated_small_b` only tied `greedy_local_search`
  - use `plot_seeded_vs_repair_gap.png`, `plot_repair_vs_greedy_gap.png`, and
    `plot_initial_feasible_fraction_vs_gain.png` together:
    if repair-only stays above `none` and restart on subset-sum-like / tight-capacity rows while
    only tying greedy on weakly correlated rows, keep it as a narrow rerun note instead of a broad
    default
  - the current narrow rerun note is:
    if the small family looks subset-sum-like or tight-capacity-like, especially when a plain GA
    pilot starts with very low `initial_feasible_fraction` (roughly `<= 0.05` in the tested local
    suite), one `repair_only` rerun is worth trying
  - the new pilot+rereun gate stays experimental:
    it reran correctly on boundary-like cases, but when it skipped rerun on anti-boundary cases the
    pilot-only result gave back too much quality to replace the narrower manual rerun note
  - the seed-budget calibration pass then turned the current defaults into a confidence-aware local
    workflow:
    - TSP: use F for quick `1-3` seed loops, but use Q on `8-10` seeds when anti-case risk or
      quality sensitivity matters
    - ZDT1: use F on `3` seeds for exploratory budget-first checks, then use `8-10` seeds if you
      want the final HV plus spread / Pareto safety read
    - knapsack: `repair_only` sanity is still a `3`-seed question, with `5` seeds only for a
      borderline family recheck
    - onemax: the control stayed flat from `n=1`, so there is still no reason to spend a larger
      seed ladder
  - the sequential compare pass then asked whether the current fixed seed notes can be turned into
    a cheaper paired-seed stop rule
  - current answer:
    - TSP exploratory loops can start with paired `3` seeds on `F`, and only rescue-target-only
      ambiguous rows really need the paired `5` step
    - TSP quality-sensitive / anti-case-aware decisions still favor going straight to `Q` on paired
      `8-10` seeds
    - ZDT1 keeps the same split in a cleaner form: paired `3` seeds is enough for exploratory `F`,
      but spread / Pareto safety still keeps the final call on `Q` with paired `8-10` seeds
    - knapsack keeps the narrow repair note: paired `3` seeds for the sanity check, paired `5`
      only for borderline tight-capacity rows
    - onemax still does not need a sequential ladder
  - the frozen local protocol helper now turns those rules into a direct runner:
    - TSP: `explore -> F 3`, `compare -> paired Q/F at 3 (rescue-target-only 5)`, `final -> Q 8-10`
    - ZDT1: `explore -> F 3`, `final safety -> Q 8-10`
    - knapsack: `sanity -> repair_only note on 3, borderline 5`
    - onemax: `control -> none on 1`
  - helper examples:
    ```bash
    python scripts/run_local_protocol.py --problem tsp --mode explore
    python scripts/run_local_protocol.py --problem tsp --mode compare --case-group rescue_target
    python scripts/run_local_protocol.py --problem tsp --mode final --anti-case-suspected
    python scripts/run_local_protocol.py --problem zdt1 --mode compare --final-safety
    python scripts/run_local_protocol.py --problem knapsack --mode sanity --borderline
    ```
  - `--execute` reuses the current study manifests and writes `protocol_decision.json` /
    `protocol_decision.md` plus optional study outputs under `outputs/local_protocols/...`
  - once the protocol is frozen, use the stress suite to pin future optimization targets rather
    than to invent a broad new default:
    ```bash
    python scripts/run_local_sweep.py --study tsp_stress_suite
    python scripts/run_local_sweep.py --study zdt1_stress_suite
    python scripts/run_local_sweep.py --study knapsack_stress_suite
    python scripts/run_local_sweep.py --study onemax_control_stress_check
    ```
  - these writes `stress_case_catalog.csv`, `tail_risk_summary.csv`, and `stress_suite_notes.md`
    so the next pass can start from pinned failure cases instead of a fresh search
  - once those weak points are pinned, recheck only the current budget-first defaults with the
    micro-hardening pass:
    ```bash
    python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_study
    python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_confirm
    python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_study
    python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_confirm
    python scripts/run_local_sweep.py --study knapsack_stress_note_freeze_check
    python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
    ```
  - current read:
    - TSP `F` stays on the same file path but now uses the stress-hardened `pop40/gen33` split
    - ZDT1 `F` stays on the same file path but now uses `cooldown=3`
    - knapsack stays parked except for the narrow repair-only note
    - onemax stays control only
  - once the hardened defaults are frozen again, refresh the pinned weak points rather than
    inventing a new search space:
    ```bash
    python scripts/run_local_sweep.py --study tsp_stress_refresh_suite
    python scripts/run_local_sweep.py --study zdt1_stress_refresh_suite
    python scripts/run_local_sweep.py --study knapsack_stress_refresh_suite
    python scripts/run_local_sweep.py --study onemax_control_refresh_check
    python scripts/build_stress_refresh_registry.py --study-name tsp_stress_refresh_suite --study-name zdt1_stress_refresh_suite --study-name knapsack_stress_refresh_suite --study-name onemax_control_refresh_check
    ```
  - read those outputs as the pinned next-pass target list:
    - TSP: anti-case tail first, rescue-target ambiguity second
    - ZDT1: spread / joint safety fail ahead of plain HV tail
    - knapsack: keep only the narrow repair-note families in view
    - onemax: no active optimization target
  - if you want to check whether those top targets actually move without reopening profile
    discovery, run the stress-target reduction pass:
    ```bash
    python scripts/run_local_sweep.py --study tsp_stress_target_reduction_study
    python scripts/run_local_sweep.py --study tsp_stress_target_reduction_confirm
    python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_study
    python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_confirm
    python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
    python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
    python scripts/build_stress_target_reduction_registry.py
    ```
  - current read:
    - TSP: keep the current fast default; inversion was interesting but did not cut anti-case
      `p90/p95` enough to replace it
    - ZDT1: keep the current fast default; nearby safety tweaks were less honest once HV
      preservation stayed in the check
  - if you want to understand *why* those top targets still survive instead of trying another
    profile sweep, run the failure-trace pass:
    ```bash
    python scripts/run_local_sweep.py --study tsp_failure_trace_suite
    python scripts/run_local_sweep.py --study zdt1_failure_trace_suite
    python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
    python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
    python scripts/build_failure_hypotheses_registry.py
    ```
  - read those outputs as the next-pass mechanism registry:
    - TSP: anti-case tail is now a seed-lock-in / late-refinement question, not a generic budget cut
    - ZDT1: safety tail is now an early-plateau / late-spread / refresh-timing question
    - knapsack: keep the note narrow
    - onemax: still no active target
  - if you want to test those pinned hypotheses directly without reopening discovery, run:
    ```bash
    python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_study
    python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_confirm
    python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_study
    python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_confirm
    python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
    python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
    python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_target_hypothesis_probe_confirm --zdt1-study-name zdt1_target_hypothesis_probe_confirm
    ```
  - read those outputs as a mechanism test, not as a new search:
    - TSP: the late-refinement hypothesis is still the best current story, but the same-budget
      probe weakened it rather than closing the anti-case tail; seed lock-in stays secondary
    - ZDT1: refresh/cooldown timing mismatch alone did not explain the whole safety story, so
      spread and joint failures still need to be split in the next pass
    - knapsack: keep the note narrow
    - onemax: still no active target
  - if you want to split "population/generation tradeoff" from "timing-only mismatch" without
    reopening broad search, run:
    ```bash
    python scripts/run_local_sweep.py --study tsp_population_generation_probe_study
    python scripts/run_local_sweep.py --study tsp_population_generation_probe_confirm
    python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_study
    python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_confirm
    python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
    python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
    python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_population_generation_probe_confirm --zdt1-study-name zdt1_timing_vs_pg_probe_confirm
    ```
  - read those outputs as a same-budget target split, not as a new profile search:
    - TSP: the generation-up tradeoff helps anti-case mean but still worsens anti-case `p90/p95`,
      so the current fast default stays in place and `seed_lockin` remains secondary
    - ZDT1: the population/generation probe now carries more spread-side information than the
      timing-only probes, but joint safety still is not cleanly closed, so the defaults stay put
    - knapsack: keep the note narrow
    - onemax: still no active target
  - if you want to widen that split-target evidence without reopening broad tuning, run:
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
  - current read from that extreme-tail closeout pass:
    - TSP still keeps the same `Q/F` split because `contour_pg_gen32_pop41` only shaved
      anti-case `p95` a little while widening `max` and slightly hurting rescue-target mean, so
      the current fast default remains the honest budget-first path
    - ZDT1 keeps the same `Q/F` split but now reads more cleanly as two targets: spread fail looks
      more like a population/generation contour problem, while joint fail is still not cleanly
      explained by timing-only tweaks once HV tail preservation is included
    - knapsack stays on the narrow repair-only note, and onemax still has no active target
  - if you want to freeze the remaining TSP tail as an operating limitation and validate whether
    the strongest ZDT1 spread candidate is replacement-worthy, run:
    ```bash
    python scripts/run_local_sweep.py --study tsp_tail_freeze_recheck
python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_study
python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_confirm
    python scripts/run_local_sweep.py --study zdt1_joint_note_freeze_check
    python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
    python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_tail_freeze_recheck --zdt1-spread-study-name zdt1_spread_candidate_boundary_confirm --zdt1-joint-study-name zdt1_joint_note_freeze_check
    ```
  - current read from that freeze / validation pass:
    - TSP still keeps the same `Q/F` split, but the remaining anti-case `p95/max` tail is now
      best read as a protocol limitation of the current fixed stack rather than an active same-budget
      contour target
    - ZDT1 still keeps the same `Q/F` split overall; `spread_pg_pop41_gen88` is the strongest
      spread-side candidate, but it closes as `note_only_stress_slice` because its spread-stress
      gains did not survive the normal / stable slice cleanly enough for a fast-default rewrite
    - ZDT1 joint timing still stays note-level only because timing-only probes did not close joint
      fail without paying too much HV tail
    - knapsack stays on the narrow repair-only note, and onemax still has no active target
  - the two-stage TSP / ZDT1 gate pass then asked whether the fast profile could be used as a
    pilot before escalating to the canonical profile
  - current answer: no promoted smart gate yet
  - for TSP, `pilot_then_canonical_v1` could recover canonical-quality routes only by escalating so
    often that it cost more than `always_canonical`, while `pilot_then_canonical_v2` saved only a
    small amount of budget
  - for ZDT1, the tested gates either spent more evaluations than `always_canonical` or stayed too
    close to `always_fast` to justify a new default rule
  - read that pass as confirmation that explicit quality-first vs budget-first choice is still the
    clean local recommendation
  - the restart-portfolio pass then asked whether one long run should be replaced by several short
    fast restarts at the same total budget
  - current answer: still no promoted multistart rule
  - for TSP, `fast_x3_equal_split` improved over `fast_once` on rescue-target cases, but it still
    lost to `canonical_once` overall and damaged corridor-like anti-cases too much
  - for ZDT1, merged fast archives did not beat either `canonical_once` or `fast_once` on holdout
    hypervolume, so the current single-run canonical/fast split stays cleaner
  - the ranking-fidelity pass then asked whether those fast profiles were still useful as cheap
    screening proxies before canonical confirmation
  - current answer: no promoted triage workflow yet
  - for TSP, `always_fast_pick_top1` was cheaper but still missed the canonical best on both
    rescue-target and anti-case rows, while top-2 / top-3 canonical confirm erased that regret only
    by spending more evaluations than the full canonical sweep
  - for ZDT1, fast-budget HV ranking stayed too weak for promotion; top-3 confirm recovered the
    canonical best, but only after spending more total evaluations than canonical-all
  - keep using canonical profiles for quality-first work and fast profiles for budget-first work;
    do not treat the fast runs as promoted screening proxies yet
  - for knapsack, cheap `none` restarts did not replace the narrow `repair_only` rerun note
  - `configs/local_profiles/knapsack_restart_experimental.json` stays as a legacy experimental
    comparator, not the preferred knapsack rerun path
  - the Q/F tolerance pass then asked whether the current quality-first vs budget-first split can
    be turned into a cleaner operating envelope
  - current answer:
    - TSP now freezes Option B only: keep the split, but do not promote one numeric tolerance rule:
      the recalibration pass showed that the hardened `pop40/gen33` fast is the current least-bad
      mixed holdout tradeoff, but the anti-case tail still moves enough across seed blocks that one
      global numeric threshold would overstate confidence
    - ZDT1 keeps the split and now has the cleaner envelope note:
      the fast profile usually stays within a small HV loss band, but use Q when Pareto-shape risk
      matters more than the saved budget
    - knapsack remains a narrow repair rerun note only
  - onemax remains a fixed control problem
  - onemax remains a fixed-baseline control problem

## Local Operating Baseline Examples

Protocol examples:

```bash
python scripts/run_local_protocol.py --problem tsp --mode explore
python scripts/run_local_protocol.py --problem tsp --mode final --anti-case-suspected
python scripts/run_local_protocol.py --problem zdt1 --mode explore
python scripts/run_local_protocol.py --problem zdt1 --mode final
python scripts/run_local_protocol.py --problem knapsack --mode sanity
```

Baseline guard examples:

```bash
python scripts/run_local_sweep.py --study local_baseline_guard_tsp
python scripts/run_local_sweep.py --study local_baseline_guard_zdt1
python scripts/run_local_sweep.py --study local_baseline_guard_knapsack
python scripts/run_local_sweep.py --study local_baseline_guard_onemax
python scripts/check_local_baseline.py --write-snapshot
python scripts/check_local_baseline.py
```

Candidate workflow examples:

```bash
python scripts/check_local_baseline.py
python scripts/run_local_candidate.py --candidate configs/local_candidates/example_zdt1_candidate.json
python scripts/run_local_candidate.py --candidate configs/local_candidates/example_tsp_candidate.json --no-execute
python scripts/run_local_candidate.py --candidate configs/local_candidates/example_knapsack_candidate.json --use-existing-output
python scripts/summarize_local_candidates.py
python scripts/summarize_local_optimization_status.py
python scripts/build_local_baseline_change_request.py --candidate-report outputs/local_candidates/.../candidate_report.json --force-draft
```

Current operating baseline:

- TSP fast is budget-first / exploratory only; anti-case / corridor suspicion or quality-sensitive
  final work still goes straight to Q `8-10`.
- ZDT1 fast is exploratory / budget-first only; final safety still goes straight to Q.
- `spread_pg_pop41_gen88` is a note-only stress-slice candidate, not the default.
- knapsack keeps no broad default and only a narrow repair note.
- onemax stays control-only.
- passing a candidate report does not auto-replace a profile; it only earns a
  manual baseline-change review.

Candidate backlog read:

- `reject_regression` / `reject_no_material_gain` -> rejected
- `note_only_stress_slice` -> note_only
- `monitor_only` -> monitor
- `candidate_promising_needs_confirm` -> promising_needs_confirm
- `candidate_passes_local_guard` -> passed_local_guard
- `candidate_requires_new_mechanism_hypothesis` -> requires_new_mechanism
- `baseline_drift_detected` -> blocked_by_baseline_drift
- `intentional_baseline_change_required` -> ready_for_change_request

Change-control reminder:

- `python scripts/build_local_baseline_change_request.py` drafts review artifacts only
- it does not rewrite `configs/local_profiles/*`
- it does not refresh `artifacts/local_baseline_snapshot.json`

Optimization closeout examples:

```bash
python scripts/check_local_baseline.py
python scripts/summarize_local_candidates.py
python scripts/summarize_local_optimization_status.py
python scripts/build_local_baseline_change_request.py --candidate-report outputs/local_candidates/.../candidate_report.json --force-draft
```

Closeout read:

- TSP:
  - fast stays budget-first / exploratory only
  - anti-case / corridor suspicion or quality-sensitive final still goes straight to Q `8-10`
  - do not reopen without a new mechanism hypothesis
- ZDT1:
  - fast stays exploratory / budget-first only
  - `spread_pg_pop41_gen88` stays note-only
  - joint fail stays monitor-only and final safety stays on Q
- knapsack:
  - no broad default; narrow repair note only
- onemax:
  - control only

Reopen criteria now live in `docs/local_reopen_criteria.md`.
