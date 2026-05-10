# GA Codex Lab

`ga-codex-lab` is an evidence-backed optimization lab for comparing cheap baselines, pure genetic
algorithms, selected hybrid GA paths, and NSGA-II under matched function-evaluation budgets.

It is built to answer a practical question:

> Which solver family should you actually choose for a problem family, size tier, and objective
> priority when you care about reproducible evidence rather than algorithm branding?

## What This Repo Is And Is Not

This repo is:

- a solver-choice lab with matched-budget benchmark evidence
- a claim-governed benchmark repo with machine-readable registry and drift checks
- a place where cheap heuristics are treated as real competitors

This repo is not:

- a one-size-fits-all "GA wins everything" showcase
- a claim about validated behavior outside the tested ranges
- a substitute for family-specific reading when the evidence splits

## Problem Families Covered

| Family | Current public reading |
| --- | --- |
| onemax / monotone bitstring | practical default is `hill_climb` |
| deceptive bitstring tested family | a representative pure GA path may be better |
| knapsack | guidance is family-conditioned rather than one broad external default |
| tsp | practical default is `nearest_neighbor_2opt` |
| zdt family | pure NSGA-II remains the default path |

## Recommended Solver Snapshot

<!-- BEGIN AUTO-GENERATED: solver-matrix-preview -->
| Problem family | Priority | Recommended solver | Scope | Latest status |
| --- | --- | --- | --- | --- |
| onemax | practical default | `hill_climb` | `internal` | `PASS` |
| bitstring deceptive trap | tested deceptive quality path | `pure-ga` | `family_conditional` | `PASS` |
| knapsack uncorrelated | practical default | `greedy_local_search` | `family_conditional` | `PASS` |
| knapsack weakly correlated | quality-first family path | `hybrid-ga` | `family_conditional` | `PASS` |
| tsp | practical default | `nearest_neighbor_2opt` | `external` | `PASS` |
| tsp | quality-first | `hybrid-ga` | `internal` | `PASS` |
| zdt family | balanced multi-metric default | `nsga2` | `external` | `PASS` |
<!-- END AUTO-GENERATED: solver-matrix-preview -->

## Internal vs External Evidence Snapshot

<!-- BEGIN AUTO-GENERATED: evidence-snapshot -->
- drift-governed ci status: `PASS`
- ci-gated claims passing: `8` / `8`
- strongest current readings:
  - `tsp_external_nn2opt_default`: Nearest-neighbor + 2-opt remains the externally supported practical default on the tested TSPLIB subset. (`PASS`)
  - `zdt_external_nsga2_over_random_archive`: Pure NSGA-II remains the externally supported default path over random archive on the tested ZDT subset. (`PASS`)
  - `bitstring_monotone_hill_climb_default`: Hill climb remains the family-conditional practical default on the tested monotone bitstring subset (OneMax, LeadingOnes). (`PASS`)
  - `knapsack_correlated_seed_repair_worth_trying`: Seed-repair hybrid remains a family-conditional worth-trying path on the tested correlated knapsack subsets. (`PASS`)
  - `tsp_medium_hybrid_internal_quality_first`: The validated internal medium TSP hybrid remains a narrow quality-first path. (`PASS`)
<!-- END AUTO-GENERATED: evidence-snapshot -->

## Current Release Status

<!-- BEGIN AUTO-GENERATED: release-status -->
- overall drift status: `PASS`
- ci-gated PASS / WARN / FAIL / NOT_EVALUATED: `8` / `0` / `0` / `0`
- latest release snapshot: `2026-04-13T12:25:50Z`
<!-- END AUTO-GENERATED: release-status -->

## Solver-Choice Reading

- onemax / monotone bitstring:
  - practical default is `hill_climb`
- deceptive bitstring tested family:
  - pure GA may be meaningful on the tested trap family
- knapsack:
  - keep the reading family-conditioned
- tsp:
  - practical default is `nearest_neighbor_2opt`
  - `tsp_medium_hybrid.json` is a narrow internal quality-first path only
- zdt family:
  - pure NSGA-II remains the default path

## Supported Execution Modes

| Mode | Who it is for | What is supported |
| --- | --- | --- |
| repo dev mode | contributors | editable install, scripts, tests, benchmark regeneration |
| installed consumer mode | external users | packaged presets, packaged demos, solver recommendation helpers, out-of-tree runs |
| maintainer / release mode | maintainers | claim drift checks, release artifact rendering, benchmark tiers |

## Quickstart

Installed consumer path:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install .
```

From any working directory after install:

```bash
ga-lab-recommend-solver --problem onemax --size 128 --priority default --format json
ga-lab-recommend-preset --problem zdt1 --size 50 --priority hv --format json
ga-lab-run --preset onemax_small --output-root ./ga-lab-outputs
ga-lab-demo baseline --output-root ./ga-lab-outputs
ga-lab-demo hybrid --output-root ./ga-lab-outputs
ga-lab-demo nsga2 --output-root ./ga-lab-outputs
```

Stable Python API:

```python
from ga_lab.api import recommend_solver, run_preset

recommendation = recommend_solver("onemax", 128, "default")
result = run_preset("onemax_small", output_root="ga-lab-outputs")
```

Repo-maintainer path:

```bash
pip install -e .[dev]
ga-lab-check-claims --fail-on FAIL
ga-lab-render-release-artifacts
```

## Local Experiment Workflow

If you are using this repo as a local experiment harness, start here instead of the release or
governance docs:

```bash
python scripts/run_local_experiment.py --preset onemax_small
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
python scripts/run_local_sweep.py --study tsp_seed_fraction_ablation_study
python scripts/run_local_sweep.py --study tsp_seed_source_ablation_study
python scripts/run_local_sweep.py --study tsp_mutation_operator_ablation_study
python scripts/run_local_sweep.py --study tsp_canonical_default_confirm
python scripts/run_local_sweep.py --study zdt1_budget_note_freeze
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

Those commands create CSV, JSON, Markdown, and PNG outputs under:

- `outputs/local_runs/`
- `outputs/local_studies/`

Adaptive local studies now also emit:

- `history_summary.csv`
- `ranking_detail.csv` for candidate-level fast-versus-canonical triage rows
- `ranking_fidelity.csv` for rank-correlation and top-k recall summaries
- `triage_workflow_summary.csv` for fast-screen / canonical-confirm cost-versus-regret summaries
- `tolerance_table.csv` for quality-first versus budget-first loss-envelope summaries
- `seed_budget_table.csv` for seed-count-versus-confidence summaries on the current local defaults
- `tsp_fast_tail_summary.csv` for TSP fast-profile anti-case tail summaries against the Q/F comparator pair
- `plot_diversity.png`
- `plot_stagnation.png`
- `plot_trigger_events.png` for trigger timing in TSP, ZDT1, and knapsack adaptive runs
- `plot_post_trigger_gain.png` for TSP mechanism-isolation studies
- `plot_refresh_schedule_vs_gain.png` for TSP delayed-trigger versus periodic-refresh comparisons
- `plot_refresh_volume_vs_gain.png` for TSP refresh-volume checks
- `plot_collapse_onset_vs_trigger.png` for TSP collapse timing versus first trigger timing
- `plot_budget_band_vs_gap.png` for TSP reduced/current budget comparisons
- `plot_instance_feature_map.png` for TSP hard-case geometry feature maps
- `plot_feature_vs_policy_gap.png` for TSP feature-to-policy-margin inspection
- `plot_bridge_score_vs_trigger_value.png` for trigger-value versus bridge-score checks
- `plot_anisotropy_vs_decay_value.png` for decay-value versus anisotropy checks
- `plot_router_regret.png` for instance-aware router regret versus fixed policies
- `plot_budget_band_vs_policy_win.png` for TSP budget-band policy win counts
- `router_decision_table.md` for the simple rule actually tested in the router pass
- `plot_mode_switch_timeline.png` for online regime-switching timelines
- `plot_diversity_vs_mode.png` for diversity traces by active TSP mode
- `plot_collapse_onset_vs_switch.png` for collapse timing versus first mode switch
- `plot_regret_vs_policy.png` for regret against the best fixed TSP policy
- `plot_budget_band_vs_regret.png` for switching regret by budget band
- `plot_rescue_target_vs_anticase_gap.png` for rescue-target gain versus anti-case damage
- `plot_mutation_rate.png` for knapsack feasibility-aware mutation studies
- `plot_seed_fraction_vs_gap.png` for TSP seeded fixed-stack gap checks
- `plot_seed_source_vs_gap.png` for TSP seeding-source anatomy checks
- `plot_mutation_operator_vs_gap.png` for TSP mutation-operator simplification checks
- `plot_initial_quality_vs_final_gap.png` for TSP seeding head-start versus final gap checks
- `plot_family_vs_regret.png` for knapsack family-conditioned regret against greedy / GA baselines
- `plot_seeded_vs_repair_gap.png` for knapsack seeding versus repair anatomy checks
- `plot_repair_vs_greedy_gap.png` for repair-only versus greedy gaps on the knapsack boundary pass
- `plot_init_feasible_vs_final_gain.png` for knapsack feasible-start versus final-gain checks
- `plot_initial_feasible_fraction_vs_gain.png` for knapsack feasible-start versus gain on rerun-boundary studies
- `plot_capacity_tightness_vs_gain.png` for knapsack capacity-tightness versus gain-over-plain-GA checks
- `plot_budget_vs_feasible_gain.png` for knapsack budget-to-feasible-gain checks
- `plot_fast_vs_canonical_rank.png` for TSP fast-versus-canonical rank fidelity
- `plot_topk_recall_vs_budget.png` for triage top-k recall versus total screening cost
- `plot_triage_cost_vs_regret.png` for TSP triage workflow cost-versus-regret checks
- `plot_rescue_target_vs_anticase_rank_fidelity.png` for TSP rank fidelity split by case family
- `plot_fast_vs_canonical_hv_rank.png` for ZDT1 fast-versus-canonical HV ranking checks
- `plot_triage_cost_vs_hv_regret.png` for ZDT1 triage workflow cost-versus-HV regret checks
- `plot_spread_safety_failures.png` for ZDT1 spread / pareto safety failures during triage
- `plot_q_vs_f_loss_distribution_recalibrated.png` for the post-hardening TSP Q/F recalibration pass
- `plot_rescue_vs_anticase_loss_recalibrated.png` for recalibrated rescue-target versus anti-case TSP tails
- `plot_budget_savings_vs_quality_loss_recalibrated.png` for recalibrated TSP budget-savings versus quality-loss checks
- `plot_tolerance_accept_rate_recalibrated.png` for recalibrated TSP tolerance-bin acceptance rates
- `plot_q_vs_f_tail_distribution.png` for TSP fast-hardening loss distributions against the quality-first comparator
- `plot_candidate_vs_anti_case_p90.png` for TSP anti-case p90 checks across fast fixed-stack variants
- `plot_candidate_vs_rescue_mean.png` for TSP rescue-target mean-loss preservation across fast variants
- `plot_seed_fraction_vs_tail.png` for TSP seed-fraction versus anti-case tail checks
- `plot_operator_vs_tail.png` for TSP mutation-operator versus anti-case tail checks
- `plot_old_fast_vs_new_fast_tail.png` for the legacy-fast versus hardened-fast TSP anti-case tail comparison
- `tsp_protocol_limitation_freeze_summary.md` for the TSP freeze readout when the remaining anti-case
  `p95/max` tail is treated as a protocol limitation
- `zdt1_spread_candidate_boundary_table.csv` for slice-by-slice boundary validation of the strongest
  ZDT1 spread candidate against the current fast default
- `zdt1_spread_candidate_boundary_notes.md` for the spread-candidate decision readout
- `plot_tsp_tail_freeze_summary.png` for the frozen TSP anti-case / rescue tail readout
- `plot_anticase_q_vs_f_tail.png` for the frozen TSP anti-case Q-versus-F tail overlay
- `plot_spread_candidate_vs_currentF.png` for ZDT1 spread candidate validation against the current
  fast default
- `plot_spread_tail_validation.png` for ZDT1 spread-tail validation across stress, holdout, and
  stable slices
- `plot_hv_preservation_vs_spread_gain.png` for ZDT1 spread gain versus HV preservation
- `plot_joint_non_regression.png` for ZDT1 joint-fail non-regression checks during spread-candidate
  validation
- `plot_stable_normal_non_regression.png` for the ZDT1 stable/normal slice non-regression check
- `plot_q_vs_f_hv_loss_distribution_recheck.png` for the ZDT1 tiny Q/F freeze recheck
- `plot_parameter_sweep.png` for one-axis numeric sweeps
- problem-specific plots such as `plot_feasibility.png`, `plot_route_distance.png`, and
  `plot_hypervolume.png`
- focused diagnostic plots such as `plot_violation.png`, `plot_diversity_vs_distance.png`, and
  `plot_hv_vs_spread.png`
- `plot_budget_vs_hv.png` for ZDT1 threshold-versus-budget checks

Current local-only adaptive read:

- onemax:
  - the control rerun kept static `none` in front on evaluations-to-target
  - `switch_controller_v1` did fire occasionally, but it still finished slower than the fixed baseline
  - keep OneMax as a fixed-baseline control problem
- knapsack:
  - broad adaptive tuning still stays parked; there is still no broad knapsack local default
  - the local family pass did find one narrow experimental profile worth keeping:
    `configs/local_profiles/knapsack_repair_local_experimental.json`
  - the key anatomy result stays `repair > seeding` on the tested small local families:
    in the seeded/repair ablation, `repair_only` matched `seeded_repair`, while `seeded_only`
    usually only recovered a head start or tied the greedy comparator
  - the confirm holdout kept that reading:
    `repair_only` and `seeded_repair` tied each other, beat `none`, and beat the old
    `knapsack_restart_experimental` path on the mixed holdout summary
  - the new rerun-boundary holdout kept the note narrow rather than broad:
    `repair_only` stayed useful on `subset_sum_like_small_b` and `tight_capacity_small_b`, while
    `weakly_correlated_small_b` only tied `greedy_local_search` instead of beating it
  - the simplest local rerun note is now:
    if the small family looks subset-sum-like or tight-capacity-like, especially when a plain GA
    pilot starts with very low `initial_feasible_fraction` (roughly `<= 0.05` in the tested local
    suite), one `repair_only` rerun is worth trying
  - the new budget-efficiency gate pass did not justify a broader pilot+rereun controller:
    `repair_rerun_gate` matched `repair_only` on boundary-like families only by spending the full
    repair budget again, while anti-boundary pilot-only exits were cheaper but noticeably weaker
  - keep that as a narrow experimental rerun note, not as a broad knapsack default
  - `seeded_repair` no longer earns its own rerun slot over `repair_only`; the repair step was the
    durable mechanism and the seeding path only added runtime
  - keep `greedy_local_search` as the practical comparator and keep
    `configs/local_profiles/knapsack_restart_experimental.json` only as a legacy experimental note
  - `feasibility_aware_mutation_v1/v3` stayed cleaner than broad restart stories but still did not
    justify promotion over `repair_only` on the holdout confirm
  - do not read the repair profile as a broad knapsack default; weakly correlated families still
    tie greedy rather than cleanly beating it
- tsp:
  - quality-first:
    `configs/local_profiles/tsp_seeded_swap_local.json` stays the canonical local default for the
    current hard-case suite
  - budget-first:
    `configs/local_profiles/tsp_seeded_swap_local_fast.json` now keeps the same seeded swap stack
    but now uses the hardened fixed cut: `population_size=40`, `generations=33`
    (configured budget `1400`, still about 25% below Q)
  - the post-hardening recalibration kept this exact fast profile frozen:
    overall mean loss stayed near `0.63%`, rescue-target mean loss near `0.59%`, anti-case `p90`
    near `2.60%`, and actual evaluation savings near `25.5%`
  - that same recalibration still did not justify a global numeric tolerance rule:
    anti-case and rescue-target tails both improved over the legacy fast reference, but the mixed
    holdout tail still moves too much for a single promoted threshold
  - the current freeze is still:
    nearest-neighbor mix seeding with `seed_fraction=0.5`, `swap` mutation, no trigger, and no
    local search
  - the new seeding-anatomy pass showed that the dominant gain comes from the seeding source itself:
    `hybrid_ga` without nearest-neighbor seeding collapsed back toward `none`
  - lower seeding (`seed_fraction=0.25`) stayed competitive, especially on rescue-target holdouts,
    but it did not fully match `seed_fraction=0.5` on the mixed anti-case + overall holdout summary
  - higher seeding (`seed_fraction=0.75`) looked strong in the coarse pass, but it was not the
    simplification target, so it stays an exploratory note rather than the canonical default
  - `swap` versus `inversion` was a second-order effect:
    `seeded_inversion_seed25` stayed competitive, but `seeded_swap_seed50` remained the best mixed
    holdout choice and is simpler because it preserves the current mutation operator
  - the budget frontier pass showed that population cuts are cleaner than generation cuts here:
    `population_075` stayed almost level with the canonical profile on both rescue-target and
    anti-case holdouts, while the 50% generation cut and the early-stop wrapper saved more budget
    but lost noticeably more route quality
  - `low_diversity_injection` is now a legacy / mechanism-comparison path, not the preferred local
    TSP default
  - `decay_mutation` stays the strong fixed comparator when you want a no-trigger baseline
  - use `tsp_seed_fraction_ablation_study`, `tsp_seed_source_ablation_study`,
    `tsp_mutation_operator_ablation_study`, and `tsp_canonical_default_confirm` together:
    if `plot_seed_source_vs_gap.png` shows a large source effect while
    `plot_seed_fraction_vs_gap.png` and `plot_mutation_operator_vs_gap.png` move only a few route
    units around the current default, keep the current seeded-swap profile frozen
- zdt1:
  - quality-first:
    keep `configs/local_profiles/zdt1_diversity_injection.json` as the reusable local default at
    the current budget: `diversity_threshold=0.55`, `refresh_fraction=0.10`, `adaptation_cooldown=4`
  - budget-first:
    `configs/local_profiles/zdt1_diversity_injection_fast.json` now keeps the same adaptive rule
    but trims population from `60` to `45` and shortens the adaptive cooldown to `3`; on the
    post-hardening recalibration it kept configured budget about 25% lower with mean HV loss near
    `0.09%`, `p90` near `0.31%`, and joint safety fails on about `20%` of runs
  - the new anatomy + canonical-confirm pass kept `threshold=0.55` as the center, showed that
    `refresh_fraction=0.10` is enough to match the old `0.20` profile on the tested holdout budget
    bands, and left `cooldown=4` as the clean default over `2` or `6`
  - the budget frontier pass showed that population cuts are cleaner than generation cuts here too:
    `population_075` kept HV closest to the default while `population_050` remained a viable
    half-budget squeeze and `generations_075` paid a much larger spread penalty
  - `threshold=0.45` and `cooldown=6` both produced coarse wins, but neither was clean enough to
    replace the simpler `0.55 / 0.10 / 4` profile as the canonical local default
  - keep `0.60` only as a nearby exploratory threshold note; do not promote a budget-conditioned
    threshold rule from this local-only pass
  - the tiny freeze rerun kept `0.55 / 0.10 / 4` on top of the local note-check budget, so the
    simplified canonical default stays frozen

Quality-first vs budget-first, in one line:

- onemax:
  - quality-first and budget-first are both still the fixed `none` control; no separate fast
    profile is worth adding
- tsp:
  - quality-first: `tsp_seeded_swap_local.json`
  - budget-first: `tsp_seeded_swap_local_fast.json`
- zdt1:
  - quality-first: `zdt1_diversity_injection.json`
  - budget-first: `zdt1_diversity_injection_fast.json`
- knapsack:
  - quality-first: `greedy_local_search` as the practical comparator, with
    `knapsack_repair_local_experimental.json` only as a narrow rerun note
  - budget-first: no separate fast profile; the pilot+rereun gate still stays experimental

Two-stage pilot / escalation read:

- tsp:
  - keep the manual split between `tsp_seeded_swap_local.json` and
    `tsp_seeded_swap_local_fast.json`
  - the tested smart gates did not earn promotion
  - `pilot_then_canonical_v1` matched or slightly beat canonical route quality only by
    escalating too often and spending more total evaluations than `always_canonical`
  - `pilot_then_canonical_v2` saved only a small amount of budget and did not improve the
    quality/budget tradeoff enough to justify the extra rule
- zdt1:
  - keep the manual split between `zdt1_diversity_injection.json` and
    `zdt1_diversity_injection_fast.json`
  - the tested smart gates either cost more than `always_canonical` or stayed too close to
    `always_fast` to justify a separate local rule
- knapsack:
  - keep `knapsack_repair_local_experimental.json` as a narrow manual rerun note only
  - the small pilot+rereun sanity check did not beat the simpler note once total rerun cost was
    counted
- onemax:
  - keep `none` as a fixed control; no smart gate is worth carrying here

Multi-start portfolio read:

- tsp:
  - keep `tsp_seeded_swap_local.json` as the quality-first default
  - keep `tsp_seeded_swap_local_fast.json` as the budget-first default
  - `fast_x3_equal_split` helped rescue-target cases relative to `fast_once`, but it still lost to
    canonical overall and paid too much anti-case damage to become a promoted multistart rule
  - read the portfolio pass as a reminder that multiple short fast restarts can be a diagnostic
    probe on bridge/ring-like cases, not a new default
- zdt1:
  - keep `zdt1_diversity_injection.json` as the quality-first default
  - keep `zdt1_diversity_injection_fast.json` as the budget-first default
  - merged fast restarts (`fast_x2_equal_split`, `fast_x2_budget075`) did not beat either
    `canonical_once` or `fast_once` on holdout HV, so there is no promoted multistart archive rule
- knapsack:
  - keep `knapsack_repair_local_experimental.json` as a narrow repair rerun note only
  - `repair_only` still beat `none_x2_equal_split` on the tested family sanity rows, so broad
    multistart remains parked
- onemax:
  - keep `none` as the fixed control; portfolio restarts add no practical value here

Practical recommendation:

- choose the quality-first profile directly when final quality matters most
- choose the budget-first fast profile directly when you are in a quick local tuning loop
- do not promote a multistart portfolio note yet; for the current local suite, single-run
  canonical vs single-run fast is still the cleaner split
- do not add a promoted `*_smart.json` gate yet; the manual split is still the cleaner local rule
- do not promote a triage workflow yet either:
  - on TSP, fast-budget ranking stayed only moderately aligned with canonical
    (`Spearman ~= 0.79`, `top_1_match ~= 0.33`)
  - on ZDT1, fast-budget HV ranking was weaker still (`Spearman ~= 0.26`, `top_1_match = 0`)
  - in both problems, top-k canonical confirm recovered quality only by spending more total
    evaluations than the full canonical sweep
- treat fast profiles as budget-first final runs, not as promoted screening proxies

Q/F tolerance envelope read:

- tsp:
  - keep the existing manual split between `tsp_seeded_swap_local.json` (Q) and
    `tsp_seeded_swap_local_fast.json` (F)
  - the recalibration pass then compared Q against both the hardened fast profile
    (`population_size=40`, `generations=33`) and the legacy fast reference
  - on the post-hardening recalibration confirm, the current F kept median loss at `0`, mean loss
    near `0.63%`, `p90` near `2.52%`, about `25.5%` actual-evaluation savings, and about `24.8%`
    runtime savings against Q
  - rescue-target mean loss stayed near `0.59%`, anti-case mean loss near `0.67%`, and the current
    F remained the least-bad mixed holdout tradeoff after the fast-hardening pass
  - however, coarse-versus-confirm seed blocks still moved the anti-case tail enough that this repo
    still does not freeze one explicit global TSP tolerance threshold such as `0.25%` or `0.50%`
  - this currently closes as Option B only: descriptive split, no promoted numeric tolerance rule
  - practical read:
    use Q when route quality is sensitive, when an anti-case/corridor-like instance is plausible,
    or when you are closing a hard-case final; use the hardened F for quick local loops or
    budget-first final runs when occasional low-single-digit tail loss is acceptable
- zdt1:
  - keep `zdt1_diversity_injection.json` as Q and `zdt1_diversity_injection_fast.json` as F
  - the hardened fast profile now keeps the same population cut plus `cooldown=3`
  - the recalibration confirm cut actual evaluations by about `25%` and runtime by about `44%`
  - HV loss stayed tight on average (`mean ~= 0.09%`, `median ~= 0.08%`, `p90 ~= 0.31%`,
    `max ~= 0.47%`)
  - `pareto_ratio` stayed flat, but spread / joint safety still failed on about `20%` of the
    tested runs, so the final safety read still belongs to Q
  - practical local envelope:
    if about `0.25%` HV loss is acceptable and a rare spread miss is tolerable, F is a reasonable
    budget-first final run; if you want stricter Pareto-shape protection, stay on Q
- knapsack:
  - keep `knapsack_repair_local_experimental.json` only as a narrow rerun note
  - the tiny freeze check still favored `repair_only` on the tested subset-sum-like / tight-capacity
    rows, but there is still no broad Q/F-style operating rule
- onemax:
  - keep `none` as the control; the freeze check still shows no need for a separate Q/F envelope

Seed-budget calibration read:

- tsp:
  - one-seed Q/F reads are still too noisy; on the seed-budget holdout pool, `n=1` flipped the
    overall call and temporarily made the anti-case slice look F-favorable
  - from `n=3` onward the overall / anti-case direction stayed on the Q side, while rescue-target
    rows were already Q-leaning even at `n=1`
  - practical rule:
    use F for quick exploratory loops with `1-3` seeds, but if the run is quality-sensitive or an
    anti-case / corridor-like instance is plausible, confirm with Q on `8-10` seeds
- zdt1:
  - `n=1` is too optimistic because HV can look fine before spread safety failures show up
  - `n=3-5` is enough for exploratory comparison, but the full HV plus spread / pareto safety read
    only settles cleanly around `8-10` seeds
  - practical rule:
    use F on `3` seeds for quick budget-first iteration, then use `8-10` seeds if you want a
    final Q/F call with Pareto-shape safety included
- knapsack:
  - the narrow repair note stayed stable even on the short ladder, but the honest operating rule is
    still only “sanity-check on `3` seeds, use `5` if the family looks borderline”
- onemax:
  - the control comparison stayed flat from `n=1`, so there is still no reason to spend a larger
    seed ladder here

Paired-seed sequential compare read:

- tsp:
  - exploratory loop:
    paired `n=3` is already enough to keep `F` on the overall slice and the anti-case slice, while
    rescue-target-only rows are the one place where extending to paired `n=5` still pays off
  - quality-sensitive call:
    if anti-case risk matters, the current honest rule is still "go straight to `Q`, confirm on
    paired `8-10` seeds"
- zdt1:
  - exploratory budget-first read:
    paired `n=3` is already enough to keep `F` for a cheap HV-first loop
  - final safety-aware call:
    the spread / Pareto safety misses remain too common for `F`, so the practical rule is still
    `Q` on paired `8-10` seeds
- knapsack:
  - the narrow `repair_only` note is still a paired `3`-seed sanity check, with paired `5` only
    for tight-capacity or otherwise borderline-looking rows
- onemax:
  - the control comparison stayed flat immediately, so there is still no reason to spend a
    sequential ladder here

The sequential pass writes `sequential_decision_table.csv` plus TSP / ZDT1 paired-seed plots such
as `plot_seed_stage_vs_ci_width.png`, `plot_seed_stage_vs_decision_flip.png`, and
`plot_seed_stage_vs_safety_fail_rate.png`.

Stress-suite read after the protocol freeze:

- current `Q` defaults stay unchanged in this pass
- TSP `F` stays on the same profile path, but the file now uses the stress-hardened
  `population_size=40`, `generations=33` split because that lowered mean loss and anti-case tail at
  the same budget
- ZDT1 `F` stays on the same profile path, but the file now uses `adaptation_cooldown=3` because
  that lowered spread / joint safety fails at the same budget
- TSP future optimization should still start from anti-case corridor tails and rescue-target
  ambiguity, not from a broad new profile search
- ZDT1 future optimization should still start from spread / Pareto safety-fail seeds, not from a
  new Q/F split
- knapsack stays broadly parked; keep only the narrow repair-only note on subset-sum-like /
  tight-capacity rows and watch weakly-correlated borderline cases
- use the small stress suite like this:

```bash
python scripts/run_local_sweep.py --study tsp_stress_suite
python scripts/run_local_sweep.py --study zdt1_stress_suite
python scripts/run_local_sweep.py --study knapsack_stress_suite
python scripts/run_local_sweep.py --study onemax_control_stress_check
```

- each stress study writes `stress_case_catalog.csv`, `stress_case_catalog.md`,
  `tail_risk_summary.csv`, and `stress_suite_notes.md` alongside the usual CSV / Markdown / PNG
  outputs
- if you want to recheck only the current budget-first defaults against their pinned weak points,
  use the micro-hardening pass:

```bash
python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_study
python scripts/run_local_sweep.py --study tsp_fast_stress_hardening_confirm
python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_study
python scripts/run_local_sweep.py --study zdt1_fast_stress_hardening_confirm
python scripts/run_local_sweep.py --study knapsack_stress_note_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
```

- current read:
  - TSP `F` is now the stress-hardened `pop40/gen33` split, but anti-case-sensitive finals still
    belong to `Q`
  - ZDT1 `F` is now the stress-hardened `cooldown=3` variant, but final safety still belongs to `Q`
  - knapsack broad default is still parked
  - onemax stays control only

Frozen local operating protocol read:

- the current Q/F split, seed budgets, and narrow knapsack repair-note sanity rule are now also
  frozen as a direct local protocol helper
- use it like this:

```bash
python scripts/run_local_protocol.py --problem tsp --mode explore
python scripts/run_local_protocol.py --problem tsp --mode compare --case-group rescue_target
python scripts/run_local_protocol.py --problem tsp --mode final --anti-case-suspected
python scripts/run_local_protocol.py --problem zdt1 --mode explore
python scripts/run_local_protocol.py --problem zdt1 --mode compare --final-safety
python scripts/run_local_protocol.py --problem knapsack --mode sanity --borderline
python scripts/run_local_protocol.py --problem onemax --mode control
```

- practical matrix:
  - TSP:
    - explore: `F` on paired `3` seeds
    - compare: paired `Q` vs `F` at `3`, but rescue-target-only ambiguity still earns the `5`-seed
      step
    - final: if anti-case suspicion or quality sensitivity matters, go straight to `Q` on paired
      `8-10` seeds
  - ZDT1:
    - explore: `F` on paired `3` seeds
    - final safety: `Q` on paired `8-10` seeds
  - knapsack:
    - keep `greedy_local_search` as the default practical baseline
    - keep `repair_only` only as a narrow `3 -> 5` seed sanity note
  - onemax:
    - keep `none` as the `1`-seed control
- the helper writes `protocol_decision.json` / `protocol_decision.md` to
  `outputs/local_protocols/...`, and `--execute` reuses the current study manifests rather than
  inventing a new local workflow
- the protocol matrix is still structurally unchanged after fast-default hardening, but the
  current stress refresh now freezes where the current defaults still wobble:
  ```bash
  python scripts/run_local_sweep.py --study tsp_stress_refresh_suite
  python scripts/run_local_sweep.py --study zdt1_stress_refresh_suite
  python scripts/run_local_sweep.py --study knapsack_stress_refresh_suite
  python scripts/run_local_sweep.py --study onemax_control_refresh_check
  python scripts/build_stress_refresh_registry.py --study-name tsp_stress_refresh_suite --study-name zdt1_stress_refresh_suite --study-name knapsack_stress_refresh_suite --study-name onemax_control_refresh_check
  ```
  - these write `current_stress_case_catalog.csv`, `tail_risk_refresh_summary.csv`,
    `future_optimization_targets.*`, and `stress_refresh_notes.md` under
    `outputs/local_studies/`
  - current read:
    - TSP keeps the same `Q/F` split, with corridor-like anti-case tail as the top future target
      and rescue-target ambiguity as the secondary one
    - ZDT1 keeps the same exploratory-vs-final split, with spread / joint safety failures ahead of
      plain HV tail
    - knapsack stays broadly parked; keep only the narrow repair-only note on subset-sum-like /
      tight-capacity rows plus weakly correlated borderline ties
    - onemax still has no active optimization target
- the next pass then asked a narrower question: can those pinned top targets actually be reduced at
  the same fast budget without opening a new profile search?
  ```bash
  python scripts/run_local_sweep.py --study tsp_stress_target_reduction_study
  python scripts/run_local_sweep.py --study tsp_stress_target_reduction_confirm
  python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_study
  python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_confirm
  python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
  python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
  python scripts/build_stress_target_reduction_registry.py
  ```
  - these write `tail_risk_reduction_summary.csv` inside the TSP / ZDT1 study dirs, then refresh
    `future_optimization_targets.*` and `stress_reduction_notes.md` under `outputs/local_studies/`
  - current read:
    - TSP keeps the same `tsp_seeded_swap_local_fast.json` budget-first path; the inversion
      micro-tweak lowered mean loss and max tail, but it did not cut anti-case `p90/p95` enough to
      replace the current default
    - ZDT1 keeps the same `zdt1_diversity_injection_fast.json` budget-first path; nearby
      refresh/cooldown tweaks either reopened safety fails or paid too much HV to be more honest
      than the current default
    - next local optimization should still start from `tsp_fast_anti_case_tail`, then
      `zdt1_fast_spread_safety_fail` / `zdt1_fast_joint_safety_fail`, with
      `tsp_rescue_target_ambiguity` still secondary
- the next pass now asks a narrower question: why do those pinned targets survive at all under the
  current defaults?
  ```bash
  python scripts/run_local_sweep.py --study tsp_failure_trace_suite
  python scripts/run_local_sweep.py --study zdt1_failure_trace_suite
  python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
  python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
  python scripts/build_failure_hypotheses_registry.py
  ```
  - these write `failure_trace_table.csv` / `failure_hypotheses.*` inside the TSP / ZDT1 study
    dirs, then refresh `future_optimization_targets.*` and `failure_trace_notes.md` under
    `outputs/local_studies/`
  - current read:
    - TSP still keeps the same `Q/F` split, but the dominant explanation is now expected to be
      some mix of anti-case seed lock-in and late refinement deficit rather than a generic “fast is
      worse” statement
    - ZDT1 still keeps `F` for exploratory work and `Q` for final safety, but the remaining weak
      point is now framed as an early-front-plateau / late-spread-collapse / refresh-timing
      mismatch question instead of a broad safety complaint
    - knapsack stays on the narrow repair-only note, and onemax still has no active target
  - if you want to test those pinned hypotheses directly without reopening profile discovery, run:
    ```bash
    python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_study
    python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_confirm
    python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_study
    python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_confirm
    python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
    python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
    python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_target_hypothesis_probe_confirm --zdt1-study-name zdt1_target_hypothesis_probe_confirm
    ```
  - current read from that target-specific probe pass:
    - TSP still keeps the same `Q/F` split because the same-budget late-refinement probe improved
      some mean rows but did not lower anti-case `p90/p95`, while seed lock-in still read as
      secondary rather than primary
    - ZDT1 still keeps `F` for exploratory work and `Q` for final safety because cooldown-only
      timing probes bought fewer safety fails only by paying too much HV, while refresh-only timing
      made the spread/joint slice worse
    - knapsack stays on the narrow repair-only note, and onemax still has no active target
  - if you want to check whether the TSP anti-case tail is really a population/generation tradeoff
    while ZDT1 spread versus joint failure should be split, run:
    ```bash
    python scripts/run_local_sweep.py --study tsp_population_generation_probe_study
    python scripts/run_local_sweep.py --study tsp_population_generation_probe_confirm
    python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_study
    python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_confirm
    python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
    python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
    python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_population_generation_probe_confirm --zdt1-study-name zdt1_timing_vs_pg_probe_confirm
    ```
  - current read from that same-budget tradeoff pass:
    - TSP still keeps the same `Q/F` split because the generation-up / population-down probe helped
      anti-case mean but still made anti-case `p90/p95` worse, so `late_refinement_deficit`
      remains the best current story only in a weakened form
    - ZDT1 still keeps `F` for exploratory work and `Q` for final safety, but the spread slice now
      looks more like a population/generation tradeoff than a timing-only mismatch, while joint
      fail still is not cleanly closed by timing probes
    - knapsack stays on the narrow repair-only note, and onemax still has no active target
  - if you want to close that split with one more extreme-tail pass before freezing the registry
    again, run:
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
    - TSP still keeps the same `Q/F` split because the closest same-budget contour point
      (`contour_pg_gen32_pop41`) only nudged anti-case `p95` down (`2.8977 -> 2.8741`) while
      making anti-case `max` worse (`3.0665 -> 3.5291`) and slightly hurting rescue-target mean,
      so the current fast default remains the least-bad budget-first tradeoff
    - ZDT1 still keeps `F` for exploratory work and `Q` for final safety, but now with clearer
      split wording: spread fail looks more like a population/generation contour issue, while joint
      fail still is not honestly closed by timing-only tweaks once HV tail preservation is included
    - knapsack stays on the narrow repair-only note, and onemax still has no active target
  - if you want to stop reopening the TSP contour and instead validate whether the strongest ZDT1
    spread candidate can really replace the fast default, run:
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
    - TSP still keeps the same `Q/F` split, but the remaining anti-case `p95/max` tail is now best
      read as a protocol limitation of the current fixed stack rather than an active same-budget
      contour-tuning target
    - protocol implication for TSP stays explicit: budget-first `F` is still for exploratory or
      fast budget-first work, while anti-case / corridor suspicion or quality-sensitive finals go
      straight to `Q 8-10`
    - ZDT1 still keeps `F` for exploratory work and `Q` for final safety; `spread_pg_pop41_gen88`
      is now closed as `note_only_stress_slice` because its spread-stress gain did not survive the
      normal / stable slice cleanly enough for a same-name replacement
    - ZDT1 joint timing stays note-level only: timing-only tweaks did not close joint fail without
      paying too much HV tail
    - knapsack stays on the narrow repair-only note, and onemax still has no active target

Use these as next-experiment hints, not as broad claims.

See [Local experiment guide](docs/local_experiment_guide.md) and
[Local protocol guide](docs/local_protocol_guide.md).

More:

- [Install](docs/install.md)
- [Quickstart](docs/quickstart.md)
- [Local experiment guide](docs/local_experiment_guide.md)
- [Python API](docs/python_api.md)
- [API stability](docs/api_stability.md)
- [FAQ](docs/faq.md)
- [Examples](examples/README.md)
- [Project card](docs/project_card.md)
- [Benchmark card](docs/benchmark_card.md)
- [Solver matrix](docs/solver_matrix.md)

## Benchmark / Governance / Reproducibility

- [Reproducibility and governance](docs/reproducibility_and_governance.md)
- [Release status](docs/release_status.md)
- [Benchmark card](docs/benchmark_card.md)
- [External validity](docs/external_validity.md)
- [External family solver guide](docs/external_family_solver_guide.md)
- [Solver choice guide](docs/solver_choice_guide.md)

## Key Limitations

- Do not generalize validated ranges beyond the tested sizes.
- Do not generalize monotone bitstring guidance to deceptive or multimodal bitstring families.
- Do not read knapsack as having one broad external-wide default.
- Do not overstate `tsp_medium_hybrid.json`; it stays narrow and internal-only.
- Do not collapse large-tier ZDT into a scalar one-best story.

## Docs Map

- newcomer:
  - [docs/README.md](docs/README.md)
  - [docs/project_card.md](docs/project_card.md)
  - [docs/quickstart.md](docs/quickstart.md)
- evaluator:
  - [docs/benchmark_card.md](docs/benchmark_card.md)
  - [docs/release_status.md](docs/release_status.md)
  - [docs/ablation_and_claims.md](docs/ablation_and_claims.md)
- package user:
  - [docs/python_api.md](docs/python_api.md)
  - [docs/api_stability.md](docs/api_stability.md)
  - [examples/README.md](examples/README.md)
- maintainer:
  - [docs/reproducibility_and_governance.md](docs/reproducibility_and_governance.md)
  - [docs/preset_scale_guide.md](docs/preset_scale_guide.md)
  - [docs/large_preset_decision_guide.md](docs/large_preset_decision_guide.md)

## Experimental Candidate Profiles

Experimental candidate profiles are documented under
[docs/candidates/](docs/candidates/index.md). These profiles are not default
algorithms and must be selected explicitly. Candidate O is approved only as a
restricted opt-in experimental profile for local ZDT-family exploratory
research.

## Public Release Artifacts

- [artifacts/claim_matrix.json](artifacts/claim_matrix.json)
- [artifacts/solver_matrix.json](artifacts/solver_matrix.json)
- [artifacts/release_snapshot.json](artifacts/release_snapshot.json)
- [docs/release_notes_v0.1.0.md](docs/release_notes_v0.1.0.md)

## Local Baseline Freeze

The local-only workflow now has a frozen regression baseline in `artifacts/local_baseline_snapshot.json`
plus a cheap checker in `artifacts/local_baseline_check.json`.

- TSP:
  - `configs/local_profiles/tsp_seeded_swap_local_fast.json` stays the budget-first / exploratory path
  - anti-case / corridor suspicion or quality-sensitive final still goes straight to
    `configs/local_profiles/tsp_seeded_swap_local.json` on `8-10` seeds
  - `tsp_fast_anti_case_tail` is frozen as a protocol limitation, not an active same-budget contour
    target
- ZDT1:
  - `configs/local_profiles/zdt1_diversity_injection_fast.json` stays the exploratory / budget-first
    path
  - `configs/local_profiles/zdt1_diversity_injection.json` still owns final safety
  - `spread_pg_pop41_gen88` stays `note_only_stress_slice`, not the default
  - `zdt1_fast_joint_safety_fail` stays `monitor_only`
- knapsack:
  - no broad default
  - keep `configs/local_profiles/knapsack_repair_local_experimental.json` as a narrow repair-only
    note on subset-sum / tight-capacity-like rows
- onemax:
  - `none` control only

Minimal baseline guard commands:

```bash
python scripts/run_local_sweep.py --study local_baseline_guard_tsp
python scripts/run_local_sweep.py --study local_baseline_guard_zdt1
python scripts/run_local_sweep.py --study local_baseline_guard_knapsack
python scripts/run_local_sweep.py --study local_baseline_guard_onemax
python scripts/check_local_baseline.py --write-snapshot
python scripts/check_local_baseline.py
```

Use the assisted runner for the current operating paths:

```bash
python scripts/run_local_protocol.py --problem tsp --mode explore
python scripts/run_local_protocol.py --problem tsp --mode final --anti-case-suspected
python scripts/run_local_protocol.py --problem zdt1 --mode explore
python scripts/run_local_protocol.py --problem zdt1 --mode final
python scripts/run_local_protocol.py --problem knapsack --mode sanity
```

## Regression-Gated Candidate Workflow

The frozen local baseline is now also the admission gate for new local candidate
ideas.

- new candidates must be expressed as manifests under `configs/local_candidates/`
- always run `python scripts/check_local_baseline.py` first
- candidate reports never auto-replace `configs/local_profiles/*`
- baseline drift and candidate improvement are intentionally separated
- candidate reports now roll up into a ledger and an optional manual change-request draft

Candidate commands:

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

Current admission read:

- TSP:
  - a same-budget contour rerun is not enough by itself
  - because `tsp_fast_anti_case_tail` is frozen as a protocol limitation, a new
    TSP fast candidate now needs a genuinely new mechanism hypothesis before it
    can even be considered for replacement
- ZDT1:
  - spread-side candidates can survive as `note_only_stress_slice` when the
    spread-stress slice improves but stable/normal non-regression is still weak
  - final safety still belongs to `Q`, so any candidate that worsens joint
    safety or HV tail is rejected
- knapsack:
  - broad default promotion is still forbidden
  - `repair_only` can only survive as a narrow family-conditioned note
- onemax:
  - still control only

Current backlog / change-control read:

- `artifacts/local_candidate_ledger.*` is the backlog view
- decision labels and lifecycle states are different on purpose:
  - label = what happened in the comparison
  - lifecycle = what should happen next in the backlog
- `candidate_passes_local_guard` or `intentional_baseline_change_required`
  may open a normal change-request draft
- `note_only_stress_slice`, `monitor_only`, and `candidate_requires_new_mechanism_hypothesis`
  do not change the baseline; they stay as note / monitor / blocked backlog items

See [Local candidate workflow](docs/local_candidate_workflow.md),
[Local change control](docs/local_change_control.md),
[Local protocol guide](docs/local_protocol_guide.md), and
[Local experiment guide](docs/local_experiment_guide.md).

## Local Optimization Cycle Closeout

The first local-only optimization cycle is now frozen as a status snapshot, not
as an open-ended tuning queue.

- current baseline remains the comparison anchor
- no profile change is currently pending
- no candidate in the ledger is ready to change the baseline
- future work must start with `python scripts/check_local_baseline.py`
- future work must enter through a candidate manifest, not a direct profile edit

Closeout / reopen commands:

```bash
python scripts/check_local_baseline.py
python scripts/summarize_local_candidates.py
python scripts/summarize_local_optimization_status.py
python scripts/build_local_baseline_change_request.py --candidate-report outputs/local_candidates/.../candidate_report.json --force-draft
```

Current closeout read:

- TSP:
  - `F` stays budget-first / exploratory
  - anti-case / corridor suspicion or quality-sensitive final still goes straight
    to `Q 8-10`
  - `tsp_fast_anti_case_tail` is frozen as a protocol limitation, so do not
    reopen TSP contour tuning without a new mechanism hypothesis
- ZDT1:
  - `F` stays exploratory / budget-first
  - final safety still belongs to `Q`
  - `spread_pg_pop41_gen88` stays `note_only_stress_slice`
  - `zdt1_fast_joint_safety_fail` stays `monitor_only`
- knapsack:
  - no broad default
  - `repair_only` stays narrow-note only
- onemax:
  - control only
  - reopen only if control drift appears

Reopen criteria live in [docs/local_reopen_criteria.md](docs/local_reopen_criteria.md).


