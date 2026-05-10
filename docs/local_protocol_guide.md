# Local Protocol Guide

This guide freezes the current local operating protocol into a small matrix plus one helper script.
It does not introduce a new adaptive family, router, gate, or profile. It only turns the current
Q/F split, seed-budget notes, and narrow repair-note sanity path into a day-to-day local workflow.

After the latest fast-default hardening plus stress-target reduction pass, the protocol matrix
still stayed structurally the same:

- TSP budget-first `F` now points at the hardened `pop40/gen33` fixed stack, but anti-case-aware
  finals still go straight to `Q`
- ZDT1 budget-first `F` now points at the hardened `cooldown=3` profile, but final safety still
  belongs to `Q 8-10`
- knapsack and onemax stayed frozen exactly as narrow-note / control-only paths

## Protocol Matrix

| problem | exploratory mode rule | compare mode rule | final mode rule | default profile(s) | seed ladder | stop label semantics |
| --- | --- | --- | --- | --- | --- | --- |
| tsp | `F` (`configs/local_profiles/tsp_seeded_swap_local_fast.json`) for quick local loops | start paired `Q` vs `F` at `3` seeds; if the read is rescue-target-only, use `5`; if anti-case suspicion or quality sensitivity matters, skip compare and go straight to `Q` | `Q` (`configs/local_profiles/tsp_seeded_swap_local.json`) on `8-10` seeds | `Q = tsp_seeded_swap_local`, `F = tsp_seeded_swap_local_fast` | explore/compare/final: `3 -> 5 -> 8 -> 10` | `accept_fast_exploratory` means the cheap local loop can stop on `F`; `reject_early` / `require_q_confirm` mean quality-sensitive work should move to `Q` |
| zdt1 | `F` (`configs/local_profiles/zdt1_diversity_injection_fast.json`) for quick HV-first reads | start paired `Q` vs `F` at `3` seeds; if final spread / Pareto safety matters, skip compare and go straight to `Q` | `Q` (`configs/local_profiles/zdt1_diversity_injection.json`) on `8-10` seeds | `Q = zdt1_diversity_injection`, `F = zdt1_diversity_injection_fast` | explore/compare/final: `3 -> 5 -> 8 -> 10` | `accept_fast_exploratory` means HV-first iteration can stay on `F`; `require_q_confirm` means the safety-aware final still belongs to `Q` |
| knapsack | broad default stays parked; use `greedy_local_search` as the default practical baseline | narrow sanity only: `repair_only` note on `3` seeds, extend to `5` only for tight-capacity or otherwise borderline rows | broad final rule still parked | `default = greedy_local_search`, `repair note = knapsack_repair_local_experimental` | sanity: `3 -> 5` | `repair_note_stable` means the narrow repair note still holds; it is not a promoted broad default |
| onemax | control only on `none` | no meaningful compare ladder | control only on `none` | `none` | control: `1` | `control_stable` means no broader ladder is worth spending here |

## Assisted Runner

Use the helper like this:

```bash
python scripts/run_local_protocol.py --problem tsp --mode explore
python scripts/run_local_protocol.py --problem tsp --mode compare --case-group rescue_target
python scripts/run_local_protocol.py --problem tsp --mode final --anti-case-suspected
python scripts/run_local_protocol.py --problem zdt1 --mode explore
python scripts/run_local_protocol.py --problem zdt1 --mode compare --final-safety
python scripts/run_local_protocol.py --problem knapsack --mode sanity --borderline
python scripts/run_local_protocol.py --problem onemax --mode control
```

Each call writes a small protocol bundle to `outputs/local_protocols/<timestamp>_<problem>_<mode>/`
with:

- `protocol_decision.json`
- `protocol_decision.md`
- optional study outputs when `--execute` is used

The JSON contract includes:

- `problem`
- `mode`
- `recommended_profile`
- `paired_compare_needed`
- `initial_seed_count`
- `escalation_path`
- `final_confirm_recommendation`
- `rationale`
- `output_paths`

## Execute Mode

`--execute` is still a local helper, not a new benchmarking surface. It reuses the current study
manifests and only slices them down to the recommended profile(s), seed count, and case-group hint.

Examples:

```bash
python scripts/run_local_protocol.py --problem tsp --mode compare --case-group rescue_target --execute --explain
python scripts/run_local_protocol.py --problem zdt1 --mode compare --execute --explain
python scripts/run_local_protocol.py --problem knapsack --mode sanity --borderline --execute
```

For compare/sanity modes, `--execute` also surfaces the selected stop label from the current
sequential study row so the output bundle tells you whether the helper stopped with:

- `accept_fast_exploratory`
- `accept_fast_budget_final`
- `reject_early`
- `require_q_confirm`
- `repair_note_stable`
- `control_stable`

## Problem Notes

- TSP:
  - anti-case / corridor suspicion still goes straight to `Q` because the fast profile remains
    useful for budget-first work, not for the quality-sensitive anti-case final
  - the post-hardening recalibration still did not justify a global numeric tolerance rule; it only
    made the descriptive split easier to defend
  - rescue-target-only rows are the one place where paired `5` still matters during compare mode
- ZDT1:
  - `F` is still fine for a quick HV-first loop
  - the `cooldown=3` hardening tightened the mean HV / safety read, but not enough to move the
    final safety path away from `Q`
  - final safety is still `Q 8-10` because spread / Pareto misses remain the real closing risk
- Knapsack:
  - keep only the narrow repair note
  - the helper is there to prevent over-claiming a broad default, not to create one
- Onemax:
  - keep the control path simple; a richer ladder still buys nothing here

## Stress Refresh Follow-Up

The protocol matrix above still stays frozen. The current follow-up is a refresh catalog plus a
future-target registry, not a new operating rule:

```bash
python scripts/run_local_sweep.py --study tsp_stress_refresh_suite
python scripts/run_local_sweep.py --study zdt1_stress_refresh_suite
python scripts/run_local_sweep.py --study knapsack_stress_refresh_suite
python scripts/run_local_sweep.py --study onemax_control_refresh_check
python scripts/build_stress_refresh_registry.py --study-name tsp_stress_refresh_suite --study-name zdt1_stress_refresh_suite --study-name knapsack_stress_refresh_suite --study-name onemax_control_refresh_check
```

Read those outputs like this:

- `current_stress_case_catalog.*`
  - the refreshed worst-tail, safety-fail, ambiguity, and borderline rows against the current
    hardened defaults
- `tail_risk_refresh_summary.csv`
  - the current `mean / p90 / p95 / max` read for the frozen defaults
- `future_optimization_targets.*`
  - the pinned next-pass targets
- `stress_refresh_notes.md`
  - short wording updates that keep the protocol honest without changing its structure

Current target registry read:

- TSP:
  - keep `configs/local_profiles/tsp_seeded_swap_local_fast.json` as the budget-first path
  - highest-priority target: `tsp_fast_anti_case_tail`
  - second target: `tsp_rescue_target_ambiguity`
  - protocol implication: anti-case suspicion still goes straight to `Q`, and rescue-target-only
    ambiguity is the only place where the compare ladder still earns the extra `5`-seed step
- ZDT1:
  - keep `configs/local_profiles/zdt1_diversity_injection_fast.json` as the budget-first path
  - highest-priority targets: `zdt1_fast_spread_safety_fail` and
    `zdt1_fast_joint_safety_fail`
  - protocol implication: the current `F 3 exploratory / Q 8-10 final safety` split stays
    correct even after hardening
- knapsack:
  - keep only the narrow repair-only note
  - watch `knapsack_repair_boundary_subset_sum_tight_capacity` before touching anything broader
- onemax:
  - keep `none` as the control
  - `onemax_no_active_target` means the current control path is only there for instrumentation

## Stress-Target Reduction Follow-Up

The protocol matrix still does not change here. This pass only checks whether the pinned top
targets can actually be reduced without reopening the search space:

```bash
python scripts/run_local_sweep.py --study tsp_stress_target_reduction_study
python scripts/run_local_sweep.py --study tsp_stress_target_reduction_confirm
python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_study
python scripts/run_local_sweep.py --study zdt1_stress_target_reduction_confirm
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_stress_target_reduction_registry.py
```

Read the result like this:

- TSP:
  - keep `configs/local_profiles/tsp_seeded_swap_local_fast.json` as the budget-first path
  - the inversion micro-tweak was interesting, but it did not lower anti-case `p90/p95` enough to
    justify rewriting the protocol row
  - protocol implication: anti-case suspicion still goes straight to `Q`
- ZDT1:
  - keep `configs/local_profiles/zdt1_diversity_injection_fast.json` as the budget-first path
  - nearby refresh/cooldown tweaks did not beat the current fast default once HV preservation and
    joint safety were read together
  - protocol implication: the current `F 3 exploratory / Q 8-10 final safety` row stays intact
- knapsack:
  - keep only the narrow repair-only note
- onemax:
  - keep the control row exactly as-is
    hygiene

## Failure-Trace Follow-Up

The protocol matrix still does not change here either. This pass only asks why the pinned TSP /
ZDT1 targets survive under the current defaults:

```bash
python scripts/run_local_sweep.py --study tsp_failure_trace_suite
python scripts/run_local_sweep.py --study zdt1_failure_trace_suite
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_failure_hypotheses_registry.py
```

Read the result like this:

- TSP:
  - keep `configs/local_profiles/tsp_seeded_swap_local_fast.json` as the budget-first path
  - the remaining caution is now explained in mechanism terms: the anti-case tail is likely tied to
    early seed lock-in and/or late refinement deficit, not just to a generic budget cut
  - protocol implication: anti-case suspicion still goes straight to `Q`, and the wording can now
    explain that the fast stack is the wrong tool when corridor-like rows need late cleanup
- ZDT1:
  - keep `configs/local_profiles/zdt1_diversity_injection_fast.json` as the exploratory path
  - the remaining safety concern is now framed as early-front-plateau / late-spread-collapse /
    refresh-timing mismatch, which is why `Q` still owns the final safety row
  - protocol implication: the current `F 3 exploratory / Q 8-10 final safety` row stays intact,
    but the note should explain *why* `Q` closes the final decision
- knapsack:
  - keep only the narrow repair-only note
- onemax:
  - keep the control row exactly as-is

## Target-Specific Hypothesis Probe Follow-Up

The protocol matrix still does not change structurally here. This pass only asks whether the
strongest pinned mechanism hypotheses survive a tiny same-budget probe:

```bash
python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_study
python scripts/run_local_sweep.py --study tsp_target_hypothesis_probe_confirm
python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_study
python scripts/run_local_sweep.py --study zdt1_target_hypothesis_probe_confirm
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_target_hypothesis_probe_confirm --zdt1-study-name zdt1_target_hypothesis_probe_confirm
```

Read the result like this:

- TSP:
  - keep `configs/local_profiles/tsp_seeded_swap_local_fast.json` as the budget-first path
  - the current best hypothesis is still `late_refinement_deficit`, but only in a weakened form:
    a small generation-up probe did not close anti-case `p90/p95`, and rescue-target mean also
    moved the wrong way
  - `seed_lockin_and_diversity_collapse` stays secondary rather than becoming the main story
  - protocol implication: anti-case suspicion still goes straight to `Q`, and the wording can now
    say that the fast stack may start well but still under-refine late corridor cleanup
- ZDT1:
  - keep `configs/local_profiles/zdt1_diversity_injection_fast.json` as the exploratory path
  - timing-only probes did not close the safety story cleanly: cooldown `2` cut fail rate but cost
    too much HV, while refresh `0.12` worsened spread / joint failures
  - the spread failure mechanism therefore stays explicitly unisolated, which is why `Q` still owns
    the final safety row
  - protocol implication: the current `F 3 exploratory / Q 8-10 final safety` row stays intact,
    and the note should explain that the remaining failure is broader than a simple timing tweak
- knapsack:
  - keep only the narrow repair-only note
- onemax:
  - keep the control row exactly as-is

## Population-Generation / Spread-vs-Joint Follow-Up

The protocol matrix still does not change structurally here either. This pass only asks whether the
current mechanism story actually moves under one same-budget tradeoff probe per target:

```bash
python scripts/run_local_sweep.py --study tsp_population_generation_probe_study
python scripts/run_local_sweep.py --study tsp_population_generation_probe_confirm
python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_study
python scripts/run_local_sweep.py --study zdt1_timing_vs_pg_probe_confirm
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_population_generation_probe_confirm --zdt1-study-name zdt1_timing_vs_pg_probe_confirm
```

Read the result like this:

- TSP:
  - keep `configs/local_profiles/tsp_seeded_swap_local_fast.json` as the budget-first path
  - the generation-up / population-down probe improved anti-case mean on the narrow stress slice,
    but it still made anti-case `p90/p95` worse in confirm, so it did not justify rewriting the
    fast default
  - `late_refinement_deficit` stays the best current explanation, but only in a weakened form;
    `seed_lockin_and_diversity_collapse` stays secondary because the seed-fraction-only probe was
    even less honest
  - protocol implication: anti-case/corridor suspicion still goes straight to `Q`, and rescue-only
    ambiguity is still the only place where the `5`-seed compare step earns its keep
- ZDT1:
  - keep `configs/local_profiles/zdt1_diversity_injection_fast.json` as the exploratory path
  - `probe_pg_pop41_gen88` was the most informative same-budget probe: it improved HV and reduced
    spread/joint failures on the fixed stress slice, but the evidence is still narrow enough that
    the protocol should not promote a new default from it yet
  - timing-only probes did not close the joint-failure story cleanly, so spread fail and joint
    safety fail should stay split in the registry
  - protocol implication: `F` remains good for exploratory HV reads, but final safety still belongs
    to `Q 8-10`
- knapsack:
  - keep only the narrow repair-only note
- onemax:
  - keep the control row exactly as-is

## TSP Tail Freeze + ZDT1 Spread Validation Follow-Up

The protocol matrix still stays structurally unchanged here. This pass closes one open question on
each side of the local budget-first defaults:

```bash
python scripts/run_local_sweep.py --study tsp_tail_freeze_recheck
python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_study
python scripts/run_local_sweep.py --study zdt1_spread_candidate_boundary_confirm
python scripts/run_local_sweep.py --study zdt1_joint_note_freeze_check
python scripts/run_local_sweep.py --study knapsack_stress_freeze_check
python scripts/run_local_sweep.py --study onemax_control_stress_freeze_check
python scripts/build_failure_hypotheses_registry.py --tsp-study-name tsp_tail_freeze_recheck --zdt1-spread-study-name zdt1_spread_candidate_boundary_confirm --zdt1-joint-study-name zdt1_joint_note_freeze_check
```

Read the result like this:

- TSP:
  - keep `configs/local_profiles/tsp_seeded_swap_local_fast.json` as the budget-first path
  - the remaining anti-case `p95/max` tail is now best read as a protocol limitation of the
    current fixed stack, not as an active same-budget contour-tuning target
  - protocol implication: anti-case / corridor suspicion and quality-sensitive finals still go
    straight to `Q 8-10`
  - wording implication: `F` remains a budget-first exploratory path, while `Q` owns the
    corridor-like cleanup cases that still need more reliable late refinement
- ZDT1 spread target:
  - keep `configs/local_profiles/zdt1_diversity_injection_fast.json` as the current fast default
    for now
  - `spread_pg_pop41_gen88` is the strongest spread-side candidate, and it improves the pinned
    spread-stress slice, but it still regresses normal / stable rows enough that it closes as
    `note_only_stress_slice`, not as a safe same-name replacement
  - protocol implication: keep it as a slice-conditioned regression reference, not as a promoted
    default
- ZDT1 joint target:
  - timing-only `cooldown2` stays out of promotion: it did not close the joint story cleanly and
    still carried HV-tail penalty
  - protocol implication: final safety still belongs to `Q 8-10`, and joint fail remains a
    monitor-only target rather than a fast-default rewrite
- knapsack:
  - keep only the narrow repair-only note
- onemax:
  - keep the control row exactly as-is

## Extreme-Tail Closeout / Split-Target Confirmation

The protocol matrix still does not change structurally here either. This pass only asks whether
the remaining TSP anti-case tail and the split ZDT1 spread/joint targets can be moved by one more
same-budget closeout pass:

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

Read the result like this:

- TSP:
  - keep `configs/local_profiles/tsp_seeded_swap_local_fast.json` as the budget-first path
  - same-budget contour probes can still move anti-case mean and sometimes `p95`, but they do not
    close anti-case `p95/max` honestly enough to replace the current default
  - protocol implication: anti-case/corridor suspicion still goes straight to `Q`, and the note
    can now say that the fast stack may start well but still miss the late cleanup tail
- ZDT1 spread:
  - keep `configs/local_profiles/zdt1_diversity_injection_fast.json` as the exploratory path
  - the spread slice now looks more population/generation-shaped than timing-shaped; the best
    same-budget PG probe reduced spread fail rate and spread tail without reopening HV tail
  - protocol implication: keep `F` for exploratory HV reads, but do not interpret that as final
    safety coverage
- ZDT1 joint:
  - keep `configs/local_profiles/zdt1_diversity_injection_fast.json` as the exploratory path
  - timing-only probes still move joint fail rate, but not cleanly enough once HV preservation
    stays in scope
  - protocol implication: final safety still belongs to `Q 8-10`, and the joint note should stay
    separate from the spread note
- knapsack:
  - keep only the narrow repair-only note
- onemax:
  - keep the control row exactly as-is

## Relationship To The Larger Local Guide

Use this guide when you already know the current local rules and want the shortest path from
"what am I trying to do?" to "which profile and how many seeds should I spend?".

Use [Local experiment guide](local_experiment_guide.md) when you want the wider experimental
history, plots, and study-manifest context that produced this matrix.

## Baseline Freeze

The protocol matrix is now also frozen as a local regression baseline.

- snapshot: `artifacts/local_baseline_snapshot.json`
- checker output: `artifacts/local_baseline_check.json`

Run the guard pack like this:

```bash
python scripts/run_local_sweep.py --study local_baseline_guard_tsp
python scripts/run_local_sweep.py --study local_baseline_guard_zdt1
python scripts/run_local_sweep.py --study local_baseline_guard_knapsack
python scripts/run_local_sweep.py --study local_baseline_guard_onemax
python scripts/check_local_baseline.py --write-snapshot
python scripts/check_local_baseline.py
```

Current frozen read:

- TSP:
  - `F` stays budget-first / exploratory only
  - anti-case / corridor suspicion still goes straight to `Q 8-10`
  - `tsp_fast_anti_case_tail` is now a frozen protocol limitation rather than an active contour target
- ZDT1:
  - `F` stays useful for exploratory HV-first work
  - final safety still belongs to `Q`
  - `spread_pg_pop41_gen88` stays note-only on the spread-stress slice
  - `zdt1_fast_joint_safety_fail` stays monitor-only, so do not read timing-only tweaks as a promoted fix
- knapsack:
  - broad default stays parked; keep only the narrow repair note
- onemax:
  - control only

## Candidate Admission

The protocol matrix above is frozen. New local ideas should come in as
candidate manifests, not as silent profile replacements.

Use the candidate workflow like this:

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

Decision read:

- TSP:
  - `F` is still budget-first / exploratory only
  - anti-case / corridor suspicion or quality-sensitive final still goes
    straight to `Q 8-10`
  - because `tsp_fast_anti_case_tail` is frozen as a protocol limitation, a
    same-budget contour candidate without a new mechanism hypothesis should
    close as `candidate_requires_new_mechanism_hypothesis` or a rejection label
- ZDT1:
  - `F` is still useful for exploratory / budget-first work
  - final safety still belongs to `Q`
  - `spread_pg_pop41_gen88` is still `note_only_stress_slice`, not a promoted
    replacement
  - `zdt1_fast_joint_safety_fail` is still `monitor_only`, so any candidate that
    reopens HV tail or joint safety should be rejected
- knapsack:
  - broad default promotion is still forbidden
  - `repair_only` can survive only as a narrow family-conditioned note
- onemax:
  - control drift only; no active candidate target

Passing a candidate guard does not rewrite `configs/local_profiles/*`, the
protocol matrix, or the baseline snapshot automatically.

Candidate backlog read:

- rejected:
  - do not rerun the same idea without new evidence
- note_only:
  - keep only as a slice-conditioned note
- monitor:
  - re-read only during a future stress refresh
- promising_needs_confirm:
  - confirm first; no change-request pack yet
- passed_local_guard:
  - may open a manual baseline change-request pack
- requires_new_mechanism:
  - do not keep nudging the same contour; reopen only with a new mechanism story

Change-control read:

- `scripts/summarize_local_candidates.py` rebuilds the candidate ledger
- `scripts/summarize_local_optimization_status.py` freezes the current cycle status, reopen criteria, and backlog closeout
- `scripts/build_local_baseline_change_request.py` drafts review artifacts only
- even a passing candidate does not auto-change the frozen baseline

## Cycle Closeout

Cycle 1 is now frozen as a local operating baseline plus a reopen-criteria pack.

Closeout commands:

```bash
python scripts/check_local_baseline.py
python scripts/summarize_local_candidates.py
python scripts/summarize_local_optimization_status.py
```

Current closeout read:

- TSP:
  - `F` remains budget-first / exploratory only
  - anti-case / corridor suspicion or quality-sensitive final still goes straight to `Q 8-10`
  - `tsp_fast_anti_case_tail` is frozen as a protocol limitation, so reopen only with a new mechanism hypothesis
- ZDT1:
  - `F` remains exploratory / budget-first only
  - final safety still belongs to `Q`
  - `spread_pg_pop41_gen88` stays note-only on the spread-stress slice
  - `zdt1_fast_joint_safety_fail` stays monitor-only rather than a rewrite trigger
- knapsack:
  - keep only the narrow repair note
  - reopen only with family-conditioned subset-sum / tight-capacity evidence
- onemax:
  - keep the control row exactly as-is
  - reopen only if control drift appears

When not to reopen:

- do not rerun the old TSP population/generation contour family
- do not treat spread-stress-only ZDT1 gain as a default replacement
- do not reopen broad knapsack default discovery
- do not turn onemax into an active optimization branch

See `docs/local_reopen_criteria.md` for the problem-by-problem reopen matrix.
