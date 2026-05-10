# Local Reopen Criteria

Local optimization cycle 1 is frozen. Reopen work only through a candidate manifest plus a baseline check.

| problem | current_state | reopen_trigger | required_candidate_label | rejection_shortcut | forbidden_reopen_pattern |
| --- | --- | --- | --- | --- | --- |
| tsp | Frozen budget-first/exploratory split; anti-case/corridor suspicion and quality-sensitive finals still go straight to Q 8-10. | A genuinely new mechanism hypothesis exists.; The candidate targets anti-case p95/max directly instead of repeating the old contour story. | candidate_promising_needs_confirm or stronger; baseline change still requires candidate_passes_local_guard. | Reject early when the idea is another same-budget PG contour rerun without a new mechanism story, or when anti-case p95/max is not improved. | Re-running the already failed population/generation contour family.; Using a stress-slice-only gain to argue for a baseline replacement. |
| zdt1 | Frozen exploratory/budget-first split; spread candidate stays note-only and final safety still belongs to Q. | A spread candidate generalizes beyond the spread-stress slice to stable/normal rows.; A new mechanism reduces joint safety failures without reopening HV tail loss. | candidate_promising_needs_confirm or stronger; replacement review still requires candidate_passes_local_guard. | Reject when the candidate wins only on the spread-stress slice or when HV/joint/Pareto non-regression breaks on stable or normal rows. | Claiming spread-stress-only gain as a fast-default replacement.; Timing-only joint candidates that pay clear HV tail penalty. |
| knapsack | No broad default; repair_only remains a narrow family-conditioned note. | New evidence is explicitly limited to subset-sum-like or tight-capacity-like families.; The candidate shows consistent gain over none/greedy/repair_only on that family. | note_only_stress_slice at most; broad default promotion stays forbidden. | Reject any candidate that implies a broad default or loses to greedy on the narrow family. | Generalizing weakly correlated ties into a broad rule.; Reopening broad default search for knapsack. |
| onemax | Frozen control-only branch on none; no active optimization target. | Baseline control drift appears.; Instrumentation regression needs a control recheck. | monitor_only at most unless there is a clear control simplification. | Reject anything more complex than the current none-control path without obvious control gain. | Reintroducing adaptive search into the control problem.; Treating Onemax as an active optimization target without drift evidence. |

## Problem Details

### TSP

- current state: Frozen budget-first/exploratory split; anti-case/corridor suspicion and quality-sensitive finals still go straight to Q 8-10.
- reopen trigger: A genuinely new mechanism hypothesis exists.; The candidate targets anti-case p95/max directly instead of repeating the old contour story.
- required evidence: Anti-case p95/max improvement versus the frozen current fast baseline.; Rescue-target mean non-regression.; Same or lower configured budget.; Baseline guard PASS before candidate comparison.
- required candidate label: candidate_promising_needs_confirm or stronger; baseline change still requires candidate_passes_local_guard.
- minimum guard checks: python scripts/check_local_baseline.py; python scripts/run_local_candidate.py --candidate <manifest>; python scripts/summarize_local_candidates.py
- rejection shortcut: Reject early when the idea is another same-budget PG contour rerun without a new mechanism story, or when anti-case p95/max is not improved.
- forbidden reopen pattern: Re-running the already failed population/generation contour family.; Using a stress-slice-only gain to argue for a baseline replacement.

### ZDT1

- current state: Frozen exploratory/budget-first split; spread candidate stays note-only and final safety still belongs to Q.
- reopen trigger: A spread candidate generalizes beyond the spread-stress slice to stable/normal rows.; A new mechanism reduces joint safety failures without reopening HV tail loss.
- required evidence: HV mean/tail non-regression versus the frozen fast baseline.; Spread fail reduction that survives stable and normal slices.; Joint safety non-regression.; Pareto-ratio non-regression.; Baseline guard PASS before candidate comparison.
- required candidate label: candidate_promising_needs_confirm or stronger; replacement review still requires candidate_passes_local_guard.
- minimum guard checks: python scripts/check_local_baseline.py; python scripts/run_local_candidate.py --candidate <manifest>; python scripts/summarize_local_candidates.py
- rejection shortcut: Reject when the candidate wins only on the spread-stress slice or when HV/joint/Pareto non-regression breaks on stable or normal rows.
- forbidden reopen pattern: Claiming spread-stress-only gain as a fast-default replacement.; Timing-only joint candidates that pay clear HV tail penalty.

### KNAPSACK

- current state: No broad default; repair_only remains a narrow family-conditioned note.
- reopen trigger: New evidence is explicitly limited to subset-sum-like or tight-capacity-like families.; The candidate shows consistent gain over none/greedy/repair_only on that family.
- required evidence: Family-conditioned feasible-quality improvement.; No broad-default claim.; Baseline guard PASS before candidate comparison.
- required candidate label: note_only_stress_slice at most; broad default promotion stays forbidden.
- minimum guard checks: python scripts/check_local_baseline.py; python scripts/run_local_candidate.py --candidate <manifest>
- rejection shortcut: Reject any candidate that implies a broad default or loses to greedy on the narrow family.
- forbidden reopen pattern: Generalizing weakly correlated ties into a broad rule.; Reopening broad default search for knapsack.

### ONEMAX

- current state: Frozen control-only branch on none; no active optimization target.
- reopen trigger: Baseline control drift appears.; Instrumentation regression needs a control recheck.
- required evidence: Control drift or instrumentation regression evidence.; Baseline guard PASS before candidate comparison.
- required candidate label: monitor_only at most unless there is a clear control simplification.
- minimum guard checks: python scripts/check_local_baseline.py
- rejection shortcut: Reject anything more complex than the current none-control path without obvious control gain.
- forbidden reopen pattern: Reintroducing adaptive search into the control problem.; Treating Onemax as an active optimization target without drift evidence.
