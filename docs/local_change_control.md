# Local Change Control

The frozen local baseline is not changed by candidate reports.

Candidate reports answer:

- did a candidate improve one pinned target?
- did it regress stable or normal slices?
- does it stay note-only, monitor-only, or rejected?

Change control answers a different question:

- is this candidate strong enough to justify a manual baseline-change review?

## Lifecycle Gate

Only these states may open a normal change-request pack:

- `passed_local_guard`
- `ready_for_change_request`

Everything else is blocked unless `--force-draft` is used.

## Commands

```bash
python scripts/check_local_baseline.py
python scripts/summarize_local_candidates.py
python scripts/summarize_local_optimization_status.py
python scripts/build_local_baseline_change_request.py --candidate-report outputs/local_candidates/.../candidate_report.json
python scripts/build_local_baseline_change_request.py --candidate-report outputs/local_candidates/.../candidate_report.json --force-draft
```

## Output Pack

Each draft pack writes:

- `change_request.json`
- `change_request.md`
- `baseline_diff_summary.md`
- `required_followup_checks.md`

Those files are review artifacts only. They do not:

- rewrite `configs/local_profiles/*`
- rewrite the protocol matrix
- rewrite the target registry
- refresh the baseline snapshot

## Expected Manual Follow-Up

If a reviewer ever decides to promote a candidate, the manual follow-up should
still include:

- rerun the relevant baseline guard study
- rerun the candidate comparison
- update the docs that describe the operating rule
- refresh the baseline snapshot explicitly
- rerun `python scripts/check_local_baseline.py`

## Problem Read

- TSP:
  - because `tsp_fast_anti_case_tail` is frozen as a protocol limitation, most
    same-budget contour candidates should never reach change-request without a
    new mechanism hypothesis
- ZDT1:
  - spread candidates may draft a pack only if they generalize beyond the
    spread-stress slice
  - joint timing-only candidates should stay blocked unless they stop paying the
    HV/safety penalty
- knapsack:
  - broad default promotion is intentionally disallowed
  - repair-only remains narrow-note only
- onemax:
  - no active target means no active change-request path

## Closeout Status

Cycle 1 is currently closed with no baseline-change-ready candidate.

- TSP:
  - anti-case tail is a frozen protocol limitation
  - same-budget contour reruns stay blocked unless a new mechanism hypothesis appears
- ZDT1:
  - spread candidate stays note-only
  - joint target stays monitor-only with final safety still on `Q`
- knapsack:
  - repair-only stays a narrow note, not a baseline-change path
- onemax:
  - control only

Use `python scripts/summarize_local_optimization_status.py` to regenerate the
closeout snapshot, reopen criteria, and backlog closeout before drafting any
manual change request.
