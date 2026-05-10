# Local Candidate Workflow

The local baseline is already frozen. New local ideas should therefore enter as
candidate manifests, not as silent profile replacements.

This workflow does not auto-promote anything. A candidate can only produce:

- a candidate report
- a ledger entry
- an optional manual change-request draft

## Baseline First

Always check the frozen baseline before candidate comparison:

```bash
python scripts/check_local_baseline.py
```

If the snapshot no longer matches the current profiles, protocol matrix, target
registry, or candidate rules, the candidate report should stop with
`baseline_drift_detected`.

## Candidate Manifest

Candidate manifests live under `configs/local_candidates/`.

- schema: `configs/local_candidates/candidate_schema.json`
- TSP example: `configs/local_candidates/example_tsp_candidate.json`
- ZDT1 example: `configs/local_candidates/example_zdt1_candidate.json`
- knapsack example: `configs/local_candidates/example_knapsack_candidate.json`

Each manifest ties a candidate to:

- one frozen baseline snapshot
- one baseline guard study
- one pinned target and hypothesis
- one same-budget or narrow-note budget policy
- one decision policy

The manifest describes a validation target only. It never replaces a profile by
itself.

## Runner

Use the runner like this:

```bash
python scripts/run_local_candidate.py --candidate configs/local_candidates/example_zdt1_candidate.json
python scripts/run_local_candidate.py --candidate configs/local_candidates/example_tsp_candidate.json --no-execute
python scripts/run_local_candidate.py --candidate configs/local_candidates/example_knapsack_candidate.json --use-existing-output
```

Outputs are written to:

- `outputs/local_candidates/<timestamp>_<candidate_id>/candidate_report.json`
- `outputs/local_candidates/<timestamp>_<candidate_id>/candidate_report.md`
- `outputs/local_candidates/<timestamp>_<candidate_id>/candidate_summary.csv`
- `outputs/local_candidates/<timestamp>_<candidate_id>/candidate_vs_baseline.csv`

## Decision Labels

Every candidate report must emit exactly one of:

- `reject_regression`
- `reject_no_material_gain`
- `note_only_stress_slice`
- `monitor_only`
- `candidate_promising_needs_confirm`
- `candidate_passes_local_guard`
- `candidate_requires_new_mechanism_hypothesis`
- `baseline_drift_detected`
- `intentional_baseline_change_required`

## Lifecycle States

Decision labels are bookkeeping outputs. Lifecycle states are backlog status.
They are related, but not identical.

| decision label | lifecycle state | meaning |
| --- | --- | --- |
| `reject_regression` | `rejected` | candidate regressed the frozen baseline |
| `reject_no_material_gain` | `rejected` | candidate did not improve enough to justify more work |
| `note_only_stress_slice` | `note_only` | candidate is useful only as a narrow stress-slice note |
| `monitor_only` | `monitor` | candidate is not promoted, but may be re-read in a future refresh |
| `candidate_promising_needs_confirm` | `promising_needs_confirm` | candidate is interesting but still needs confirm evidence |
| `candidate_passes_local_guard` | `passed_local_guard` | candidate cleared the current local gate and may earn a change-request draft |
| `candidate_requires_new_mechanism_hypothesis` | `requires_new_mechanism` | do not rerun the same micro-tuning path without a new mechanism story |
| `baseline_drift_detected` | `blocked_by_baseline_drift` | fix the baseline mismatch first |
| `intentional_baseline_change_required` | `ready_for_change_request` | candidate is ready for a manual baseline change request |

Lifecycle state does not update the baseline. It only changes how the backlog
should be read.

## Candidate Ledger

Candidate reports are now gathered into one ledger:

```bash
python scripts/summarize_local_candidates.py
```

Artifacts:

- `artifacts/local_candidate_ledger.json`
- `artifacts/local_candidate_ledger.csv`
- `artifacts/local_candidate_ledger.md`
- `artifacts/local_candidate_summary.json`
- `artifacts/local_candidate_summary.md`

The ledger is the machine-readable backlog. It shows:

- which target and hypothesis each candidate touched
- which decision label it received
- which lifecycle state it now lives in
- whether it can affect the baseline
- what the next action is

## Baseline Change-Request Pack

Passing a candidate guard still does not rewrite the baseline automatically.
Instead, baseline-change follow-up goes through a manual draft pack:

```bash
python scripts/build_local_baseline_change_request.py --candidate-report outputs/local_candidates/.../candidate_report.json
python scripts/build_local_baseline_change_request.py --candidate-report outputs/local_candidates/.../candidate_report.json --force-draft
```

Outputs are written to:

- `outputs/local_change_requests/<timestamp>_<candidate_id>/change_request.json`
- `outputs/local_change_requests/<timestamp>_<candidate_id>/change_request.md`
- `outputs/local_change_requests/<timestamp>_<candidate_id>/baseline_diff_summary.md`
- `outputs/local_change_requests/<timestamp>_<candidate_id>/required_followup_checks.md`

Normal admission:

- only `candidate_passes_local_guard`
- only `intentional_baseline_change_required`
- or `ready_for_change_request`

Forced draft:

- `--force-draft` can draft a pack for note-only or monitor candidates
- forced drafts are explicitly not approvals

## Baseline Drift vs Candidate Improvement

- baseline drift:
  - snapshot hash mismatch on profiles, protocol matrix, target statuses, or candidate rules
  - candidate comparison should stop first
- candidate improvement:
  - baseline snapshot still matches
  - candidate output improves one target while keeping required non-regression checks
- intentional baseline change:
  - candidate passed the local guard, but profile replacement still requires a
    conscious snapshot/docs review

Passing a candidate guard does not update `configs/local_profiles/*`, the
protocol matrix, or the baseline snapshot automatically.

## Candidate Backlog Policy

- rejected:
  - do not rerun the same idea
  - reopen only with new stress evidence or a new mechanism hypothesis
- note_only:
  - do not promote to a broad default
  - keep only as a documented slice-conditioned note
- monitor:
  - baseline stays unchanged
  - recheck only during a later stress refresh or if evidence sharpens
- promising_needs_confirm:
  - confirm evidence is required
  - do not open a baseline change request yet
- passed_local_guard:
  - a change-request pack may be drafted
  - still no automatic profile replacement
- blocked_by_baseline_drift:
  - fix baseline drift before reading the candidate
- requires_new_mechanism:
  - do not keep nudging the same micro-tuning contour
  - reopen only with a genuinely new mechanism hypothesis

## Problem-Specific Admission Rules

| problem | current baseline profile(s) | frozen decision | allowed candidate type | promotion threshold | note-only threshold | rejection condition |
| --- | --- | --- | --- | --- | --- | --- |
| tsp | `Q = configs/local_profiles/tsp_seeded_swap_local.json`, `F = configs/local_profiles/tsp_seeded_swap_local_fast.json` | `F` stays budget-first / exploratory and anti-case tail stays frozen as protocol limitation | only candidates with a genuinely new mechanism hypothesis | must lower anti-case `p95/max` without materially worsening rescue-target mean or budget | stress-slice-only gain can be recorded, but it does not reopen the default | reject if anti-case `p95/max` worsens, rescue-target mean worsens, or no new mechanism hypothesis exists |
| zdt1 | `Q = configs/local_profiles/zdt1_diversity_injection.json`, `F = configs/local_profiles/zdt1_diversity_injection_fast.json` | `F` stays exploratory / budget-first and final safety stays on `Q` | same-budget spread or joint candidates tied to a pinned target | must improve spread/joint target while preserving HV mean/p90, pareto, and stable/normal slices | stress-slice-only gain closes as `note_only_stress_slice` | reject if stable/normal slices regress, HV tail worsens materially, pareto drops, or joint fail worsens |
| knapsack | `default = greedy_local_search`, `repair note = configs/local_profiles/knapsack_repair_local_experimental.json` | no broad default; repair note stays narrow | family-conditioned repair-note checks only | broad default promotion is forbidden | keep as note-only when the narrow family wins without broad claims | reject if the candidate implies a broad default or loses the narrow family slice |
| onemax | `none` control | no active target | control drift checks only | no active promotion path | monitor-only | reject anything more complex than the frozen control without a clear control gain |

## Cycle Closeout

The first local optimization cycle is now closed:

- the frozen baseline remains the comparison anchor
- no candidate in the current ledger is ready to change the baseline
- future work must start with `python scripts/check_local_baseline.py`
- future work must still enter as candidate manifests

Closeout commands:

```bash
python scripts/check_local_baseline.py
python scripts/summarize_local_candidates.py
python scripts/summarize_local_optimization_status.py
```

Closeout read:

- TSP:
  - `example_tsp_pg_contour_rejected` closes as `requires_new_mechanism`
  - do not reopen without a new mechanism hypothesis that directly targets anti-case `p95/max`
- ZDT1:
  - `example_zdt1_spread_candidate_note_only` stays `note_only`
  - do not treat a spread-stress-only win as baseline-replacement evidence
- knapsack:
  - `example_knapsack_repair_note` stays `note_only`
  - keep it narrow and family-conditioned
- onemax:
  - no active candidate path

Reopen criteria now live in `docs/local_reopen_criteria.md`.
