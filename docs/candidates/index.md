# Candidate Profiles

## Purpose

This document lists the current experimental candidate profiles and where to
find their evidence.

- Candidate profiles are different from the default NSGA-II path.
- Every candidate here must be selected explicitly.
- Approval in this document never means default promotion.

## Candidate Status Summary

| candidate | status | allowed use | disallowed use | guide |
| --- | --- | --- | --- | --- |
| candidate_d_uniform_crossover | Approved for opt-in experimental profile | explicit opt-in experimental profile; local benchmark comparison | default replacement; productization; general MOEA superiority claim | status matrix only |
| candidate_j_h_lite_retry2 | Approved for opt-in experimental profile | explicit opt-in experimental profile; local benchmark research; candidate comparison harness | default replacement; productization; general MOEA superiority claim | [usage guide](nsga2_candidate_j_opt_in_usage.md) |
| candidate_n_low_g_tail_mutation_light | Phase 1 passed with trade-offs | Phase 1 evidence reference only; low-g operator-quality hypothesis analysis; future design input | default replacement; approved opt-in profile; change request; Phase 2 execution without new approval | [Phase 1 closure](../../artifacts/nsga2_candidate_n_phase1_closure_report.md) |
| candidate_o_spread_preserving_variation_light | Approved for restricted opt-in experimental profile | explicit opt-in experimental profile; local research benchmark; ZDT-family exploratory comparison; candidate comparison harness; future review evidence generation | default replacement; change request candidate; production optimizer; general MOEA superiority claim; `pymoo` replacement claim | [usage guide](nsga2_candidate_o_opt_in_usage.md) |

## Hold Candidates

| candidate | status | note |
| --- | --- | --- |
| candidate_l_sparse_parent_bias_light | Hold for more evidence | keep as hypothesis input only |
| candidate_m_boundary_preservation_light | Hold for more evidence | keep as hypothesis input only |

## Candidate O Summary

- `candidate_o_spread_preserving_variation_light` is **Approved for restricted opt-in experimental profile**.
- Its approved scope is limited to ZDT-family exploratory work and local research benchmarks.
- The strongest evidence is repeated spread improvement versus `candidate_n` on ZDT1 stress.
- DTLZ2/DTLZ3 small smoke completed without a scope downgrade, but the result is future review evidence only and does not broaden approval scope.
- The DTLZ smoke note keeps `scope_change = none` and is only a broader non-ZDT review input.
- WFG1/WFG2 small smoke completed as future review evidence only, and the decision stayed `Restricted opt-in scope maintained, WFG smoke positive`.
- WFG evidence does not broaden candidate_o approval scope because reference-front, HV, IGD, and adapter limitations make the interpretation narrower than the ZDT slices.
- broader non-ZDT evidence has now been packaged, but the decision still keeps `scope_change = none`.
- A meaningful `pymoo` spread gap still remains on occupied bins, spacing, nondominated count, segment-0 allocation, and runtime.
- Low-g component trade-offs versus `candidate_n` still exist and must stay visible in reports.
- It is not approved as a default replacement.
- It is not CR approved.
- It is not product-ready.

## Required Checks Before Running Any Candidate

- use an artifact suffix
- run the fairness checker
- run a default drift audit before important comparisons
- confirm candidate isolation
- run the local baseline check
- use multiple seeds
- include `pymoo` comparison if claiming external relevance

## Links

- [candidate_o usage guide](nsga2_candidate_o_opt_in_usage.md)
- [candidate_o opt-in review](../../artifacts/reviews/nsga2_candidate_o_opt_in_review.md)
- [candidate_o Phase 2 report](../../artifacts/nsga2_spread_preserving_phase2_report_candidate_o_phase2_rerun1.md)
- [candidate_o non-ZDT smoke report](../../artifacts/nsga2_candidate_o_non_zdt_smoke_report_dtlz_smoke1_rerun1.md)
- [candidate_o DTLZ smoke review note](../../artifacts/hypotheses/nsga2_candidate_o_dtlz_smoke_review_note.md)
- [candidate_o WFG smoke report](../../artifacts/nsga2_candidate_o_wfg_smoke_report_wfg_smoke1.md)
- [candidate_o WFG smoke plan](../../artifacts/hypotheses/nsga2_candidate_o_wfg_smoke_plan.md)
- [candidate_o broader non-ZDT evidence review](../../artifacts/reviews/nsga2_candidate_o_broader_non_zdt_evidence_review.md)
- [candidate_o scope decision note](../../artifacts/reviews/nsga2_candidate_o_scope_decision_note.md)
- [candidate status matrix](../../artifacts/nsga2_candidate_status_matrix.md)
- [operator-quality backlog](../../artifacts/hypotheses/nsga2_operator_quality_backlog.json)
