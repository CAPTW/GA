# NSGA-II Candidate O Opt-in Usage Guide

## 1. Status

- `candidate_id`: `candidate_o_spread_preserving_variation_light`
- `approval_status`: `approved_restricted_opt_in`
- `approval_type`: `restricted_opt_in_experimental_profile`
- `default_changed=false`
- current allowed scope: explicit opt-in experimental profile, local research benchmark, ZDT-family exploratory comparison, candidate comparison harness, and future review evidence generation

## 2. What This Candidate Changes

`candidate_o_spread_preserving_variation_light` is a follow-up to
`candidate_n_low_g_tail_mutation_light`.

- base candidate: `candidate_n_low_g_tail_mutation_light`
- mechanism: `spread_preserving_variation_light`
- intent: preserve more objective-space spread without direct objective-space repair
- trade-off: spread improvement can come with some low-g component dilution
- execution mode: explicit opt-in only

This candidate is not part of the default NSGA-II path.

## 3. When to Use

- ZDT-family exploratory benchmark work
- spread-preserving variation comparison
- `candidate_o` vs `candidate_n` comparison
- local research experiment
- evidence generation for future restricted-profile review

## 4. When Not to Use

- default NSGA-II replacement
- production optimizer
- `pymoo` replacement claim
- general MOEA superiority claim
- constrained optimization
- industrial optimization
- single-seed performance claim
- broader non-ZDT approval claim or scope expansion without a separate review
- WFG-based scope expansion or broader non-ZDT approval claim without a separate review

## 5. Required Checks Before Running

- use an artifact suffix
- run the fairness checker
- run a default drift audit before important comparisons
- confirm candidate isolation
- run the local baseline check
- use multiple seeds
- include `pymoo` comparison if claiming any external relevance

## 6. Known Trade-offs

- repeated spread improvement versus `candidate_n` on ZDT1 stress
- some low-g component trade-off versus `candidate_n`
- persistent `pymoo` spread and runtime gap
- DTLZ2/DTLZ3 smoke adds safety evidence, but broader non-ZDT scope is still not approved
- WFG1/WFG2 smoke adds future review evidence only, and the reference-front, HV, IGD, and adapter limitations still narrow the interpretation

## 7. Non-ZDT Review Status

- DTLZ2/DTLZ3 small smoke completed as future review evidence only
- authoritative artifact: `artifacts/nsga2_candidate_o_non_zdt_smoke_report_dtlz_smoke1_rerun1.md`
- review note: `artifacts/hypotheses/nsga2_candidate_o_dtlz_smoke_review_note.md`
- decision: restricted opt-in scope maintained, non-ZDT smoke positive
- scope change: none
- broader non-ZDT review decision artifact: `artifacts/hypotheses/nsga2_candidate_o_broader_non_zdt_review_decision.md`
- WFG1/WFG2 small smoke completed as future review evidence only
- WFG authoritative artifact: `artifacts/nsga2_candidate_o_wfg_smoke_report_wfg_smoke1.md`
- WFG decision: restricted opt-in scope maintained, WFG smoke positive
- WFG status: completed_positive
- WFG planning artifact: `artifacts/hypotheses/nsga2_candidate_o_wfg_smoke_plan.md`
- important limit: DTLZ/WFG smoke evidence does not broaden approval scope beyond the current restricted opt-in profile
- broader non-ZDT scope still requires a separate review
- broader evidence review artifact: `artifacts/reviews/nsga2_candidate_o_broader_non_zdt_evidence_review.md`
- scope decision note: `artifacts/reviews/nsga2_candidate_o_scope_decision_note.md`
- scope change: none
- pymoo gap remains visible across spread/count/runtime-oriented readings and must stay visible in any future review package

## 8. Evidence Required Before Broader Review

Before any CR discussion, default discussion, or broader external claim:

- complete a non-ZDT validation review
- keep fairness fail at zero
- keep default drift at NO DRIFT
- preserve candidate isolation
- avoid non-finite objective failures
- show no catastrophic regression outside the ZDT family
- package DTLZ/WFG evidence together with explicit metric/reference limitations instead of citing smoke results in isolation
- keep WFG-specific metric and reference limitations explicit in any later review

## 9. Example Commands

python scripts/validate_nsga2_spread_preserving_phase2.py --problems zdt1,zdt2,zdt3 --zdt1-seeds 30 --other-seeds 10 --budget 760 --artifact-suffix candidate_o_review_example

python scripts/validate_nsga2_candidate_o_non_zdt_smoke.py --problems dtlz2,dtlz3 --seeds 10 --budget 760 --artifact-suffix dtlz_smoke1_rerun1

python scripts/validate_nsga2_candidate_o_wfg_smoke.py --problems wfg1,wfg2 --seeds 5 --budget 760 --artifact-suffix wfg_smoke1

python -m pytest tests/test_nsga2_candidate_isolation.py tests/test_parameter_fairness.py tests/test_fairness_runner_integration.py -q

python scripts/check_local_baseline.py --output-dir artifacts/candidate_o_usage_guard

## 10. Exit Criteria

- large regression in future ZDT, DTLZ, or WFG validation
- any fairness fail
- any default contamination
- widening `pymoo` spread gap
- artifact reproducibility failure
- misuse as a default optimizer
