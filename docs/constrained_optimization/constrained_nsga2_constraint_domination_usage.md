# Constrained NSGA-II Constraint-Domination Usage Guide

## 1. Status

- `path_id`: `constrained_nsga2_constraint_domination`
- `approval_status`: `approved_restricted_experimental_opt_in_with_limitations`
- `approval_type`: `restricted_experimental_opt_in_path`
- `default_changed=false`
- `scope_change=none`
- current allowed scope: explicit constrained NSGA-II experimental opt-in path, `constrained_zdt_box_toy` and `constrained_dtlz_box_toy` research smoke/stress and external comparison evidence review, constraint-domination protocol validation, constrained MO artifact/fairness validation, future review evidence generation
- current broader review package: `artifacts/reviews/constrained_nsga2_broader_evidence_review_update.md`, `artifacts/reviews/constrained_nsga2_scope_decision_update.md`, `artifacts/reviews/constrained_nsga2_evidence_package.md`
- current external comparison review package: `artifacts/reviews/constrained_nsga2_external_comparison_review.md`, `artifacts/reviews/constrained_nsga2_external_parity_decision_note.md`
- current external stress review package: `artifacts/reviews/constrained_nsga2_external_stress_review.md`, `artifacts/reviews/constrained_nsga2_external_parity_decision_update.md`, `artifacts/reviews/constrained_nsga2_external_stress_scope_decision.md`
- current validation closure package: `artifacts/reviews/constrained_nsga2_evidence_matrix.md`, `artifacts/reviews/constrained_nsga2_active_validation_pause_decision.md`, `artifacts/hypotheses/constrained_nsga2_future_roadmap.md`, `artifacts/constrained_nsga2_validation_closure_report.md`
- active validation status: paused; restricted experimental opt-in path preserved

## 2. What This Path Does

`constrained_nsga2_constraint_domination` is a separate constrained multi-objective
path centered on `run_constrained_nsga2`.

- it keeps constrained NSGA-II separate from the default `run_nsga2` path
- it uses constraint-domination ordering: feasible beats infeasible, feasible-feasible uses Pareto plus crowding, infeasible-infeasible uses violation ordering
- it has smoke evidence on `constrained_zdt_box_toy` and `constrained_dtlz_box_toy`
- it has a first pymoo external comparison smoke on `constrained_zdt_box_toy` and `constrained_dtlz_box_toy`
- it has external stress evidence on `constrained_zdt_box_toy` and `constrained_dtlz_box_toy` with seeds 20 and budgets 760/1500
- it has a validation closure decision that pauses active validation while preserving the restricted opt-in path
- it is explicit opt-in only
- it is different from the default NSGA-II path

## 3. What This Path Does Not Do

- it is not a default NSGA-II replacement
- it is not a product optimizer
- it is not penalty-based constraint handling
- it is not repair-based constraint handling
- it is not an external constrained parity claim
- it is not constrained MO full benchmark support
- it is not a general constrained MOEA claim

## 4. When to Use

- constrained_zdt_box_toy research smoke review
- constrained_dtlz_box_toy research smoke review
- constrained_zdt_box_toy / constrained_dtlz_box_toy external comparison review, with no parity claim
- constrained_zdt_box_toy / constrained_dtlz_box_toy external stress evidence review, with no parity claim
- closure review or future roadmap planning under `scope_change=none`
- constraint-domination protocol experiments
- constrained MO artifact and fairness validation
- future constrained MO review evidence generation

## 5. When Not to Use

- production optimization
- industrial constrained MOEA
- default algorithm replacement
- broad constrained MOEA claim
- toy smoke superiority claim
- constrained DTLZ/ZDT general claim
- default `run_nsga2` constrained support claim
- claim that constrained NSGA-II is better than pymoo
- claim that external comparison completion means external parity

## 6. Required Checks Before Running

- use an artifact suffix
- use multiple seeds for meaningful comparisons
- review the constrained fairness summary
- confirm `actual_evaluations` matches the requested budget
- keep finite constraint validation enabled
- run the local baseline check before sharing results
- run NSGA-II regression tests before important comparisons

## 7. Example Command

python scripts/validate_constrained_nsga2_smoke.py --dimension 6 --seeds 5 --budget 760 --artifact-suffix constrained_nsga2_review_example

## 8. Current Evidence

| benchmark | evidence status | constrained NSGA-II signal | limitation |
|---|---|---|---|
| `constrained_zdt_box_toy` | smoke completed | feasible_rate 1.0, mean_total_violation 0.0, exact 760 evaluation accounting, fairness fail 0 | feasible-only HV/reference-distance were mixed against the random archive baseline |
| `constrained_dtlz_box_toy` | second toy smoke completed | feasible_rate 1.0, mean_total_violation 0.0, exact 760 evaluation accounting, fairness fail 0 | still toy-only evidence and not stress or external parity evidence |
| `constrained_zdt_box_toy` + `constrained_dtlz_box_toy` | pymoo external comparison smoke completed | constrained NSGA-II and pymoo tied on feasible_rate and mean_total_violation in this smoke; exact evaluation accounting held | external parity not established; pymoo had stronger feasible-only quality and runtime signal in this smoke; operator family warning applies |
| `constrained_zdt_box_toy` + `constrained_dtlz_box_toy` | pymoo external stress completed | seeds 20, budgets 760/1500; fairness fail 0; exact evaluation accounting held; feasibility/violation tied vs pymoo | external parity not established; pymoo feasible-only HV/reference-distance and runtime were stronger overall; spacing mixed |
| validation closure | active validation paused | restricted opt-in evidence is preserved and future work is roadmapped | no scope expansion, no default replacement, no external parity claim |

## 9. Known Trade-offs / Limitations

- evidence is limited to two project-local toy benchmarks: `constrained_zdt_box_toy` and `constrained_dtlz_box_toy`
- feasible-only HV/reference-distance signal is mixed and must be reported as a limitation
- feasible-only HV/reference-distance/spacing must be computed only on feasible fronts and treated as unavailable when no feasible front exists
- runtime can be materially heavier than the random archive baseline and must be reported as a trade-off
- pymoo external comparison smoke is completed, but external parity is not established
- pymoo external stress is completed, but external parity is not established
- pymoo feasible-only quality and runtime were stronger than constrained NSGA-II in the smoke and stress; this must not be hidden or converted into a superiority claim
- external operator family difference is a warning and must remain visible in external comparison interpretation
- DEAP remains a secondary hold and is not implemented
- penalty and repair handling are not implemented
- there is no default NSGA-II constrained support
- active validation is paused; future workstreams require separate approval and must follow the roadmap in `artifacts/hypotheses/constrained_nsga2_future_roadmap.md`

## 10. Exit / Re-review Criteria

- second constrained MO benchmark failure
- any fairness fail
- any default path contamination
- any `actual_evaluations` mismatch
- any non-finite regression
- misuse as a default optimizer
- product or general constrained optimizer claim misuse
- external parity or pymoo-superiority misuse
