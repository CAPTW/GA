# Constrained Optimization Experimental Paths

This index tracks constrained experimental paths that are separate from the
default GA and NSGA-II implementations.

- Every path here must be selected explicitly.
- Approval in this document means restricted experimental opt-in only.
- Nothing here is a default replacement or a product readiness signal.

| path | status | allowed use | disallowed use | guide |
| --- | --- | --- | --- | --- |
| constrained_ga_feasibility_first | Approved for restricted experimental opt-in path; broader toy evidence added; scope_change=none | explicit constrained single-objective experimental opt-in path; constrained_sphere, constrained_box_quadratic, and constrained_equality_plane_quadratic local research smoke/stress evidence as restricted evidence only; feasibility-first protocol experiments; constrained artifact, fairness, and trace validation; future review evidence generation | default GA replacement; NSGA-II constrained support claim; product or industrial constrained optimization; penalty or repair support claim; constrained multi-objective support claim; general constrained optimizer superiority claim | [usage guide](constrained_ga_feasibility_first_usage.md) |
| constrained_nsga2_constraint_domination | Approved for restricted experimental opt-in path with broader constrained MO and pymoo external stress evidence added; active validation paused; opt-in only; scope_change=none; external_parity_established=false | explicit constrained NSGA-II experimental opt-in path; constrained_zdt_box_toy and constrained_dtlz_box_toy toy research smoke/stress and external comparison evidence review; constraint-domination protocol validation; constrained MO artifact and fairness validation; future review evidence generation under separate approval | default NSGA-II replacement; default NSGA-II constrained support claim; product or industrial constrained MOEA; penalty or repair support claim; broad constrained MOEA superiority claim; external parity claim; pymoo superiority claim; full benchmark generalization | [usage guide](constrained_nsga2_constraint_domination_usage.md) |

## Links

- [opt-in review](../../artifacts/reviews/constrained_ga_feasibility_first_opt_in_review.md)
- [opt-in review update](../../artifacts/reviews/constrained_ga_feasibility_first_opt_in_review_update.md)
- [broader evidence review](../../artifacts/reviews/constrained_ga_broader_evidence_review.md)
- [broader evidence review update](../../artifacts/reviews/constrained_ga_broader_evidence_review_update.md)
- [scope decision note](../../artifacts/reviews/constrained_ga_scope_decision_note.md)
- [scope decision update](../../artifacts/reviews/constrained_ga_scope_decision_update.md)
- [constrained NSGA-II opt-in review](../../artifacts/reviews/constrained_nsga2_opt_in_review.md)
- [constrained NSGA-II smoke report](../../artifacts/constrained_nsga2_smoke_report_constrained_nsga2_smoke1.md)
- [constrained NSGA-II DTLZ smoke report](../../artifacts/constrained_nsga2_dtlz_smoke_report_constrained_nsga2_dtlz_smoke1.md)
- [constrained NSGA-II opt-in review report](../../artifacts/constrained_nsga2_opt_in_review_report.md)
- [constrained NSGA-II broader evidence review update](../../artifacts/reviews/constrained_nsga2_broader_evidence_review_update.md)
- [constrained NSGA-II scope decision update](../../artifacts/reviews/constrained_nsga2_scope_decision_update.md)
- [constrained NSGA-II evidence package](../../artifacts/reviews/constrained_nsga2_evidence_package.md)
- [constrained NSGA-II broader review update report](../../artifacts/constrained_nsga2_broader_review_update_report.md)
- [constrained NSGA-II external comparison review](../../artifacts/reviews/constrained_nsga2_external_comparison_review.md)
- [constrained NSGA-II external parity decision note](../../artifacts/reviews/constrained_nsga2_external_parity_decision_note.md)
- [constrained NSGA-II external comparison report](../../artifacts/constrained_nsga2_external_comparison_report_pymoo_constrained_compare1.md)
- [constrained NSGA-II external comparison review report](../../artifacts/constrained_nsga2_external_comparison_review_report.md)
- [constrained NSGA-II external stress report](../../artifacts/constrained_nsga2_external_stress_report_pymoo_constrained_stress1.md)
- [constrained NSGA-II external stress review](../../artifacts/reviews/constrained_nsga2_external_stress_review.md)
- [constrained NSGA-II external parity decision update](../../artifacts/reviews/constrained_nsga2_external_parity_decision_update.md)
- [constrained NSGA-II external stress scope decision](../../artifacts/reviews/constrained_nsga2_external_stress_scope_decision.md)
- [constrained NSGA-II external stress review report](../../artifacts/constrained_nsga2_external_stress_review_report.md)
- [constrained NSGA-II evidence matrix](../../artifacts/reviews/constrained_nsga2_evidence_matrix.md)
- [constrained NSGA-II active validation pause decision](../../artifacts/reviews/constrained_nsga2_active_validation_pause_decision.md)
- [constrained NSGA-II future roadmap](../../artifacts/hypotheses/constrained_nsga2_future_roadmap.md)
- [constrained NSGA-II validation closure report](../../artifacts/constrained_nsga2_validation_closure_report.md)
- [stress report](../../artifacts/constrained_ga_stress_report_constrained_ga_stress1.md)
- [second toy stress report](../../artifacts/constrained_box_quadratic_stress_report_constrained_box_quadratic_stress1.md)
- [tightness stress report](../../artifacts/constrained_box_quadratic_tightness_report_tightness_stress1.md)
- [equality smoke report](../../artifacts/constrained_equality_plane_quadratic_smoke_report_equality_smoke1.md)
- [equality tolerance stress report](../../artifacts/constrained_equality_tolerance_stress_report_equality_tolerance_stress1.md)
- [contract backlog](../../artifacts/hypotheses/constrained_optimization_contract_backlog.json)
