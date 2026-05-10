# Constrained GA Feasibility-First Usage Guide

## 1. Status

- `path_id`: `constrained_ga_feasibility_first`
- `approval_status`: `approved_restricted_experimental_opt_in`
- `approval_type`: `restricted_experimental_opt_in_path`
- `default_changed=false`
- current allowed scope: explicit constrained single-objective experimental opt-in path, project-local constrained toy research smoke/stress evidence generation, feasibility-first protocol experiments, constrained artifact/fairness/trace validation, future review evidence generation
- `scope_change=none`: broader inequality/equality toy evidence has been added, but this path remains restricted experimental opt-in only

## 2. What This Path Does

`constrained_ga_feasibility_first` is a separate constrained single-objective
GA path centered on `run_constrained_single_objective_ga`.

- it uses feasibility-first comparison instead of mixing constraint violation into a penalty scalar
- it has current evidence on `constrained_sphere` smoke/stress, `constrained_box_quadratic` smoke/stress/tightness, and `constrained_equality_plane_quadratic` smoke/tolerance stress as restricted experimental evidence
- it is explicit opt-in only
- it is different from the default `run_single_objective_ga` path

## 3. What This Path Does Not Do

- it is not a default GA replacement
- it is not NSGA-II constraint-domination
- it is not penalty-based constraint handling
- it is not repair-based constraint handling
- it is not constrained multi-objective optimization support
- it is not a product optimizer

## 4. When to Use

- constrained_sphere research smoke and stress work
- constrained_box_quadratic research smoke, stress, and tightness characterization work
- constrained_equality_plane_quadratic research smoke and equality tolerance stress work
- feasibility-first protocol experiments
- constraint schema, fairness, trace, and artifact validation
- future constrained benchmark preparation before broader review

## 5. When Not to Use

- production optimization
- industrial constrained optimization
- constrained NSGA-II claims
- constrained ZDT or DTLZ claims
- single-run superiority claims
- default algorithm replacement

## 6. Required Checks Before Running

- use an artifact suffix
- use multiple seeds for meaningful comparisons
- review the constrained fairness summary
- confirm `actual_evaluations` matches the requested budget
- keep finite constraint validation enabled
- run the local baseline check before sharing results
- run mutation and finiteness regression tests before important comparisons

## 7. Example Command

python scripts/validate_constrained_ga_stress.py --dimension 5 --seeds 30 --budgets 300,1000 --artifact-suffix constrained_ga_review_example

python scripts/validate_constrained_ga_stress.py --problem constrained_box_quadratic --dimension 6 --seeds 30 --budgets 300,1000 --artifact-suffix constrained_box_quadratic_review_example

## 8. Known Trade-offs

- runtime is slower than `random_search_feasibility_first` in the current stress evidence
- evidence is limited to project-local toy benchmarks: `constrained_sphere`, `constrained_box_quadratic`, and `constrained_equality_plane_quadratic`
- `constrained_box_quadratic` budget 300 showed mixed paired `best_feasible_objective`; budget 1000 improved that objective signal, but this remains toy evidence
- tightness stress characterized easy/default/strict variants, but this is still project-local toy evidence
- equality tolerance stress characterized loose/default/strict tolerance variants, including no-feasible/null metric behavior for strict random search, but this remains toy evidence
- per-constraint trace is available as aggregate evidence, while raw trace samples/scope remain limited and must be reported as a limitation
- there is no external constrained comparator yet
- penalty and repair handling are not implemented
- NSGA-II constrained support is not implemented

## 9. Exit / Re-review Criteria

- failure on a second constrained single-objective benchmark
- any fairness fail
- any default path contamination
- any `actual_evaluations` mismatch
- any non-finite constraint regression
- misuse as a default optimizer
- product or general constrained optimizer claim misuse

## 10. Current Evidence Notes

- `constrained_box_quadratic` smoke passed with fairness `pass 140 / warning 0 / fail 0`.
- `constrained_box_quadratic` seed+budget stress passed with fairness `pass 1680 / warning 0 / fail 0`.
- `constrained_box_quadratic` tightness stress passed with fairness `pass 2040 / warning 0 / fail 0`.
- `constrained_equality_plane_quadratic` smoke passed with fairness `pass 140 / warning 0 / fail 0`.
- `constrained_equality_plane_quadratic` tolerance stress passed with fairness `pass 3840 / warning 0 / fail 0`.
- budget 300 repeated feasibility and violation positive signal, while paired `best_feasible_objective` was mixed.
- budget 1000 repeated feasibility, objective, and violation positive signal.
- easy/default/strict tightness variants repeated feasibility and violation positive signal, and best objective paired signal favored constrained GA at budget 1000.
- loose/default/strict equality tolerance variants repeated feasibility, objective, violation, and equality_satisfaction positive signal; random search strict runs had no feasible solution and used `null` feasible objective instead of NaN.
- this evidence does not change default status, NSGA-II status, penalty/repair status, product status, or general constrained optimizer status.
