# NSGA-II Candidate J Opt-in Usage Guide

## 1. Status

- `candidate_id`: `candidate_j_h_lite_retry2`
- Current decision: `Approved for opt-in experimental profile`
- `default_changed=false`
- Default NSGA-II change: `no`
- Usage scope: explicit opt-in only, for local benchmark research and candidate evidence generation

## 2. What This Candidate Changes

`candidate_j_h_lite_retry2` is a follow-up to the uniform-crossover line that started with `candidate_d_uniform_crossover` and was later stress-tested through `candidate_h_uniform_dedup_mutation_boost`.

- Base candidate: `candidate_h_uniform_dedup_mutation_boost`
- Operator family: uniform-crossover line, kept separate from the default arithmetic-crossover path
- "h-lite retry2" means the candidate keeps light duplicate handling with at most two retry attempts, instead of the stronger duplicate-and-boost behavior used in `candidate_h`
- Dedup and mutation retry are present only in the explicit candidate path
- The default internal NSGA-II path is not changed, and this candidate is only available through explicit runner or config selection

## 3. Allowed Use

- local benchmark research
- ZDT, DTLZ, and WFG smoke comparison runs
- candidate comparison harness
- explicit experimental profile evaluation
- evidence generation for future change-request review

## 4. Disallowed Use

- default NSGA-II replacement
- production optimizer
- general MOEA superiority claim
- claim that it is broadly superior to `pymoo` or `DEAP`
- direct application to constrained or industrial optimization without new evidence
- performance claims from a single-seed result

## 5. Evidence Summary

| benchmark family | positive evidence | negative evidence | interpretation |
| --- | --- | --- | --- |
| ZDT | Better trade-off than `candidate_d` and gentler convergence profile than `candidate_h`; useful gains in HV, distance-family metrics, coverage, and duplicate-rate control | Spacing and nondominated count still lag external comparators | Strongest evidence family for keeping candidate_j as opt-in |
| DTLZ | DTLZ2 and DTLZ4 showed rescue signals compared with `candidate_d`/`candidate_h` | DTLZ3 remained mixed and still shows diversity weakness | Good enough for opt-in research, not enough for default promotion |
| WFG | WFG1/WFG2 smoke did not collapse; trade-off remained moderate | WFG smoke is limited evidence, and external diversity metrics still lead | Supports "opt-in unchanged", not a stronger approval level |
| external comparator | Some ZDT and WFG metrics were competitive; fairness contract stayed clean | `pymoo` and `DEAP` still lead on spacing and nondominated count | External gap is narrower than before, but still real |
| fairness checker | Fail count stayed at zero; warnings were explainable and limited to external operator-family differences | Warnings still exist and must remain visible in reports | Acceptable for controlled opt-in use only |

## 6. Known Risks

- spacing weakness
- nondominated_count weakness
- DTLZ3 gap
- WFG metric and reference-front limitation
- external diversity gap
- risk of ZDT-family overfitting

## 7. Required Checks Before Use

- fairness checker result is `pass` or warning-only
- `default_changed=false` remains intact
- actual evaluations match the requested budget contract
- repeated seeds are used; do not rely on a one-seed run
- artifact suffix is set so results do not overwrite prior evidence
- local baseline check passes
- candidate isolation remains intact

## 8. Example Commands

python scripts/validate_nsga2_candidate_j_extended.py --artifact-suffix candidate_j_example

python scripts/validate_nsga2_candidate_j_extended.py --problems ZDT1,ZDT2,ZDT3,DTLZ2,DTLZ3 --artifact-suffix candidate_j_recheck

python scripts/check_local_baseline.py --output-dir artifacts/candidate_j_usage_guard

## 9. Exit Criteria

- repeated regressions in additional WFG or DTLZ smoke
- widening external diversity gap
- any fairness fail
- contamination of the default path
- inability to reproduce the documented artifacts
