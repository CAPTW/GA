# Benchmark Card

This page is auto-generated from the claim registry, the latest checked-in benchmark summaries, and
the latest claim drift report.

<!-- BEGIN AUTO-GENERATED: benchmark-card -->
## Benchmark Fairness

- Primary comparison basis: matched function evaluation budget.
- `configured_budget` is the planned objective-evaluation budget for a solver path.
- `actual_evaluations_used` is the objective-call count actually spent by the run.
- `extra_evaluations_from_hybrid` counts extra objective calls used by local search or refinement.
- Initial population evaluation and final re-evaluation stay inside the reported budget policy.
- Wall-clock time is recorded, but it is secondary to matched-budget quality.

## Validated Ranges

| Problem | Validated sizes | Size key |
| --- | --- | --- |
| onemax | `32, 64, 128` | `genome_length` |
| knapsack | `20, 30, 80` | `problem_options.num_items` |
| tsp | `10, 20, 50` | `problem_options.num_cities` |
| zdt1 | `10, 20, 50` | `genome_length` |

## Evidence Snapshot

| Bucket | Count |
| --- | --- |
| Official | 4 |
| Family-conditional | 4 |
| Experimental | 3 |
| ci-gated PASS | 8 |

## Strongest Externally Supported Claims

| Claim | Comparator | Latest status |
| --- | --- | --- |
| Nearest-neighbor + 2-opt remains the externally supported practical default on the tested TSPLIB subset. | `official_pure_ga / official_hybrid` | `PASS` |
| Pure NSGA-II remains the externally supported default path over random archive on the tested ZDT subset. | `random_archive / mutation_archive` | `PASS` |

## Family-Conditional Evidence

| Claim | Scope | Latest status |
| --- | --- | --- |
| Hill climb remains the family-conditional practical default on the tested monotone bitstring subset (OneMax, LeadingOnes). | bitstring monotone | `PASS` |
| A representative pure GA path may outperform hill climb on the tested deceptive trap subset. | bitstring deceptive | `PASS` |
| Greedy local search remains the family-conditional practical default on the tested uncorrelated knapsack subset. | knapsack uncorrelated | `PASS` |
| Seed-repair hybrid remains a family-conditional worth-trying path on the tested correlated knapsack subsets. | knapsack correlated | `PASS` |

## Experimental Only

| Claim | Reading | Latest status |
| --- | --- | --- |
| The tested Jump_k row is still too narrow to support a stable solver default. | Report-only. It is more important to avoid overclaiming than to force a stable Jump_k default from one representative row. | `PASS` |
| The tested subset-sum-like knapsack slice remains experimental. | The evidence is intentionally treated as exploratory so docs do not overstate structured-family support. | `PASS` |
| Large ZDT1 still requires a metric-priority tradeoff note rather than a one-best claim. | Report-only note claim. Governance here is primarily about preventing claim creep in docs. | `NOT_EVALUATED` |

## Important Limitations

- `2` official claims remain intentionally internal-only.
- Bitstring and knapsack should be read through family-conditioned guidance, not broad external defaults.
- `tsp_medium_hybrid.json` remains a narrow internal quality-first path.
- Large-tier ZDT still needs a tradeoff note instead of a scalar one-best claim.
<!-- END AUTO-GENERATED: benchmark-card -->
