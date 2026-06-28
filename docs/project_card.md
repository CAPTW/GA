# Project Card

## What This Repository Is

The Evolutionary Solver Benchmark Lab is an evidence-backed optimization lab for comparing cheap heuristics, pure genetic
algorithms, selected hybrid GA paths, and NSGA-II under matched evaluation budgets.

It is not a "GA wins everything" demo. The repo is designed to answer a more useful question:

> Which solver family should you actually choose for a problem family, a size tier, and an
> objective priority when you care about matched-budget evidence rather than algorithm branding?

## Problem Families Covered

- bitstring families:
  - OneMax
  - LeadingOnes
  - deceptive trap
  - Jump_k
- knapsack:
  - validated internal instances
  - canonical external families from kplib
- tsp:
  - validated internal synthetic instances
  - representative TSPLIB subset
- multi-objective:
  - ZDT1 internal path
  - ZDT2 / ZDT3 external support subset

## Solver-Choice Philosophy

- Cheap baselines are real competitors, not strawmen.
- Function evaluation budget is the primary fairness policy.
- External support matters, but narrow internal paths stay narrow when external evidence is weak.
- Family-conditioned rules are preferred over broad defaults when problem families split.
- Claims are governed by a machine-readable registry and a drift checker, not only by prose docs.

## What Is Strongest Today

- TSP practical default:
  - `nearest_neighbor_2opt` has external support on the tested TSPLIB subset.
- ZDT family default path:
  - pure NSGA-II has external support over random archive baselines on the tested ZDT subset.
- Solver-choice rigor:
  - the repo distinguishes internal, external, family-conditional, and experimental claims.
- Governance:
  - claim registry, drift report, and benchmark tiers are already wired into CI-style workflows.

## Where The Repo Is Intentionally Narrow

- onemax does not become a claim about every bitstring problem.
- bitstring deceptive and multimodal families are kept separate from monotone families.
- knapsack does not claim one broad external-wide default.
- `tsp_medium_hybrid.json` remains a narrow internal quality-first path only.
- large-tier ZDT remains tradeoff-based instead of collapsing into a scalar one-best claim.

## Why This Is More Than A Toy GA Repo

- matched-budget baseline comparison
- hybrid ablation rather than black-box "hybrid is better" claims
- external benchmark checks with provenance notes
- machine-readable claims
- drift detection and benchmark governance

## Best Entry Points

- [README](../README.md)
- [Benchmark card](benchmark_card.md)
- [Solver matrix](solver_matrix.md)
- [Release status](release_status.md)
- [Reproducibility and governance](reproducibility_and_governance.md)
