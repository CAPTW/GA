# Codex workflows for GA R&D

- One operator change per task
- State acceptance criteria before making a change

## Current implementation focus

- Multi-representation engine:
  - bit, real, and permutation support
- Configurable operators:
  - tournament/rank/roulette selection
  - one-point/uniform/order/arithmetic crossover
  - bit-flip/gaussian/swap/inversion mutation
- Multi-objective capability:
  - NSGA-II mode with Pareto front extraction and crowding preservation
- Dashboard:
  - Streamlit-based convergence/summary visualizer

## Suggested next tasks

1. Add additional permutation crossover families (PMX/CX)
2. Add explicit decoding/repair utilities for constrained real/categorical spaces
3. Add hypervolume / spread metrics for NSGA-II summaries
4. Add lightweight regression tests for each new strategy and problem

## Execution examples

- OneMax baseline: `configs/onemax_baseline.json`
- Real GA: `configs/onemax_real_ga.json`
- NSGA-II (ZDT1): `configs/zdt1_nsga2.json`
- TSP baseline: `configs/tsp_baseline.json`
- Knapsack baseline: `configs/knapsack_baseline.json`

## Collaboration rules

- Branch and worktree conventions live in `docs/worktree_rules.md`
- Shared Codex procedures live in `skills/`
- Local nightly regression automation lives in `scripts/run_nightly.py`
