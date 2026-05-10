# GA Codex Lab Agent Instructions

## Goal
This repository is a research and development harness for genetic algorithm experiments.
Prioritize reproducibility, small diffs, measurable improvements, and clear experiment reporting.

## Working rules
1. Before changing algorithm behavior, read the active JSON config and the matching problem module.
2. Keep random seeds explicit and deterministic in tests.
3. Prefer config-driven changes over hard-coded constants.
4. When changing selection, crossover, mutation, or survivor logic, run tests and at least one baseline experiment.
5. Preserve output schemas in `summary.json` and `history.csv` unless the change intentionally versions them.
6. Avoid silent metric renaming.
7. Update README or docs when adding a new problem, operator, or CLI flag.

## Validation checklist
- Run: `python -m pytest`
- Run: `python scripts/run_experiment.py --config configs/onemax_baseline.json`
- If behavior changes, compare best fitness, convergence speed, and reproducibility against the previous run.

## Preferred task style for Codex
- Break work into small, testable changes.
- Explain assumptions if the repository does not define them.
- When uncertain, add a failing test first, then implement the fix.

## Local shared procedures
- Use `skills/baseline-regression/SKILL.md` when algorithm, problem, config, or reporting changes need benchmark verification.
- Use `skills/add-problem/SKILL.md` when adding a new benchmark problem.
- Use `skills/summarize-results/SKILL.md` when converting run directories into Markdown, JSON, and CSV reports.
- Use `docs/worktree_rules.md` for branch, worktree, and role-boundary rules.
- Use `scripts/run_nightly.py` or `make nightly` for the local nightly regression workflow.
