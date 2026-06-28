# Branch, Worktree, and Role Rules

Use these rules when multiple contributors or automation jobs are working in
parallel.

## Branch and Worktree Rules

- Use one branch and one worktree per task.
- Branch names should follow `work/<role>/<task>`.
- Keep the task slug stable across the branch name, worktree folder, and output
  folder names.
- If a workspace snapshot is not attached to git, keep following the same task
  slug convention in docs and output paths.
- Do not share generated configs or run directories between active tasks.
- Write task-local experiment outputs under `outputs/<task-slug>/...` or
  `outputs_test/<task-slug>/...`.
- Nightly automation owns `outputs/nightly/<timestamp>_nightly_regression/...`
  and should not be reused for ad hoc experiments.

## Role Boundaries

- Operator research:
  `src/ga_lab/core/`, `src/ga_lab/algorithms/`, `tests/test_operators.py`,
  `tests/test_comparison.py`
- Benchmark addition:
  `src/ga_lab/problems/`, `configs/`, `tests/test_problems.py`,
  `tests/test_factory.py`, `tests/test_problem_outputs.py`,
  `tests/test_baseline_regression.py`
- Report generation:
  `scripts/run_baselines.py`, `scripts/summarize_results.py`,
  `scripts/run_nightly.py`, `output/`, `outputs_test/`
- CI maintenance:
  `.github/workflows/`, `Makefile`, `pyproject.toml`, `docs/worktree_rules.md`

Keep diffs inside one role boundary whenever possible. If a task must cross
boundaries, finish the engine or benchmark change first, then update reporting
and CI in a second pass.

## Conflict-Prone Files

These files are touched by many tasks and should be edited deliberately:

- `README.md`
- `Makefile`
- `pyproject.toml`
- `.github/workflows/ci.yml`

When a task needs one of these files, avoid unrelated cleanup in the same
change.

## Shared Procedure Entry Points

- Baseline regression: `tests/test_baseline_regression.py` plus the relevant
  benchmark manifest under `configs/`.
- Add problem: update `src/ga_lab/problems/`, `configs/`, and the problem tests
  together.
- Result summary: `scripts/summarize_results.py`.
- Nightly automation: `scripts/run_nightly.py`.

## Project Boundaries

This repository is a single-project research harness today, not a monorepo.

- Experiment engine stays in `src/ga_lab/`.
- Visualization stays in `scripts/streamlit_dashboard.py` and generated reports.
- Future service code should live outside `src/ga_lab/`, for example under
  `apps/service/` or `services/`.

Do not mix service-layer concerns into engine modules just to support dashboards
or automation.
