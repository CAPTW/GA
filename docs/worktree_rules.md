# Branch, Worktree, and Role Rules

Use these rules when multiple Codex threads or contributors are working in parallel.

## Branch and worktree rules

- Use one branch and one worktree per task.
- Branch names should follow `codex/<role>/<task>`.
- Keep the task slug stable across the branch name, worktree folder, and output folder names.
- If a workspace snapshot is not attached to git, keep following the same task slug convention in docs and output paths.
- Do not share generated configs or run directories between active tasks.
- Write task-local experiment outputs under `outputs/<task-slug>/...` or `outputs_test/<task-slug>/...`.
- Nightly automation owns `outputs/nightly/<timestamp>_nightly_regression/...` and should not be reused for ad hoc experiments.

## Role boundaries

- Operator research:
  `src/ga_lab/core/`, `src/ga_lab/algorithms/`, `tests/test_operators.py`, `tests/test_comparison.py`
- Benchmark addition:
  `src/ga_lab/problems/`, `configs/`, `tests/test_problems.py`, `tests/test_factory.py`, `tests/test_problem_outputs.py`, `tests/test_baseline_regression.py`
- Report generation:
  `scripts/run_baselines.py`, `scripts/summarize_results.py`, `scripts/run_nightly.py`, `docs/prompts/`, `output/`, `outputs_test/`
- CI maintenance:
  `.github/workflows/`, `Makefile`, `pyproject.toml`, `AGENTS.md`, `docs/worktree_rules.md`

Keep diffs inside one role boundary whenever possible. If a task must cross boundaries, finish the engine or benchmark change first, then update reporting and CI in a second pass.

## Conflict-prone files

These files are touched by many tasks and should be edited deliberately:

- `README.md`
- `Makefile`
- `pyproject.toml`
- `.github/workflows/ci.yml`
- `tests/test_prompt_templates.py`
- `AGENTS.md`

When a task needs one of these files, avoid unrelated cleanup in the same change.

## Shared procedure entry points

- Baseline regression skill: `skills/baseline-regression/SKILL.md`
- Add problem skill: `skills/add-problem/SKILL.md`
- Result summary skill: `skills/summarize-results/SKILL.md`
- Nightly automation: `scripts/run_nightly.py`

## Project boundaries

This repository is a single-project research harness today, not a monorepo.

- Experiment engine stays in `src/ga_lab/`
- Visualization stays in `scripts/streamlit_dashboard.py` and generated reports
- Future service code should live outside `src/ga_lab/`, for example under `apps/service/` or `services/`

Do not mix service-layer concerns into engine modules just to support dashboards or automation.
