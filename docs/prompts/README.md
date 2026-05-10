# Codex Task Prompts

These templates keep small GA-lab changes consistent and reviewable.

Available prompts:

- `add_selection.md`: add a new selection operator through the registry path
- `add_problem.md`: add a new problem with config, tests, and docs
- `baseline_compare.md`: rerun baselines and compare against fixed artifacts
- `review.md`: Codex first-pass PR review prompt
- `ci_failure_analysis.md`: Codex failure triage prompt for CI logs and artifacts
- `baseline_regression.md`: Codex baseline regression prompt for CI smoke and regression signals
- `update_readme.md`: update README after feature or workflow changes

Usage pattern:

1. Pick the closest template.
2. Fill in the task-specific names and files.
3. Run the listed verification commands.
4. Confirm tests, baseline comparison, and README updates before closing the task.

Companion repo-local skills:

- `skills/baseline-regression/SKILL.md`
- `skills/add-problem/SKILL.md`
- `skills/summarize-results/SKILL.md`
