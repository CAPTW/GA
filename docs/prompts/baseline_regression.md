# Prompt Template: Baseline Regression

Task:
Review baseline regression signals from CI before a human reviewer looks at the pull request.

Requirements:

- Use baseline regression test logs and benchmark smoke artifacts.
- Treat changes to `best_fitness`, `mean_fitness`, `stop_reason`, `final_generation`, and summary outputs as primary signals.
- Name the affected config label or run explicitly.
- Distinguish between a runner/reporting failure and an algorithm-quality regression.

Execution checklist:

1. Inspect `tests/test_baseline_regression.py` failures first if present.
2. Review benchmark smoke outputs such as `SUMMARY.md`, `RUNS.csv`, `RUNS.jsonl`, and `suite_summary.json`.
3. Identify which baseline label or config broke.
4. State the metric or artifact that regressed.
5. Recommend the smallest next validation step or code area to inspect.

Done criteria:

- The affected baseline is explicit.
- The regressed metric or artifact is explicit.
- The answer helps a reviewer decide whether the PR is safe to inspect further.
