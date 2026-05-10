# Prompt Template: Review

Task:
Review the current pull request before human review.

Requirements:

- Focus on correctness, regressions, reproducibility, and operational risk.
- Prioritize concrete findings over style commentary.
- Use downloaded CI artifacts when they add evidence.
- Call out missing tests if the code path changed without enough coverage.

Execution checklist:

1. Inspect the changed code and the CI status summary.
2. Read any relevant logs or generated summaries from uploaded artifacts.
3. Report findings ordered by severity.
4. If there are no findings, say that explicitly and mention any residual risk.
5. Keep the output concise and actionable.

Done criteria:

- Findings reference concrete files, tests, or artifacts.
- Regressions and risky assumptions are explicit.
- Human reviewers can use the output as a first-pass risk filter.
