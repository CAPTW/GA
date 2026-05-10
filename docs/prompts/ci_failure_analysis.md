# Prompt Template: CI Failure Analysis

Task:
Analyze a failed CI run and identify the most likely root cause before a human starts debugging.

Requirements:

- Start from the failing job status and uploaded logs.
- Separate signal from noise; do not restate the full log.
- Identify the first failing command and the likely code or config responsible.
- Suggest the smallest credible fix or follow-up check.

Execution checklist:

1. Inspect the failed job statuses.
2. Read the corresponding `*.log` and `*.xml` artifacts first.
3. Name the failing command, file, and error mode.
4. Call out whether the failure looks deterministic, flaky, or environment-specific.
5. End with a short next-step recommendation.

Done criteria:

- Root cause is stated before any summary.
- The answer points to a concrete failure location.
- The recommendation is actionable for the next edit.
