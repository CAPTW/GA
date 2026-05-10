# Prompt Template: Baseline Compare

Task:
Compare the current branch against the documented baseline suite.

Requirements:

- Use `configs/baselines/manifest.json`
- Produce comparison artifacts in a throwaway output directory
- Call out baseline regressions before any summary

Execution checklist:

1. Run `python scripts/run_baselines.py --manifest configs/baselines/manifest.json --output-root outputs_test`.
2. Inspect:
   - `RUNS.csv`
   - `RUNS.jsonl`
   - `suite_summary.json`
3. Compare each baseline label by:
   - `best_fitness`
   - `mean_fitness`
   - `stop_reason`
   - `final_generation`
4. If a regression exists, name the affected config and metric explicitly.
5. If no regression exists, state that clearly.
6. Keep the conclusion short and actionable.

Done criteria:

- Baseline status is explicit.
- The output paths are recorded.
- Regressions are tied to config labels, not vague descriptions.
