---
name: summarize-results
description: Use when one or more experiment run directories need to be turned into shareable Markdown, JSON, and CSV summaries without manual spreadsheet work.
---

# Summarize Results

Use this skill after ad hoc runs, baseline suites, or nightly automation.

## Single collection summary

For any directory that contains one or more run folders:

```bash
python scripts/summarize_results.py --results-dir outputs
```

This writes:

- `SUMMARY.md`
- `results_summary.json`
- `RUNS.csv`
- `RUNS.jsonl`
- `RETENTION.md`
- `retention_plan.json`

## Suite-first workflow

If the runs come from a suite manifest, prefer the suite runner first:

```bash
python scripts/run_baselines.py --manifest configs/baselines/manifest.json --output-root outputs_test
```

That already emits suite-level `SUMMARY.md`, `suite_summary.json`, `RUNS.csv`, and `RUNS.jsonl`.

## Interpretation checklist

- Start with `SUMMARY.md` for the headline result.
- Use `suite_summary.json` or `results_summary.json` when a downstream tool needs structured data.
- Use `RUNS.csv` when comparing seeds or labels in a table.
- Use `RETENTION.md` and `retention_plan.json` to decide which run directories stay.

## Escalation

- If the user needs operator-vs-operator evidence, run `configs/comparisons/onemax_operator_compare_10seeds.json`.
- If the user needs a dashboard view, open `scripts/streamlit_dashboard.py`.
- If the user needs a short review narrative, start from `docs/prompts/baseline_compare.md`.
