---
name: baseline-regression
description: Use when a GA operator, problem, config, reporting path, or CI guardrail changes and you need to verify baseline behavior quickly before closing the task.
---

# Baseline Regression

Use this skill for any change that can alter reproducibility or benchmark quality:

- operator logic in `src/ga_lab/core/` or `src/ga_lab/algorithms/`
- problem behavior in `src/ga_lab/problems/`
- config defaults in `configs/`
- reporting or retention logic in `scripts/` or `src/ga_lab/experiment/`
- CI updates that can hide or misreport regressions

## Fast gate

Run the pinned regression tests first:

```bash
python -m pytest tests/test_baseline_regression.py tests/test_comparison.py
```

If the task touched Codex workflow docs or prompts, also run:

```bash
python -m pytest tests/test_prompt_templates.py tests/test_codex_workflows.py
```

## Smoke suite

Run the fast suite manifest into a throwaway directory:

```bash
python scripts/run_baselines.py --manifest configs/ci/baseline_smoke.json --output-root outputs_test
```

Inspect the generated:

- `SUMMARY.md`
- `suite_summary.json`
- `RUNS.csv`
- `RETENTION.md`

## Full baseline compare

If algorithm or problem behavior changed, run the main manifest:

```bash
python scripts/run_baselines.py --manifest configs/baselines/manifest.json --output-root outputs_test
```

If the task is an operator comparison or seed-sweep change, also run:

```bash
python scripts/run_baselines.py --manifest configs/comparisons/onemax_operator_compare_10seeds.json --output-root outputs_test
```

Use `docs/prompts/baseline_compare.md` when the result needs a short review-ready writeup.

## Done criteria

- Regressions are called out before any summary prose.
- The affected label and metric are named explicitly.
- Output paths are recorded.
- If behavior changed intentionally, the new baseline expectation is updated in `tests/test_baseline_regression.py`.
