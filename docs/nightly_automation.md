# Nightly Automation

The local nightly workflow bundles regression checks, a seed sweep, and summary generation into one command.

## Default command

```bash
make nightly
```

Dry-run the plan without executing commands:

```bash
make nightly-dry-run
```

## What it runs

1. `pytest tests/test_baseline_regression.py tests/test_comparison.py`
2. `python scripts/run_baselines.py --manifest configs/ci/baseline_smoke.json`
3. `python scripts/run_baselines.py --manifest configs/comparisons/onemax_operator_compare_10seeds.json`
4. `python scripts/summarize_results.py` over the nightly output root

## Output layout

Each run gets an isolated directory under:

```text
outputs/nightly/<timestamp>_nightly_regression/
```

Important artifacts:

- `nightly_plan.json`
- `recent_changes.json`
- `nightly_status.json`
- `NIGHTLY.md`
- `logs/*.log`
- top-level `SUMMARY.md`, `results_summary.json`, `RUNS.csv`, `RUNS.jsonl`
- nested suite-level `SUMMARY.md`, `suite_summary.json`, `RETENTION.md`

## Recent-change summary

If git metadata is available, the nightly runner records changed files using `git diff <base>..HEAD`.

- Default base: `HEAD~1`
- Fallback: `git status --short`

If the workspace is not a git checkout, the runner records that explicitly and still executes the fixed regression suite.

## Scheduling

This repository does not require GitHub cron jobs for nightly checks.

- Use the OS scheduler, CI, or another trusted automation runner against the repository root.
- Schedule `make nightly` from a clean checkout of this repository.
- Keep nightly outputs isolated under `outputs/nightly/` so concurrent runs do not reuse the same artifacts.
