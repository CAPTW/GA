# Prompt Template: Add Selection

Task:
Add a new selection operator named `<selection_name>` to the GA lab.

Requirements:

- Register the operator through the selection registry in `src/ga_lab/core/selection.py`
- Avoid editing `runner.py`
- Keep existing baselines compatible
- Add or update unit tests in `tests/test_operators.py`
- Add a config example only if the operator is user-facing now
- Update `README.md` if available selection names or usage changed

Implementation checklist:

1. Implement the selection function.
2. Add one registry entry with the operator name and compatible algorithms.
3. Confirm `GAConfig.validate()` accepts the new name without adding a new hardcoded list.
4. Add a focused operator test that exercises the new behavior.
5. Run `python -m pytest`.
6. If the operator changes baseline behavior, run `python scripts/run_baselines.py --manifest configs/baselines/manifest.json --output-root outputs_test`.
7. Summarize whether baseline metrics changed.

Done criteria:

- The new selection is selectable from config.
- Existing baseline configs still pass.
- Tests and README are updated in the same change.
