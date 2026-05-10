# Prompt Template: Add Problem

Task:
Add a new problem named `<problem_name>` to the GA lab.

Requirements:

- Add the implementation under `src/ga_lab/problems/`
- Register it in `src/ga_lab/problems/registry.py`
- Do not edit `runner.py`
- Add tests for problem behavior, factory wiring, and output integration
- Add at least one config example in `configs/`
- Update `README.md`

Implementation checklist:

1. Implement the problem class with `fitness()` and `metadata()`.
2. Register the problem in `src/ga_lab/problems/registry.py`.
3. Add or update compatibility metadata so factory validation works.
4. Add focused tests in:
   - `tests/test_problems.py`
   - `tests/test_factory.py`
   - `tests/test_problem_outputs.py` when the problem exposes summary metrics
5. Add a runnable config example in `configs/`.
6. Run `python -m pytest`.
7. If the new problem is part of the standard benchmark set, update `configs/baselines/manifest.json` and rerun baseline comparison.
8. Update README sections for implemented problems and examples.

Done criteria:

- The new problem runs through config without runner changes.
- Tests, config examples, and README move together.
- Any baseline suite change is explicit and reviewed.
