---
name: add-problem
description: Use when a new GA benchmark problem must be added with registry wiring, configs, tests, baseline decisions, and documentation kept in sync.
---

# Add Problem

Use this skill when the repository needs a new benchmark problem wired through the standard config and registry flow.

## Required files

- Implementation: `src/ga_lab/problems/<problem_name>.py`
- Registry: `src/ga_lab/problems/registry.py`
- Config example: `configs/`
- Tests:
  - `tests/test_problems.py`
  - `tests/test_factory.py`
  - `tests/test_problem_outputs.py` when new summary fields are exposed
- Docs: `README.md`

Do not bypass the registry or add problem-specific runner branches.

## Workflow

1. Read `docs/prompts/add_problem.md` and the closest existing problem module.
2. Implement `fitness()` plus metadata that matches the representation and genome constraints.
3. Register the problem in `src/ga_lab/problems/registry.py`.
4. Add at least one runnable config under `configs/`.
5. Add focused tests in `tests/test_problems.py` and `tests/test_factory.py`.
6. If the problem adds decoded or domain-specific summary fields, extend `tests/test_problem_outputs.py`.
7. Update `README.md`.

## Baseline decision

If the new problem belongs in the shared benchmark set:

- update `configs/baselines/manifest.json`
- rerun baseline comparison with `python scripts/run_baselines.py --manifest configs/baselines/manifest.json --output-root outputs_test`
- decide whether `tests/test_baseline_regression.py` should pin a golden result for it

If the problem is experimental only, keep it out of the baseline manifest and say so explicitly.

## Done criteria

- The problem runs through config without runner edits.
- Tests, config examples, and docs are updated together.
- Benchmark inclusion is explicit, not implied.
