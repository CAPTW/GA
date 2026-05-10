# Repo Audit Summary

## Repo Structure

실제 Git repository root는 `ga-codex-lab/`이다. source-only baseline은 기존 구조를 유지하며 다음 계층을 포함했다.

- `src/ga_lab/`: GA runtime, algorithms, operators, problems, config, runner
- `tests/`: algorithm, operator, config, factory, constraint, checkpoint tests
- `configs/`: JSON experiment configs and presets
- `docs/`: existing governance, optimization, release, reproducibility docs
- `scripts/`: validation, comparison, benchmark, reporting utilities
- `services/`, `skills/`, `audit/`, `benchmarks/`, `claims/`: repo support modules and governance assets

Generated outputs and large artifacts remain in the working tree but are ignored by Git.

## Main Algorithm Files

- `src/ga_lab/algorithms/single_objective.py`: single-objective GA population loop, elitism, selection, crossover, mutation, adaptive policy hooks, checkpoint/resume support.
- `src/ga_lab/algorithms/nsga2.py`: NSGA-II loop, non-dominated sorting, crowding distance, rank/crowding survivor selection, multi-objective history metrics, opt-in diagnostics/checkpoint support.
- `src/ga_lab/algorithms/constrained_single_objective.py`: feasibility-first constrained single-objective experimental path.
- `src/ga_lab/algorithms/constrained_nsga2.py`: constraint-domination constrained multi-objective experimental path.
- `src/ga_lab/algorithms/_shared.py`: fitness validation, logging rows, convergence helpers, objective direction helpers.

## Operator and Representation Files

- `src/ga_lab/core/representation.py`: bit, real, permutation representation adapters and repair/validate hooks.
- `src/ga_lab/core/crossover.py`: one-point, uniform, arithmetic, order crossover.
- `src/ga_lab/core/mutation.py`: bit-flip, gaussian, swap, inversion mutation.
- `src/ga_lab/core/selection.py`: tournament, rank, roulette, crowded tournament selection.
- `src/ga_lab/factory.py`: runtime context assembly and post-operator representation repair.

## Config and Runner

- `src/ga_lab/config.py`: JSON config loader, nested plugin blocks, aliases, seed validation, objective directions, plugin validation.
- `src/ga_lab/runner.py`: `run_experiment`, output directory creation, `config.json`, `config.canonical.json`, `summary.json`, `run_metadata.json`, `history.csv`.
- `src/ga_lab/consumer_cli.py` and `src/ga_lab/entrypoints.py`: `ga-lab-run`, preset, demo, recommendation entrypoints.

JSON support exists. YAML support was not found.

## Existing NSGA-II Summary

The NSGA-II implementation includes vector fitness validation, objective direction handling, non-dominated sorting, crowding distance, rank/crowding survival, tournament-compatible selection state, hypervolume/reference metrics, diagnostics hooks, and checkpoint/resume support through opt-in checkpoint modules.

The implementation is preserved for future Advanced Mode. It should not be rewritten for the first Dorm Assignment PoC.

## Existing Single-Objective GA Summary

The single-objective GA includes fixed-size population initialization, scalar fitness evaluation, elitism, tournament/rank/roulette selection, crossover/mutation hooks, representation repair after operators, adaptive policy hooks, history rows, target fitness, early stop, and checkpoint/resume support.

This is the best baseline for Basic Mode, but Dorm Assignment needs a new problem adapter, assignment-aware representation/decoding, scoring, repair, operators, and exports.

## Tests

Relevant existing tests include:

- `tests/test_nsga2.py`
- `tests/test_constrained_nsga2.py`
- `tests/test_constrained_single_objective_ga.py`
- `tests/test_operators.py`
- `tests/test_mutation_contract.py`
- `tests/test_fitness_validation.py`
- `tests/test_config.py`
- `tests/test_factory.py`
- checkpoint/resume tests under `tests/test_*checkpoint*.py`

## Source-Only Baseline Policy

Baseline commit includes source/config/docs/tests and selected repo support files only. Generated outputs, checkpoints, and artifacts were excluded from Git tracking without deleting local files.
