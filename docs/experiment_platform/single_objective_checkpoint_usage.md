# Single-Objective GA Checkpoint Usage Guide

## 1. Status

- status: Phase 1 single-objective checkpoint passed
- latest stress status: 20-seed OneMax checkpoint stress passed
- default_changed: false
- activation: explicit opt-in only
- supported path: single-objective GA
- unsupported paths: NSGA-II, constrained GA, constrained NSGA-II, external comparator

## 2. What This Feature Does

- Writes checkpoint files for explicit opt-in single-objective GA runs.
- Stores population, fitness, evaluation, and generation state.
- Captures and restores RNG state under tested conditions.
- Validates resume compatibility before continuing a run.
- Uses an atomic write helper for checkpoint files.
- Supports explicit `resume_from` checkpoint paths.
- Under the checkpoint stress package, repeated OneMax runs passed exact-match checks for `best_fitness`, `best_genome`, `actual_evaluations`, and `history`.

## 3. What This Feature Does Not Do

- It is not default checkpointing.
- It is not NSGA-II checkpointing.
- It is not constrained checkpointing.
- It is not production fault tolerance.
- It is not distributed checkpointing.
- It is not parallel evaluation.
- It does not guarantee reproducibility across every custom operator or configuration.
- It does not make a production reliability claim.

## 4. Required Checks Before Use

- Same config hash.
- Same problem, dimension, and bounds/representation options.
- Same objective direction.
- Same operator signature.
- `requested_budget >= checkpoint.actual_evaluations`.
- Finite checkpoint fitness values.
- RNG state availability, or explicit warning that true deterministic resume is unavailable.
- Overwrite protection for checkpoint artifacts.
- Compatibility gate behavior for config hash mismatch, operator mismatch, smaller requested budget, non-finite checkpoint fitness, corrupted checkpoint files, overwrite attempts, and missing RNG warnings.

## 5. Known Limitations

- Tested on a small single-objective OneMax verification and a 20-seed OneMax checkpoint stress.
- Custom operator identity can be hard to guarantee.
- Broader single-objective problem/operator diversity stress is not yet complete.
- NSGA-II and constrained paths are unsupported.
- Production reliability is not claimed.

## 6. Stress Review Status

The `checkpoint_stress1` package passed 20/20 OneMax exact-match runs and 7/7 compatibility negative tests. Artifact safety checks covered checkpoint write/load, atomic write policy, overwrite protection, and corrupted checkpoint fail-fast handling.

This stress evidence strengthens the explicit opt-in single-objective GA checkpoint scope. It does not expand the feature to default GA execution, NSGA-II, constrained GA, constrained NSGA-II, external comparator checkpointing, production fault tolerance, distributed checkpointing, or parallel evaluation.

## 7. Example Pseudocode

The following is pseudocode shaped to match the current API. Keep checkpointing explicit and local to the run that needs it.

```python
from ga_lab.experiment.algorithm_checkpoint import CheckpointConfig

checkpoint_config = CheckpointConfig(
    enabled=True,
    output_dir="artifacts/checkpoints",
    run_id="local_single_objective_run_1",
    interval_generations=5,
    resume_from=None,
    write_latest=False,
)

summary, history = run_single_objective_ga(
    config=config,
    problem=problem,
    selection_fn=selection_fn,
    crossover_fn=crossover_fn,
    mutation_fn=mutation_fn,
    init_fn=init_fn,
    rng=rng,
    checkpoint_config=checkpoint_config,
)

resume_config = CheckpointConfig(
    enabled=True,
    output_dir="artifacts/checkpoints",
    run_id="local_single_objective_run_1_resume",
    resume_from="artifacts/checkpoints/local_single_objective_run_1/checkpoint_gen_5.json",
)
```
