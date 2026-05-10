# NSGA-II Checkpoint Usage Guide

## Status

- status: NSGA-II checkpoint stress passed with structured history-equivalence caveat
- default_changed: false
- activation: explicit opt-in only
- supported path: NSGA-II checkpoint/resume through explicit `checkpoint_config`
- resume support: explicit `checkpoint_config.resume_from`
- verification status: small ZDT1 Phase 2C verification passed; ZDT1 20 seeds x budgets 300/760 stress completed
- stress status: stress2 40/40 exact/equivalence pass; history exact 38/40 and structured `NaN` equivalence 2/40
- compatibility negative tests: 10/10 passed in stress execution
- unsupported: constrained checkpoint, external comparator checkpoint, parallel evaluation, production fault tolerance

## What This Feature Does

- Writes checkpoint files at NSGA-II generation boundaries when `checkpoint_config` is explicitly enabled.
- Supports explicit resume from an NSGA-II checkpoint via `checkpoint_config.resume_from`.
- Stores decision vectors and objective values.
- Stores rank, crowding distance, and front-index cache.
- Stores RNG state when available.
- Stores budget, generation, and evaluation metadata.
- Restores decision vectors, objective values, RNG state, actual evaluations, generation index, and history under compatibility gates.
- Recomputes rank/crowding/front state from restored objectives and compares against cached state.
- Validates checkpoint compatibility before resume.
- Uses an atomic write helper and overwrite protection.
- Under the Phase 2C verification package, actual evaluations, objective matrix, nondominated set, metrics, and history matched the uninterrupted small ZDT1 run.
- Under the NSGA-II checkpoint stress package, ZDT1 20 seeds x budgets 300/760 passed 40/40 exact/equivalence checks using stress2 as the final authoritative artifact.
- Records compatibility negative test behavior and artifact safety evidence for stress execution.

## What This Feature Does Not Do

- It is not default checkpointing.
- It is not constrained NSGA-II checkpointing.
- It is not external comparator checkpointing.
- It is not production fault tolerance.
- It is not distributed checkpointing.
- It is not parallel evaluation.
- It does not guarantee broad benchmark reproducibility.
- It does not guarantee every custom operator/configuration is reproducible.
- It does not turn structured history equivalence into exact history equality.

## Required Checks Before Use

- Use an explicit `checkpoint_config`.
- Use `checkpoint_config.resume_from` only for NSGA-II checkpoints created by the compatible explicit opt-in path.
- Use a safe `run_id` and `output_dir`.
- Do not overwrite existing checkpoints unless explicitly allowed for a controlled local test.
- Keep objective values finite.
- Check config, operator, and problem signature compatibility before relying on a checkpoint artifact.
- Ensure requested budget is not smaller than checkpoint actual evaluations.
- Require RNG state for deterministic resume.
- Keep a disabled-path regression check for important evidence runs.
- Run the local baseline guard for important governance evidence runs.
- Treat stress2 as the final authoritative NSGA-II checkpoint stress artifact, while retaining stress1 as a strict-history diagnostic artifact.
- Distinguish exact history match from structured `NaN` representation equivalence.

## Known Limitations

- Tested on small ZDT1 Phase 2C verification and ZDT1 20 seeds x budgets 300/760 stress.
- Stress2 passed 40/40 exact/equivalence checks, but history exact match was 38/40.
- 2/40 stress cases used structured `NaN` JSON representation equivalence for history; actual evaluations, objective matrix, nondominated set, metrics, generation count, and final population equivalence still matched.
- Compatibility negative tests passed 10/10, but broader negative coverage is not exhaustive.
- Broad benchmark/problem/operator diversity stress is not yet complete.
- Constrained paths are unsupported.
- External comparator checkpoint is unsupported.
- Production reliability is not claimed.

## Example Pseudocode

The following is pseudocode shaped to match the current API. Keep checkpointing and resume explicit.

```python
from ga_lab.experiment.algorithm_checkpoint import CheckpointConfig

checkpoint_config = CheckpointConfig(
    enabled=True,
    output_dir="artifacts/checkpoints/nsga2",
    run_id="local_nsga2_run",
    interval_generations=1,
    resume_from=None,
    write_latest=False,
    allow_overwrite=False,
)

summary, history = run_nsga2(
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
    output_dir="artifacts/checkpoints/nsga2",
    run_id="local_nsga2_resume_run",
    interval_generations=1,
    resume_from="artifacts/checkpoints/nsga2/local_nsga2_run/checkpoint_gen_1.json",
    write_latest=False,
    allow_overwrite=False,
)

resumed_summary, resumed_history = run_nsga2(
    config=config,
    problem=problem,
    selection_fn=selection_fn,
    crossover_fn=crossover_fn,
    mutation_fn=mutation_fn,
    init_fn=init_fn,
    rng=rng,
    checkpoint_config=resume_config,
)
```

## Re-review Triggers

- Broad NSGA-II checkpoint stress planning or execution.
- New problem/operator diversity evidence.
- Any change to NSGA-II rank/crowding/front recomputation.
- Any change to RNG capture/restore policy.
- Any attempt to extend checkpoint/resume to constrained NSGA-II.
- Any production, distributed, external comparator, or parallel evaluation claim.
