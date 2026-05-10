# Runner-level Seed Parallelism Usage Guide

## Status

Status: process backend hardening passed with timeout/resource limitations.

Default backend: `serial`.

This feature is explicit opt-in only. It is runner-level row/seed execution for selected stress runners, not algorithm-internal evaluation parallelism.

## Supported Runners

| runner | status | note |
|---|---|---|
| constrained NSGA-II stress runner | supported, thread verified | Small constrained ZDT box toy serial/thread and resume+thread verification passed. |
| constrained GA stress runner | supported, thread verified, process smoke verified | Small constrained sphere serial/thread, resume+thread, process, and resume+process verification passed. |

## Supported Backends

| backend | status | note |
|---|---|---|
| `serial` | default | Preserves existing runner behavior. |
| `thread` | explicit opt-in, verified | Verified on selected stress runners with deterministic merge. |
| `process` | explicit opt-in, hardened for selected constrained GA smoke | Requires `--allow-process-backend`; not production scalability. |

## What This Feature Does

- Executes planned stress rows at row level.
- Supports serial and thread backends for selected stress runners.
- Supports process backend for selected constrained GA stress runner smoke with explicit allow flag.
- Preserves deterministic merge order in final artifacts.
- Runs resume skip-completed before parallel/process submission.
- Keeps final artifact writing in the main process.
- Captures row failures with row key, backend, exception type, message, and order index.
- Records process picklability failures as failure rows where possible.

## What This Feature Does Not Do

- It is not algorithm-internal evaluation parallelism.
- It is not production scalability.
- It is not distributed execution.
- It is not GPU or cloud execution.
- It does not control external comparator nested parallelism.
- It does not add checkpoint stress runner parallelism.
- It does not extend checkpoint/resume exactness guarantees.
- It does not broaden candidate or constrained approval scope.
- It does not guarantee speedup.

## Required Checks Before Use

- Use an explicit backend option for non-default execution.
- For process backend, pass the explicit allow flag.
- Set a deliberate worker count.
- Use a fresh artifact suffix; do not overwrite existing evidence artifacts.
- Run a local baseline guard for important evidence runs.
- Run serial vs parallel parity checks before enabling a new runner.
- Inspect the resume source artifact before combining resume with process or thread row execution.
- Confirm workers do not write final artifacts.
- Confirm row payloads and worker functions are process-pickle safe before process use.

## Process Backend Caveats

- Process backend remains explicit opt-in through `--allow-process-backend`.
- Worker functions must be top-level importable for Windows spawn compatibility.
- Lambda, local function, closure-heavy, or unpicklable payload paths are expected to fail clearly.
- Timeout handling is metadata-level failed-row handling, not a production timeout guarantee.
- Resource warnings do not imply performance tuning or speedup.
- Process smoke evidence is limited to constrained GA stress runner on a tiny `constrained_sphere` run.
- External comparator nested parallelism is not controlled.

## Known Limitations

- Only selected stress runners are covered.
- Process backend is not verified across all runners.
- Runtime speedup is not guaranteed.
- Hidden RNG or evaluator side effects can break parity.
- External comparator runners are not covered.
- Checkpoint stress runners are not covered.
- Broad benchmark stress was not run for parallelism.
- All custom evaluators/operators are not guaranteed to be pickle-safe.

## Example Commands

Constrained NSGA-II stress runner with thread backend:

```powershell
python scripts/validate_constrained_nsga2_stress.py --problems constrained_zdt_box_toy --seeds 2 --budgets 80 --population-size 4 --artifact-suffix runner_seed_parallel_thread_example --output-dir artifacts --row-execution-backend thread --workers 2
```

Constrained GA stress runner with thread backend:

```powershell
python scripts/validate_constrained_ga_stress.py --problem constrained_sphere --dimension 3 --seeds 2 --budgets 20 --population-size 4 --artifact-suffix runner_seed_parallel_cga_thread_example --output-dir artifacts --row-execution-backend thread --workers 2
```

Constrained GA stress runner with process backend:

```powershell
python scripts/validate_constrained_ga_stress.py --problem constrained_sphere --dimension 3 --seeds 2 --budgets 20 --population-size 4 --artifact-suffix process_backend_example --output-dir artifacts --row-execution-backend process --workers 2 --allow-process-backend
```

Resume plus process backend:

```powershell
python scripts/validate_constrained_ga_stress.py --problem constrained_sphere --dimension 3 --seeds 2 --budgets 20 --population-size 4 --artifact-suffix process_backend_resume_example --output-dir artifacts --resume-from artifacts/constrained_ga_stress_results_process_backend_resume_source1.json --row-execution-backend process --workers 2 --allow-process-backend
```

## Re-review Triggers

- Adding another runner.
- Enabling process backend in a broader smoke or stress run.
- Adding algorithm-internal evaluation map parallelism.
- Adding external comparator nested parallelism control.
- Observing serial/thread/process parity drift.
- Observing nondeterministic row order or artifact overwrite risk.
- Timeout or worker resource behavior changes.
- Any production scalability or speedup claim.
