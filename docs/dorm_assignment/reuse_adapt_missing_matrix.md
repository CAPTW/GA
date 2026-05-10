# Reuse Adapt Missing Matrix

Status values: `REUSE`, `ADAPT`, `MISSING`, `UNCLEAR`.

| Category | Feature | Status | File/Location | Required Action |
|---|---|---:|---|---|
| Algorithm Core | population loop | REUSE | `src/ga_lab/algorithms/single_objective.py` | Use existing loop for Basic Mode. |
| Algorithm Core | single-objective GA runner | REUSE | `src/ga_lab/algorithms/single_objective.py`, `src/ga_lab/runner.py` | Keep runner and call through config/runtime context. |
| Algorithm Core | NSGA-II runner | REUSE | `src/ga_lab/algorithms/nsga2.py` | Preserve for Advanced Mode. |
| Algorithm Core | constrained optimization support | ADAPT | `src/ga_lab/constraints.py`, constrained algorithm files | Use generic violation summaries as reference; implement dorm-specific scoring/repair separately. |
| Algorithm Core | scalar fitness evaluation | REUSE | `src/ga_lab/problems/base.py`, `single_objective.py` | Return `Fitness = -TotalCost`. |
| Algorithm Core | vector objective evaluation | REUSE | `src/ga_lab/problems/base.py`, `nsga2.py` | Preserve for future multi-objective Dorm mode. |
| Algorithm Core | non-dominated sorting | REUSE | `src/ga_lab/algorithms/nsga2.py` | No Basic Mode change. |
| Algorithm Core | crowding distance | REUSE | `src/ga_lab/algorithms/nsga2.py` | No Basic Mode change. |
| Algorithm Core | tournament selection | REUSE | `src/ga_lab/core/selection.py` | Works with existing selection state. |
| Algorithm Core | elitism | REUSE | `src/ga_lab/algorithms/single_objective.py` | Keep elite carryover. |
| Algorithm Core | crossover | ADAPT | `src/ga_lab/core/crossover.py` | Existing functions work for generic genomes; add room-block crossover for dorm quality. |
| Algorithm Core | mutation | ADAPT | `src/ga_lab/core/mutation.py` | Existing swap can start; add slot/room/watch/distance-aware variants. |
| Algorithm Core | repair | ADAPT | `src/ga_lab/core/representation.py`, `src/ga_lab/factory.py` | Generic representation repair exists; add dorm constraint repair. |
| Algorithm Core | constraint handling | ADAPT | `src/ga_lab/constraints.py`, constrained algorithms | Generic continuous constraint machinery exists; dorm rules need discrete violation model. |
| Algorithm Core | random seed control | REUSE | `src/ga_lab/config.py`, `src/ga_lab/utils/seed.py`, tests | Use existing `seed` field and random alias validation. |
| Algorithm Core | checkpoint/resume | ADAPT | `src/ga_lab/experiment/*checkpoint*`, algorithm files | Existing opt-in checkpoints exist; add only after first Dorm PoC if needed. |
| Representation | chromosome / individual representation | ADAPT | `src/ga_lab/core/representation.py` | Current `Genome = list[float]`; encode student/EMPTY ids as integer-like slots or add adapter. |
| Representation | permutation representation | ADAPT | `src/ga_lab/core/representation.py` | Useful for unique assignment but assumes `0..length-1`; adapt mapping to students + EMPTY. |
| Representation | fixed-length vector representation | REUSE | `src/ga_lab/core/representation.py` | Slot-level chromosome can be fixed length. |
| Representation | assignment decoding | MISSING | not found | Add decoder from chromosome index to `room_id + slot_label`. |
| Representation | dummy/empty placeholder support | MISSING | not found | Add `EMPTY_1..N` placeholder generation and decoding semantics. |
| Operators | order crossover | REUSE | `src/ga_lab/core/crossover.py` | Can preserve permutation uniqueness. |
| Operators | PMX | MISSING | not found | Optional later operator. |
| Operators | block crossover | MISSING | not found | Add room-block crossover for dorm-specific locality. |
| Operators | swap mutation | REUSE | `src/ga_lab/core/mutation.py` | Useful first mutation for permutation slots. |
| Operators | shuffle mutation | MISSING | not found | Optional room/segment shuffling variant. |
| Operators | constraint-aware mutation | MISSING | not found | Add rule-aware targeted mutation. |
| Operators | repair-after-mutation hook | REUSE | `src/ga_lab/factory.py` | Existing wrapper repairs after mutation/crossover. |
| Data / Config | config loader | REUSE | `src/ga_lab/config.py` | JSON config can carry dorm options. |
| Data / Config | JSON/YAML support | ADAPT | `src/ga_lab/config.py` | JSON exists; YAML missing. |
| Data / Config | problem registry | ADAPT | `src/ga_lab/problems/registry.py` | Add dorm problem to static registry. |
| Data / Config | custom problem plugin support | UNCLEAR | `src/ga_lab/problems/registry.py` | Registry is static; external plugin loading not found. |
| Data / Config | runner mode switch | ADAPT | `src/ga_lab/config.py`, `src/ga_lab/algorithms/registry.py` | Existing algorithm switch exists; add documented Basic/Advanced dorm modes. |
| Data / Config | output directory control | REUSE | `src/ga_lab/runner.py`, CLI `--output-root` | Use current output root. |
| Data / Config | reproducibility seed | REUSE | `src/ga_lab/config.py`, `src/ga_lab/utils/seed.py` | Keep seed fixed in PoC tests. |
| Output / Logging | best solution export | ADAPT | `src/ga_lab/runner.py`, problem metrics hooks | Summary exists; dorm needs `final_assignment.csv`. |
| Output / Logging | history export | REUSE | `src/ga_lab/runner.py` | Existing `history.csv`. |
| Output / Logging | fitness log | REUSE | `src/ga_lab/runner.py`, `_shared.py` | Existing history rows can carry cost metrics. |
| Output / Logging | run metadata | REUSE | `src/ga_lab/runner.py` | Existing `run_metadata.json`. |
| Output / Logging | CSV/JSON export | ADAPT | `src/ga_lab/experiment/tracking.py`, `runner.py` | Existing generic outputs; add dorm reports. |
| Output / Logging | visualization hooks | UNCLEAR | `scripts/plot_local_results.py`, dashboards | Generic plotting exists; dorm plots need design. |
| Tests | unit tests | REUSE | `tests/` | Existing test style is reusable. |
| Tests | deterministic seed tests | ADAPT | config/checkpoint tests | Add dorm-specific deterministic assertions. |
| Tests | operator contract tests | REUSE | `tests/test_operators.py`, `tests/test_mutation_contract.py` | Extend for dorm operators. |
| Tests | constrained optimization tests | ADAPT | constrained tests | Reuse patterns, not exact continuous constraints. |
| Tests | small smoke tests | REUSE | `tests/test_*smoke*`, runner tests | Add 8-student / 2-room smoke. |
