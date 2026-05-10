# Rule-Aware GA Design

## Problem Definition

The Dorm Assignment problem assigns about 90-100 students to 4-person dorm rooms. Existing GA/NSGA-II assets remain intact. The first implementation target should be an interpretable single-objective Rule-aware GA. Existing NSGA-II should remain available for later multi-objective candidate comparison.

## Basic and Advanced Modes

Basic Mode:

- Single-objective Rule-aware GA.
- Minimize weighted rule cost.
- Expose rule-level metrics so operators and staff can interpret results.

Advanced Mode:

- Existing NSGA-II extension preserved.
- Used later for Pareto candidate comparison, not first PoC.

## Slot-Level Chromosome

Chromosome is a fixed-length list of assigned `student_id` or `EMPTY` placeholder ids.

- Index = room-slot position.
- Slot = `room_id + A/B/C/D`.
- Decoding maps one gene to one row in the assignment table.

Example:

```text
R101-A: S001
R101-B: S034
R101-C: S022
R101-D: S045
R102-A: S010
```

When student count and slot count differ, add placeholders.

Example:

```text
94 students + 24 rooms * 4 slots = 96 slots
Add EMPTY_1 and EMPTY_2
```

## Constraints

Hard constraints:

- duplicate assignment violation
- missing student violation
- capacity violation
- severe conflict same-room violation

Semi-hard constraints:

- too-close distance violation
- watch placement violation
- repeat roommate 2+ violation
- officer overlap violation

Soft constraints:

- repeat roommate once penalty
- semester policy penalty
- room balance penalty
- frequent stay grouping bonus

## Weighted Cost

Initial single-objective cost:

```text
TotalCost =
100000 * duplicate_violation
+ 100000 * capacity_violation
+ 100000 * severe_conflict_violation
+ 30000  * dangerous_close_distance_violation
+ 10000  * watch_placement_violation
+ 5000   * repeat_roommate_2plus_penalty
+ 3000   * officer_overlap_penalty
+ 1000   * repeat_roommate_once_penalty
+ 500    * semester_policy_penalty
+ 300    * room_balance_penalty
- 200    * frequent_stay_grouping_bonus

Fitness = -TotalCost
```

The existing single-objective runner can maximize this fitness.

## Semester Policy

`semester_mode = first`:

- prefer department cohesion.

`semester_mode = second`:

- prefer navigation / engineering mix.

## Repair Strategy

Repair should run after initialization, crossover, and mutation where feasible:

1. duplicate/missing student repair
2. room capacity repair
3. severe conflict separation repair
4. too-close distance repair
5. officer overlap repair
6. watch placement repair

The existing `factory.py` repair-after-operator hook can be adapted, but the dorm repair itself is new domain logic.

## Operator Strategy

Reusable/adaptable:

- Existing permutation representation can be adapted for unique assignment with students plus `EMPTY`.
- Existing order crossover can be adapted for permutation-style chromosomes.
- Existing swap mutation can be reused as student/slot swap mutation.

Likely new:

- room-block crossover
- slot swap mutation naming/config wrapper
- room swap mutation
- watch-slot mutation
- distance-aware swap mutation

## NSGA-II Extension

Do not delete or rewrite existing NSGA-II. For Advanced Mode, split objectives as:

- Objective 1: minimize 생활지도 리스크
- Objective 2: minimize repeated roommate exposure
- Objective 3: maximize semester policy satisfaction
- Objective 4: maximize operational efficiency

The existing NSGA-II implementation already has non-dominated sorting, crowding distance, objective direction handling, and history metrics.

## Preservation Strategy

- Keep existing `ga`, `nsga2`, and `hybrid_ga` paths.
- Add Dorm Assignment as a new problem module and registry entry.
- Keep source-only baseline small; do not add generated assignments, checkpoints, outputs, or real student data.
- Add toy anonymized data only in a later implementation step if explicitly approved.
