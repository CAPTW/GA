# Minimum PoC Plan

This is a plan only. It does not create implementation code or data.

## Toy Dataset Plan

Use an 8-student / 2-room anonymized toy dataset for first validation.

- 8 anonymized students: `S001` through `S008`
- 2 rooms: `R101`, `R102`
- 8 slots total: `R101-A`..`R101-D`, `R102-A`..`R102-D`
- No real student names, emails, phone numbers, service numbers, or personal identifiers.

## Sample CSV Structure

`students.csv` planned columns:

```text
student_id,role,department,track,watch_level,frequent_stay
```

`rooms.csv` planned columns:

```text
room_id,building,floor,zone,cctv_proximity,capacity
```

`relations.csv` planned columns:

```text
student_id_a,student_id_b,relation_type,weight
```

Allowed planned `relation_type` values:

- `previous_roommate`
- `severe_conflict`
- `too_close`

## Minimum Command Draft

```powershell
ga-lab-run --config configs/dorm_assignment/toy_8_students.json --output-root outputs_dorm_poc
```

This command is a draft for a later implementation step. It was not run in this audit phase.

## New Files Needed

- `src/ga_lab/problems/dorm_assignment.py`
- `src/ga_lab/dorm_assignment/__init__.py`
- `src/ga_lab/dorm_assignment/schema.py`
- `src/ga_lab/dorm_assignment/loaders.py`
- `src/ga_lab/dorm_assignment/decoder.py`
- `src/ga_lab/dorm_assignment/scoring.py`
- `src/ga_lab/dorm_assignment/repair.py`
- `src/ga_lab/dorm_assignment/operators.py`
- `src/ga_lab/dorm_assignment/export.py`
- `configs/dorm_assignment/toy_8_students.json`
- `tests/test_dorm_assignment_schema.py`
- `tests/test_dorm_assignment_scoring.py`
- `tests/test_dorm_assignment_repair.py`
- `tests/test_dorm_assignment_runner_smoke.py`

Toy CSV files should be generated only in the implementation phase and must remain anonymized.

## Existing Files To Modify

- `src/ga_lab/problems/registry.py`: register `dorm_assignment`.
- `src/ga_lab/core/representation.py`: either adapt permutation placeholder support or add a new assignment representation.
- `src/ga_lab/core/crossover.py`: add room-block crossover if implemented as core plugin.
- `src/ga_lab/core/mutation.py`: add dorm-specific mutations if implemented as core plugins.
- `src/ga_lab/factory.py`: connect domain repair hooks if representation-level repair is not enough.
- `src/ga_lab/runner.py`: add optional dorm-specific report export hook after the generic summary/history path.

## Existing Functions / Classes To Reuse

- `GAConfig` and `load_config` from `src/ga_lab/config.py`
- `run_experiment` from `src/ga_lab/runner.py`
- `run_single_objective_ga` from `src/ga_lab/algorithms/single_objective.py`
- `run_nsga2` from `src/ga_lab/algorithms/nsga2.py` for future Advanced Mode
- `RepresentationAdapter` and permutation repair pattern from `src/ga_lab/core/representation.py`
- `order_crossover` from `src/ga_lab/core/crossover.py`
- `swap_mutation` from `src/ga_lab/core/mutation.py`
- `tournament_select` and `SelectionState` from `src/ga_lab/core/selection.py`
- `ProblemMetadata` and `as_fitness_vector` from `src/ga_lab/problems/base.py`
- `write_history_csv` and `write_json` from the existing tracking layer

## Test Plan

- Schema tests: validate required CSV columns and reject duplicate `student_id`.
- Decoder tests: map chromosome positions to `room_id` and A/B/C/D slots.
- Placeholder tests: add `EMPTY_n` when capacity exceeds student count.
- Scoring tests: duplicate, missing, severe conflict, officer overlap, repeat roommate, watch placement.
- Repair tests: duplicate/missing repair and severe conflict separation on toy fixtures.
- Operator tests: swap mutation preserves assignment multiset.
- Smoke test: 8 students / 2 rooms produces a decoded assignment and reports.
- Determinism test: fixed random seed reproduces the same best assignment or identical final cost/log signature.

## Success Criteria

- Toy dataset 8명 / 2개 호실 실행 가능.
- severe conflict pair가 같은 방에 없음.
- officer overlap penalty 계산 가능.
- repeat roommate penalty 계산 가능.
- watch placement penalty 계산 가능.
- `final_assignment.csv` 생성 계획 수립.
- `fitness_log.csv` 생성 계획 수립.
- random seed 고정 시 결과 재현 가능하도록 설계.
