# Dorm Assignment Gap Analysis

Status values: `REUSE`, `ADAPT`, `MISSING`, `UNCLEAR`. Priority: `P0` minimum PoC, `P1` quality, `P2` advanced.

| Dorm Requirement | Existing Support | Status | Needed Work | Priority |
|---|---|---:|---|---:|
| students.csv loader | JSON config loader only | MISSING | Add CSV loader with anonymized ids and validation. | P0 |
| rooms.csv loader | JSON config loader only | MISSING | Add room schema loader and slot expansion. | P0 |
| relations.csv loader | No dorm relations loader | MISSING | Add severe/too-close/previous roommate relation loader. | P0 |
| anonymized student_id support | Generic genome ids only | ADAPT | Define `student_id` as opaque string, map to integer genes internally. | P0 |
| room_id, building, floor, zone, cctv_proximity | No dorm schema | MISSING | Add `RoomRecord` schema and distance/proximity metadata. | P0 |
| slot labels A/B/C/D | Fixed vector positions exist | ADAPT | Map chromosome index to `(room_id, slot_label)`. | P0 |
| previous roommates | No domain data | MISSING | Add relation type and repeat roommate scoring. | P0 |
| severe conflict pairs | Generic constraints only | MISSING | Add same-room violation and repair. | P0 |
| too-close pairs | Generic constraints only | MISSING | Add distance/proximity violation model. | P1 |
| officer/NCO role | No domain data | MISSING | Add student role field and overlap scoring. | P0 |
| department | No domain data | MISSING | Add field for first semester cohesion policy. | P1 |
| track: navigation / engineering | No domain data | MISSING | Add field for second semester mix policy. | P1 |
| watch_level | No domain data | MISSING | Add field and watch placement scoring. | P1 |
| frequent_stay | No domain data | MISSING | Add field and grouping bonus. | P2 |
| slot-level assignment | Fixed-length vector exists | ADAPT | Use chromosome index as room-slot position. | P0 |
| room-slot structure: room_id + A/B/C/D | No decoder | MISSING | Add room slot table builder. | P0 |
| EMPTY placeholder for capacity mismatch | No placeholder support | MISSING | Add `EMPTY_n` generation and export handling. | P0 |
| chromosome decode to assignment table | Problem metrics hooks only | MISSING | Add decoder and CSV report model. | P0 |
| duplicate assignment violation | Permutation repair uniqueness exists | ADAPT | Add explicit duplicate/missing violation counts. | P0 |
| capacity violation | Fixed slots prevent capacity if decoded correctly | ADAPT | Validate room-slot count and non-empty capacity. | P0 |
| severe conflict same-room violation | Generic constraint summaries exist | MISSING | Add same-room conflict rule. | P0 |
| too-close distance violation | No dorm distance model | MISSING | Add room distance/proximity lookup. | P1 |
| repeat roommate penalty | No relation scoring | MISSING | Add once vs 2+ repeat penalties. | P0 |
| officer overlap penalty | No role scoring | MISSING | Add per-room role overlap penalty. | P0 |
| semester policy penalty | Config options exist | MISSING | Add `semester_mode` and scoring branch. | P1 |
| watch placement penalty | No watch placement model | MISSING | Add slot/room placement rule. | P1 |
| frequent stay grouping bonus | No domain bonus | MISSING | Add soft bonus calculation. | P2 |
| room balance penalty | Population/problem metrics hooks exist | MISSING | Add per-room balance scoring. | P1 |
| duplicate/missing student repair | Permutation repair exists for `0..N-1` | ADAPT | Add student+EMPTY aware repair. | P0 |
| capacity repair | Slot representation helps | ADAPT | Add room capacity validator; repair should keep fixed slots. | P0 |
| severe conflict separation repair | No domain repair | MISSING | Add swap search repair stage. | P0 |
| too-close distance repair | No domain repair | MISSING | Add distance-aware swap repair. | P1 |
| officer overlap repair | No domain repair | MISSING | Add role-aware swap repair. | P1 |
| watch placement repair | No domain repair | MISSING | Add watch-slot targeted repair. | P1 |
| room-block crossover | No room block operator | MISSING | Add operator that swaps whole room blocks. | P1 |
| student swap mutation | Swap mutation exists | ADAPT | Reuse swap over slot genes with dorm decoder. | P0 |
| slot swap mutation | Swap mutation exists | ADAPT | Rename/configure as dorm slot swap mutation. | P0 |
| room swap mutation | No room-level operator | MISSING | Add full-room block swap mutation. | P1 |
| watch-slot mutation | No watch-specific operator | MISSING | Add targeted watch placement mutation. | P1 |
| distance-aware swap mutation | No distance model | MISSING | Add mutation using room distance/proximity metadata. | P2 |
| final_assignment.csv | Generic history/summary only | MISSING | Add final decoded assignment export. | P0 |
| room_summary.csv | Generic summary only | MISSING | Add per-room report. | P0 |
| violation_report.csv | Generic constraint summary only | MISSING | Add rule-level report. | P0 |
| fitness_log.csv | `history.csv` exists | ADAPT | Either reuse `history.csv` or alias/copy with dorm metrics. | P0 |
| run_summary.json | `summary.json` exists | ADAPT | Extend with dorm rule metrics. | P0 |
| optional fitness_curve.png | Generic plotting scripts exist | UNCLEAR | Add after CSV metrics settle. | P2 |
| optional room_score_distribution.png | No dorm visualization | MISSING | Add later visualization. | P2 |
