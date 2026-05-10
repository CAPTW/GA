# Recommended presets by problem

이 문서는 validated range 안에서 바로 쓸 수 있는 preset lookup을 정리한다.
pure preset이 기본이며, 승격된 hybrid preset은 별도 표기로 넣었다.

## Validated ranges

| Problem | Size key | Validated sizes |
| --- | --- | --- |
| onemax | `genome_length` | `32 / 64 / 128` |
| knapsack | `problem_options.num_items` | `20 / 30 / 80` |
| tsp | `problem_options.num_cities` | `10 / 20 / 50` |
| zdt1 | `genome_length` | `10 / 20 / 50` |

## Pure GA preset lookup

| Problem | Size | Preset | Note |
| --- | --- | --- | --- |
| onemax | `32` | `configs/presets/onemax_small.json` | validated pure GA |
| onemax | `64` | `configs/presets/onemax_medium.json` | validated pure GA |
| onemax | `128` | `configs/presets/onemax_large.json` | large closure completed |
| knapsack | `20` | `configs/presets/knapsack_small.json` | validated pure GA |
| knapsack | `30` | `configs/presets/knapsack_medium.json` | validated pure GA |
| knapsack | `80` | `configs/presets/knapsack_large.json` | large PASS maintained |
| tsp | `10` | `configs/presets/tsp_small.json` | validated pure GA |
| tsp | `20` | `configs/presets/tsp_medium.json` | pure GA baseline for comparison |
| tsp | `50` | `configs/presets/tsp_large.json` | large PASS maintained |
| zdt1 | `10` | `configs/presets/zdt1_small.json` | validated NSGA-II |
| zdt1 | `20` | `configs/presets/zdt1_medium.json` | validated NSGA-II |
| zdt1 | `50` | `configs/presets/zdt1_large.json` | large default remains balanced |

## Promoted hybrid preset

| Problem | Size | Preset | Why it was promoted |
| --- | --- | --- | --- |
| tsp | `20` | `configs/presets/tsp_medium_hybrid.json` | matched-budget confirm에서 pure GA와 nearest-neighbor + 2-opt baseline을 모두 이겼다 |

## How to read this table

- pure preset lookup은 "이 저장소의 기본 GA / NSGA-II preset"을 뜻한다.
- solver family 선택까지 포함하려면 [solver_choice_guide.md](./solver_choice_guide.md)를 본다.
- `tsp_medium_hybrid.json`은 quality-oriented hybrid path로만 승격되었고, TSP 전체 default를 바꾼 것은 아니다.

## Helper examples

```bash
python scripts/recommend_preset.py --problem onemax --size 128 --priority robust
python scripts/recommend_preset.py --problem zdt1 --size 50 --priority hv
python scripts/recommend_solver.py --problem tsp --size 20 --priority quality
```
