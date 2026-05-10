# Preset scale guide

이 문서는 small / medium / large preset를 "검증된 size tier" 기준으로 읽는 방법을 정리한다.

## Validated size ranges

| Problem | Size key | Validated sizes | Current large tier |
| --- | --- | --- | --- |
| onemax | `genome_length` | `32 / 64 / 128` | `128` |
| knapsack | `problem_options.num_items` | `20 / 30 / 80` | `80` |
| tsp | `problem_options.num_cities` | `10 / 20 / 50` | `50` |
| zdt1 | `genome_length` | `10 / 20 / 50` | `50` |

여기서 `large`는 "이 저장소에서 현재 검증된 가장 큰 tier"를 뜻한다.
검증하지 않은 더 큰 size까지 자동 보장한다는 뜻은 아니다.

## Current large-tier status

| Problem | Large preset | Status | Key evidence | Cost note |
| --- | --- | --- | --- | --- |
| onemax | `configs/presets/onemax_large.json` | closed | 15 seeds confirm, target hit rate `1.0` | mean runtime about `0.25s` |
| knapsack | `configs/presets/knapsack_large.json` | PASS maintained | confirm 기준 pure GA large preset 유지 가능 | solver family choice는 별도 |
| tsp | `configs/presets/tsp_large.json` | PASS maintained | pure GA large preset 자체는 유지 가능 | practical default는 별도 |
| zdt1 | `configs/presets/zdt1_large.json` | closed | 10 seeds confirm, strong multi-metric default | mean runtime about `7.3s` |

## Pure preset lookup

### onemax

- `32` -> `configs/presets/onemax_small.json`
- `64` -> `configs/presets/onemax_medium.json`
- `128` -> `configs/presets/onemax_large.json`

### knapsack

- `20` -> `configs/presets/knapsack_small.json`
- `30` -> `configs/presets/knapsack_medium.json`
- `80` -> `configs/presets/knapsack_large.json`

### tsp

- `10` -> `configs/presets/tsp_small.json`
- `20` -> `configs/presets/tsp_medium.json`
- `50` -> `configs/presets/tsp_large.json`

### zdt1

- `10` -> `configs/presets/zdt1_small.json`
- `20` -> `configs/presets/zdt1_medium.json`
- `50` -> `configs/presets/zdt1_large.json`

## Promoted hybrid preset

- `tsp`, `num_cities=20` quality-oriented path:
  - `configs/presets/tsp_medium_hybrid.json`
  - pure GA preset과 nearest-neighbor + 2-opt baseline을 모두 matched budget에서 이긴 validated hybrid preset이다.

## How to use this guide

1. 문제와 size key를 먼저 확인한다.
2. size가 validated range 안이면 대응하는 pure preset에서 시작한다.
3. solver family까지 포함해서 고르려면 [solver_choice_guide.md](./solver_choice_guide.md)를 본다.

## What this guide does not claim

- validated size 밖의 일반화
- one-size-fits-all 해석
- large tier가 practical default solver까지 자동 결정해 준다는 주장
