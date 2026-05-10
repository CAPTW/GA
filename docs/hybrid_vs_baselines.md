# Hybrid GA vs Baselines

이 문서는 pure GA가 밀리던 문제에서 hybrid 구성이 실제로 무엇을 보완했고, 무엇은 끝내 default solver를 바꾸지 못했는지 정리한다.

## Scope

모든 해석은 validated size range 안에서만 한다.

| Problem | Validated sizes |
| --- | --- |
| onemax | `32 / 64 / 128` |
| knapsack | `20 / 30 / 80` |
| tsp | `10 / 20 / 50` |
| zdt1 | `10 / 20 / 50` |

## Fairness Policy

- primary basis: matched function evaluation budget
- initial population evaluation included
- final population re-evaluation included
- hybrid local-search objective calls included inside the same budget
- runtime is reported, but it is secondary to matched-budget quality

## Current Gap Recap

| Problem | Pure-GA baseline | Strongest cheap baseline | Current gap |
| --- | --- | --- | --- |
| onemax | `onemax_{small,medium,large}.json` | `hill_climb` | same target, far more evaluations |
| knapsack | `knapsack_{small,medium,large}.json` | `greedy_local_search` | pure GA beats random, but default solver를 바꾸지는 못함 |
| tsp | `tsp_{small,medium,large}.json` | `nearest_neighbor_2opt` | medium/large에서 cheap baseline이 더 강함 |
| zdt1 | `zdt1_{small,medium,large}.json` | `mutation_archive` at large | pure NSGA-II가 이미 핵심 스토리 |

## Ablation Readout

### onemax

- added hybrid: 없음
- reading:
  - hill climb practical default 결론이 더 강해졌다
  - mutation-only EA는 pure GA보다도 느렸다
  - 따라서 onemax에서는 solver-choice rule alone가 가장 명확하다

### knapsack

검토한 구성요소:

- greedy-seeded population only
- repair only
- elite local improvement only
- seed + repair
- seed + repair + local improvement

paired 10 seeds 결과:

| Size | Pure GA | Seeded only | Repair only | Seed + repair | Full hybrid | Greedy baseline | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `20` | `265.85` | `265.26` | `267.17` | `267.17` | `267.17` | `265.26` | repair가 핵심 |
| `30` | `356.21` | `357.19` | `357.19` | `357.19` | `357.19` | `357.19` | 여러 hybrid가 greedy와 tie |
| `80` | `1086.19` | `1098.57` | `1098.57` | `1098.57` | `1098.57` | `1098.57` | pure GA gap은 닫지만 baseline edge는 없음 |

reading:

- small에서는 repair가 핵심 기여였다
- medium/large에서는 seeded-only와 repair-only만으로도 greedy baseline 수준까지 올라갔다
- full hybrid local improvement는 품질을 더 올리지 못했고 extra evaluations와 runtime만 늘렸다

decision:

- seed/repair recipe: Experimental
- full memetic local-improvement recipe: Not recommended
- practical default: 여전히 `greedy_local_search`

### tsp

검토한 구성요소:

- nearest-neighbor seeding only
- bounded 2-opt only
- official preset (`tsp_medium_hybrid.json`)
- strongest cheap baseline (`nearest_neighbor_2opt`)

validated medium(`20`) paired 10 seeds:

| Method | Mean route distance | Reading |
| --- | --- | --- |
| Pure GA | `467.73` | baseline보다 약함 |
| Seeded only | `438.82` | 큰 폭으로 개선, seeding이 핵심 |
| Local-only | `444.49` | pure GA보다는 좋지만 baseline을 넘지는 못함 |
| Official hybrid | `439.19` | pure GA와 baseline 모두 이김 |
| NN + 2-opt baseline | `445.47` | cheap default |

paired interpretation:

- official hybrid vs pure GA:
  - clear win
- official hybrid vs NN + 2-opt:
  - small but statistically visible win
  - relative improvement about `1.4%`
  - p-value `0.0469`
- official hybrid vs seeded-only:
  - no clear difference
- seeded-only vs baseline:
  - mean은 더 좋았지만 trend 수준에 머물렀다

reading:

- medium에서는 nearest-neighbor seeding이 핵심 기여였다
- 2-opt refinement는 official preset을 baseline 위로 올리는 데는 도움을 줬지만, seeded-only보다 분명히 더 낫다고 말할 정도는 아니었다
- 그래서 `tsp_medium_hybrid.json`은 Official을 유지하되, "validated medium quality-first"라는 좁은 claim으로만 남긴다

validated large(`50`) paired 10 seeds:

| Method | Mean route distance | Reading |
| --- | --- | --- |
| Pure GA | `1113.28` | baseline 대비 크게 열세 |
| Seeded only | `596.77` | pure GA gap은 크게 줄임 |
| Full hybrid | `589.97` | pure GA gap closing에는 성공 |
| NN + 2-opt baseline | `584.46` | 여전히 가장 좋음 |

decision:

- medium hybrid preset: Official, but narrow claim only
- large hybrid path: Not recommended

### zdt1

- promoted hybrid: 없음
- reading:
  - 이 저장소의 multi-objective 공식 경로는 여전히 pure NSGA-II다
  - large에서는 mutation archive가 HV-first cheap alternative로는 의미 있지만, default path로 승격되지는 못했다

## Promotion Freeze

### Official

- `configs/presets/tsp_medium_hybrid.json`
  - scope: `tsp`, `num_cities=20`
  - reading: medium quality-first path only
- `configs/presets/zdt1_{small,medium,large}.json`
  - scope: validated zdt1 sizes
  - reading: default multi-metric path

### Experimental

- knapsack seed/repair hybrid recipe
  - pure GA gap closing에는 의미
  - cheap baseline default는 대체하지 못함
- zdt1 large `mutation_archive`
  - HV-first cheap alternative only

### Not Recommended

- onemax practical hybridization attempts
- knapsack full memetic local-improvement hybrid
- TSP large hybrid path
