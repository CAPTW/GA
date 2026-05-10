# External Family Solver Guide

이 문서는 internal validated claim을 그대로 외부로 일반화하지 않고, canonical benchmark family 기준으로 어디까지 말할 수 있는지를 정리합니다.

## Scope

외부 해석도 아래 validated size 범위에 맞닿아 있는 representative subset에서만 다룹니다.

| Problem | Internal validated sizes | External family check |
| --- | --- | --- |
| onemax | `32 / 64 / 128` | `OneMax(128)`, `LeadingOnes(128)`, `trap4(128)`, `jump4(128)` |
| knapsack | `20 / 30 / 80` | kplib `uncorrelated(50)`, `weakly_correlated(50)`, `strongly_correlated(100)`, `subset_sum(100)` |
| tsp | `10 / 20 / 50` | TSPLIB `ulysses22`, `berlin52` |
| zdt1 | `10 / 20 / 50` | `ZDT2(20)`, `ZDT3(50)` |

## Provenance

| Family | Source | Note |
| --- | --- | --- |
| bitstring families | local canonical synthetic definitions | raw dataset 없음 |
| knapsack families | [kplib](https://github.com/likr/kplib) | README 기준 CC BY 4.0, fetch/cache only |
| tsp instances | [TSPLIB mirror](https://github.com/mastqe/tsplib) | mirror license가 명확하지 않아 fetch-only cache |
| zdt families | local analytical definitions | raw dataset 없음 |

## Fairness

- 1차 비교 기준은 matched function evaluation budget입니다.
- 초기 population 평가와 final re-evaluation도 budget에 포함합니다.
- hybrid local search가 objective call을 쓰면 `extra_evaluations_from_hybrid`로 기록하고 같은 budget 안에 포함합니다.
- runtime은 함께 기록하지만 solver-choice의 1차 판정 기준은 아닙니다.

## Current Claim Freeze

| Problem family | Current internal official claim | Current external supported claim | Current strongest comparator | Remaining gap | Family-conditional? | Evidence still missing |
| --- | --- | --- | --- | --- | --- | --- |
| bitstring family | onemax validated size에서는 `hill_climb`가 practical default | monotone family에서는 `hill_climb` practical default | `hill_climb` / pure GA / mutation-only EA | deceptive, multimodal family는 다른 ranking을 보임 | Yes | 더 다양한 deceptive / multimodal family |
| knapsack family | internal practical default는 `greedy_local_search` | single broad default보다 family-conditioned rule이 더 정직함 | `greedy_local_search` / seed-repair hybrid | family별로 ranking이 갈림 | Yes | 더 넓은 kplib subset과 seed set |
| tsp | practical default는 `nearest_neighbor_2opt` | tested TSPLIB subset에서 same direction 재현 | `nearest_neighbor_2opt` | medium hybrid external 승격 근거 부족 | No | 추가 TSPLIB subset은 있으면 좋지만 필수는 아님 |
| zdt | default path는 pure NSGA-II | tested `ZDT2/ZDT3` subset에서 random archive 대비 우세 | `random_archive` / `mutation_archive` | large metric tradeoff 문구는 계속 필요 | No | 추가 family와 metric-priority 반복 확인 |

## Bitstring Family Results

| Family | Compared solvers | Reading |
| --- | --- | --- |
| `OneMax(128)` | `hill_climb` vs pure GA vs mutation-only EA vs random | `hill_climb` mean evaluations-to-target `548.6`, pure GA `4827.0`, mutation-only `9954.1` |
| `LeadingOnes(128)` | same | `hill_climb` final best fitness `128.0`, pure GA `71.4`, mutation-only `70.2` |
| `trap4(128)` | same | pure GA `111.3`, mutation-only `105.9`, hill climb `99.8` |
| `jump4(128)` | same | pure GA / mutation-only / hill climb가 모두 `124.0`으로 tie |

정리:

- monotone family에서는 `hill_climb` practical default가 external tested subset에서도 유지됩니다.
- deceptive trap family에서는 pure GA가 더 좋습니다.
- `Jump_k` tested subset에서는 broad rule이 나오지 않아 Experimental / insufficient로 둡니다.
- 따라서 "bitstring 문제면 hill climb" 같은 broad claim은 과장입니다.

## Knapsack Family Results

| Family | Solver ranking on tested subset | Reading |
| --- | --- | --- |
| uncorrelated `n=50` | `greedy_local_search` = seed-repair hybrid > pure GA > random | greedy practical default 유지 |
| weakly correlated `n=50` | seed-repair hybrid > greedy > pure GA > random | family-conditioned hybrid signal |
| strongly correlated `n=100` | seed-repair hybrid > greedy > pure GA > random | family-conditioned hybrid signal |
| subset-sum `n=100` | seed-repair ≈ pure GA ≈ random > greedy | 차이가 너무 작아 practical claim으로 승격하지 않음 |

핵심 수치:

- uncorrelated:
  - greedy `20995.0`
  - pure GA `20925.4`
  - seed-repair `20995.0`
- weakly correlated:
  - seed-repair `15764.3`
  - greedy `15712.0`
  - pure GA `15613.7`
- strongly correlated:
  - seed-repair `35617.0`
  - greedy `35372.0`
  - pure GA `34969.6`
- subset-sum:
  - seed-repair `29017.0`
  - pure GA `29016.9`
  - random `29016.3`
  - greedy `29001.6`

정리:

- external knapsack family 전체에 single broad default를 고정하지 않습니다.
- uncorrelated family에서는 `greedy_local_search`를 유지해도 충분합니다.
- weakly / strongly correlated family에서는 seed-repair hybrid를 시도할 실익이 있습니다.
- subset-sum-like family에서는 practical 차이가 너무 작아 solver-choice rule을 새로 만들지 않습니다.

## TSP / ZDT Maintenance

### TSP

- `ulysses22`
  - `nearest_neighbor_2opt`: `7087.1`
  - official medium hybrid: `7141.8`
  - pure GA: `7629.0`
- `berlin52`
  - `nearest_neighbor_2opt`: `7767.7`
  - large hybrid: `7999.3`
  - pure GA: `13100.0`

reading:

- cheap heuristic default의 external support는 유지됩니다.
- `tsp_medium_hybrid.json`을 external official path로 넓히지는 않습니다.

### ZDT

- `ZDT2(20)`
  - NSGA-II HV `11.2716`
  - random archive HV `8.5979`
  - mutation archive HV `11.1372`
- `ZDT3(50)`
  - NSGA-II HV `11.7491`
  - random archive HV `9.3798`
  - mutation archive HV `12.1591`

reading:

- pure NSGA-II default path의 external support는 유지됩니다.
- 다만 large tier와 family별 metric priority tradeoff는 계속 분리해서 써야 합니다.

## Classification

| Item | Classification |
| --- | --- |
| monotone bitstring practical default = `hill_climb` | Family-conditional external |
| deceptive trap tested family = pure GA 우세 | Family-conditional external |
| `Jump_k` broad solver rule | Experimental / insufficient evidence |
| knapsack uncorrelated family = `greedy_local_search` | Family-conditional external |
| knapsack weakly correlated family = seed-repair hybrid worth trying | Family-conditional external |
| knapsack strongly correlated family = seed-repair hybrid worth trying | Family-conditional external |
| knapsack subset-sum broad rule | Experimental / insufficient evidence |
| tsp default = `nearest_neighbor_2opt` | External supported |
| tsp medium hybrid external promotion | Experimental / insufficient evidence |
| zdt default path = pure NSGA-II | External supported |

## Why There Is No `--family` Solver Helper Yet

`recommend_solver.py`는 여전히 `problem + size + priority`까지만 다룹니다.

이유:

- `trap`, `weakly_correlated`, `subset_sum` 같은 label은 현재 preset contract가 아니라 benchmark family semantics입니다.
- family-conditioned 외부 규칙은 문서와 summary에서는 충분히 설명 가능하지만, stable CLI contract로 고정하기에는 아직 범위가 좁습니다.
- 특히 bitstring `Jump_k`와 knapsack subset-sum family는 아직 broad rule이 없습니다.
