# External Validity

이 문서는 internal validated claim과 external benchmark evidence를 분리해서 기록합니다.

핵심 질문은 다음입니다.

- internal validated range에서 맞았던 solver-choice rule이
- public / canonical benchmark subset에서도
- 같은 function evaluation budget 기준으로 유지되는가?

## Scope

External evidence도 아래 validated size range에 닿는 representative subset에서만 해석합니다.

| Problem | Internal validated sizes |
| --- | --- |
| onemax | `32 / 64 / 128` |
| knapsack | `20 / 30 / 80` |
| tsp | `10 / 20 / 50` |
| zdt1 | `10 / 20 / 50` |

validated range 밖 일반화는 하지 않습니다.

## Evidence Layers

| Layer | Meaning |
| --- | --- |
| Internal official | 현재 repo validated range 안에서 재현된 공식 claim |
| External supported | canonical benchmark subset에서도 같은 방향이 재현된 claim |
| Family-conditional external | broad claim은 아니지만 특정 family에서는 외부 근거가 성립하는 claim |
| Experimental / insufficient evidence | 신호는 있으나 범위가 좁거나 결과가 혼재한 경우 |

## Provenance

| Family | Source | Redistribution note |
| --- | --- | --- |
| bitstring canonical families | local synthetic definitions | raw dataset 없음 |
| knapsack | [kplib](https://github.com/likr/kplib) | README 기준 CC BY 4.0, cache fetch only |
| tsp | [TSPLIB mirror](https://github.com/mastqe/tsplib) | mirror license가 명확하지 않아 fetch-only cache |
| zdt family | local analytical definitions | raw dataset 없음 |

## Fairness Policy

- 1차 비교 기준은 matched function evaluation budget입니다.
- 초기 population 평가와 final re-evaluation도 budget에 포함합니다.
- hybrid local search가 objective call을 쓰면 `extra_evaluations_from_hybrid`로 기록하고 같은 budget 안에 포함합니다.
- runtime은 함께 기록하지만 1차 판정 기준은 아닙니다.

## Current Status

| Claim | Internal status | External status | Reading |
| --- | --- | --- | --- |
| onemax practical default = `hill_climb` | Official | Internal only | generic bitstring claim으로 넓히지 않습니다 |
| monotone bitstring practical default = `hill_climb` | Note only | Family-conditional external | `OneMax(128)`, `LeadingOnes(128)` tested subset에서 유지 |
| deceptive trap tested family = pure GA 우세 | Note only | Family-conditional external | `trap4(128)`에서 pure GA > mutation-only > hill climb |
| `Jump_k` broad solver rule | Note only | Experimental / insufficient evidence | `jump4(128)`에서는 셋이 모두 `124.0` |
| knapsack practical default = `greedy_local_search` | Official | Internal only | external-wide single default로는 고정하지 않습니다 |
| knapsack uncorrelated family | Note only | Family-conditional external | greedy practical default 유지 |
| knapsack weakly correlated family | Note only | Family-conditional external | seed-repair hybrid signal |
| knapsack strongly correlated family | Note only | Family-conditional external | seed-repair hybrid signal |
| knapsack subset-sum family | Note only | Experimental / insufficient evidence | practical 차이가 너무 작음 |
| tsp practical default = `nearest_neighbor_2opt` | Official | External supported | tested TSPLIB subset에서 유지 |
| `tsp_medium_hybrid.json` external promotion | Official | Experimental / insufficient evidence | `ulysses22`에서 baseline을 넘지 못함 |
| zdt default path = pure NSGA-II | Official | External supported | tested `ZDT2(20)`, `ZDT3(50)`에서 random archive 대비 우세 |

## Problem Notes

### Bitstring

- `OneMax(128)`
  - `hill_climb` mean evaluations-to-target `548.6`
  - pure GA `4827.0`
  - mutation-only `9954.1`
- `LeadingOnes(128)`
  - `hill_climb` final best fitness `128.0`
  - pure GA `71.4`
  - mutation-only `70.2`
- `trap4(128)`
  - pure GA `111.3`
  - mutation-only `105.9`
  - hill climb `99.8`
- `jump4(128)`
  - pure GA / mutation-only / hill climb 모두 `124.0`

결론:

- monotone family에서는 hill climb가 practical default라는 문장을 external tested subset까지 넓힐 수 있습니다.
- deceptive family까지 묶어서 "bitstring이면 hill climb"라고 쓰는 것은 과장입니다.

### Knapsack

- uncorrelated `n=50`
  - greedy `20995.0`
  - pure GA `20925.4`
  - seed-repair `20995.0`
- weakly correlated `n=50`
  - seed-repair `15764.3`
  - greedy `15712.0`
  - pure GA `15613.7`
- strongly correlated `n=100`
  - seed-repair `35617.0`
  - greedy `35372.0`
  - pure GA `34969.6`
- subset-sum `n=100`
  - seed-repair `29017.0`
  - pure GA `29016.9`
  - random `29016.3`
  - greedy `29001.6`

결론:

- uncorrelated family에서는 greedy practical default가 유지됩니다.
- weakly / strongly correlated family에서는 seed-repair hybrid를 고려할 수 있습니다.
- subset-sum-like family에서는 차이가 너무 작아 external practical claim으로 올리지 않습니다.

### TSP

- `ulysses22`
  - `nearest_neighbor_2opt` `7087.1`
  - official hybrid `7141.8`
  - pure GA `7629.0`
- `berlin52`
  - `nearest_neighbor_2opt` `7767.7`
  - large hybrid `7999.3`
  - pure GA `13100.0`

결론:

- cheap heuristic default는 external support를 얻었습니다.
- medium hybrid path는 internal medium quality-first claim으로만 남깁니다.

### ZDT

- `ZDT2(20)`에서는 NSGA-II가 random archive와 mutation archive를 모두 앞섭니다.
- `ZDT3(50)`에서는 NSGA-II가 random archive는 앞서지만, mutation archive가 HV에서는 더 좋습니다.

결론:

- external support는 "pure NSGA-II default path"까지입니다.
- large metric tradeoff 문구는 계속 필요합니다.

## Related Outputs

- [external_family_summary.md](/Users/IDEAL/OneDrive/문서/Antigravity/GA/ga-codex-lab/outputs/benchmark_summary/external_family_summary.md)
- [external_family_summary.json](/Users/IDEAL/OneDrive/문서/Antigravity/GA/ga-codex-lab/outputs/benchmark_summary/external_family_summary.json)
## Related Docs

- [External family solver guide](external_family_solver_guide.md)
- [Solver choice guide](solver_choice_guide.md)
- [Reproducibility and governance](reproducibility_and_governance.md)
