# Ablation And Claims Freeze

이 문서는 현재 저장소가 공식으로 남기는 claim, experimental로 남기는 claim, 비추천으로 내리는 claim을 고정합니다.

## Scope

모든 해석은 validated size range 안에서만 합니다.

| Problem | Validated sizes |
| --- | --- |
| onemax | `32 / 64 / 128` |
| knapsack | `20 / 30 / 80` |
| tsp | `10 / 20 / 50` |
| zdt1 | `10 / 20 / 50` |

## Statistical Hardening Policy

- paired seed design
- mean / std / median
- 95% bootstrap CI
- paired rank-biserial effect size
- paired significance test
- function evaluation budget fairness

## Internal Claims Freeze

| Item | Status | Why |
| --- | --- | --- |
| onemax practical default = `hill_climb` | Official | same target reachability, far fewer evaluations |
| knapsack practical default = `greedy_local_search` | Official | cheap baseline를 stable하게 대체할 external-wide evidence가 없음 |
| `tsp_medium_hybrid.json` | Official | medium(`20`) quality-first internal path로만 유지 |
| zdt NSGA-II preset family | Official | 가장 설명 가능한 multi-metric default path |
| knapsack seed/repair hybrid | Experimental | pure-GA gap closing에는 의미 있음 |
| zdt1 large `mutation_archive` | Experimental | cheap HV-first path로는 의미 있음 |
| onemax GA-family practical solver claim | Not recommended | hill climb가 더 낫다 |
| knapsack full memetic hybrid | Not recommended | complexity 대비 added value 부족 |
| tsp large hybrid | Not recommended | strongest cheap baseline을 넘지 못함 |

## Ablation Takeaways

### onemax

- hill climb vs pure GA, large(`128`)
  - success rate: 둘 다 `1.0`
  - mean evaluations-to-target: `548.6` vs `4827.0`
- mutation-only EA
  - mean evaluations-to-target: `9954.1`

결론:

- practical default는 계속 `hill_climb`
- mutation-only EA도 practical solver로는 승격하지 않습니다

### knapsack

- internal ablation에서는 seed / repair가 pure-GA gap closing의 핵심이었습니다
- full local-improvement memetic path는 extra complexity만 늘리고 실익이 약했습니다

external family addendum:

- uncorrelated family에서는 greedy practical default 유지
- weakly / strongly correlated family에서는 seed/repair hybrid signal
- subset-sum family에서는 broad rule 없음

### tsp

- medium(`20`) ablation에서는 nearest-neighbor seeding이 핵심 기여였습니다
- official hybrid는 internal medium quality-first path로만 유지합니다
- external subset에서는 hybrid official path를 더 넓히지 않습니다

### zdt

- pure NSGA-II default path는 유지합니다
- large에서는 HV, spread, Pareto coverage tradeoff 문구를 계속 분리합니다

## External Addendum

| Item | External status | Reading |
| --- | --- | --- |
| monotone bitstring practical default = `hill_climb` | Family-conditional external | `OneMax`, `LeadingOnes` tested subset |
| deceptive trap tested family = pure GA 우세 | Family-conditional external | `trap4(128)` |
| `Jump_k` broad rule | Experimental / insufficient evidence | `jump4(128)` tie |
| knapsack uncorrelated family | Family-conditional external | greedy practical default 유지 |
| knapsack weakly correlated family | Family-conditional external | seed/repair hybrid signal |
| knapsack strongly correlated family | Family-conditional external | seed/repair hybrid signal |
| knapsack subset-sum family | Experimental / insufficient evidence | practical 차이가 너무 작음 |
| tsp default = `nearest_neighbor_2opt` | External supported | tested TSPLIB subset |
| `tsp_medium_hybrid.json` external promotion | Experimental / insufficient evidence | baseline을 넘지 못함 |
| zdt NSGA-II default path | External supported | tested `ZDT2`, `ZDT3` subset |

## What Is Actually Proven

- onemax practical default는 `hill_climb`
- knapsack internal practical default는 `greedy_local_search`
- tsp practical default는 `nearest_neighbor_2opt`
- medium(`20`)의 `tsp_medium_hybrid.json`은 internal quality-first path
- zdt default path는 pure NSGA-II

## What Is Only A Promising Direction

- knapsack seed/repair hybrid
- zdt1 large `mutation_archive` as HV-first cheap path
- family-conditioned bitstring / knapsack rules의 더 넓은 benchmark expansion

## What Is Not Safe To Claim

- validated size range 밖 일반화
- bitstring 전체에 single broad default
- knapsack external-wide single default
- `tsp_medium_hybrid.json` external official path
- zdt를 single scalar winner 문제처럼 설명하는 문장
