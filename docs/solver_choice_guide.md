# Solver Choice Guide

이 문서는 validated size range 안에서 어떤 solver family를 기본으로 써야 하는지 정리합니다.

핵심 원칙은 단순합니다.

- 기본값은 internal validated evidence를 따릅니다.
- external evidence가 broad claim을 지지하지 못하면 family-conditioned rule로만 남깁니다.
- function evaluation budget이 1차 공정성 기준입니다.

## Scope

| Problem | Validated sizes |
| --- | --- |
| onemax | `32 / 64 / 128` |
| knapsack | `20 / 30 / 80` |
| tsp | `10 / 20 / 50` |
| zdt1 | `10 / 20 / 50` |

## Budget Fairness

- 초기 population 평가와 final re-evaluation도 budget에 포함합니다.
- hybrid local search가 objective call을 쓰면 같은 budget 안에 포함합니다.
- runtime은 함께 보지만 solver choice의 1차 기준은 아닙니다.

## Official / Experimental / Not Recommended

| Item | Status | Scope | Reading |
| --- | --- | --- | --- |
| `hill_climb` for onemax | Official | validated onemax sizes | practical default |
| `greedy_local_search` for knapsack | Official | validated knapsack sizes | internal practical default |
| `tsp_medium_hybrid.json` | Official | `tsp`, `num_cities=20` only | narrow internal quality-first path |
| `zdt1_{small,medium,large}.json` | Official | validated zdt1 sizes | default multi-metric path |
| knapsack seed/repair hybrid | Experimental | validated knapsack sizes | family-conditioned external signal exists |
| zdt1 large `mutation_archive` | Experimental | `genome_length=50` | cheap HV-first alternative |
| onemax GA-family practical solver claims | Not recommended | validated onemax sizes | hill climb가 더 낫다 |
| knapsack full memetic local-improvement hybrid | Not recommended | validated knapsack sizes | complexity 대비 이득이 약하다 |
| TSP large hybrid path | Not recommended | `num_cities=50` | strongest cheap baseline을 넘지 못한다 |

## Problem-by-Problem Rules

### onemax

- practical default:
  - `hill_climb`
- why:
  - validated large(`128`)에서는 success rate가 pure GA와 같이 `1.0`
  - mean evaluations-to-target는 `548.6` vs pure GA `4827.0`
- pure GA를 쓰는 경우:
  - crossover / mutation / selection operator 실험
  - manifest 기반의 순수 GA artifact 비교
- external note:
  - monotone bitstring family에서는 같은 방향이 재현됩니다
  - 하지만 deceptive / multimodal family까지 묶어 broad claim으로 쓰지는 않습니다

### knapsack

- internal practical default:
  - `greedy_local_search`
- external family-conditioned note:
  - uncorrelated family에서는 그대로 greedy를 권장합니다
  - weakly / strongly correlated family에서는 seed/repair hybrid를 시도할 실익이 있습니다
  - subset-sum-like family에서는 broad rule을 만들지 않습니다
- why:
  - pure GA는 random sampling보다 낫지만 cheap baseline을 안정적으로 뒤집지는 못합니다
  - seed/repair hybrid는 pure-GA gap closing에는 의미가 있지만 external-wide default까지는 아닙니다

### tsp

- practical default:
  - `nearest_neighbor_2opt`
- official narrow path:
  - validated medium(`20`)에서만 `configs/presets/tsp_medium_hybrid.json`
- reading:
  - medium hybrid는 internal quality-first path로는 유지합니다
  - external subset에서는 baseline을 넘지 못했기 때문에 external official path로 넓히지 않습니다
- large:
  - large hybrid는 not recommended입니다

### zdt1

- default:
  - `configs/presets/zdt1_small.json`
  - `configs/presets/zdt1_medium.json`
  - `configs/presets/zdt1_large.json`
- why:
  - validated internal range에서 multi-metric default path로 가장 설명 가능성이 높습니다
  - external `ZDT2` / `ZDT3` subset에서도 random archive 대비 우세가 재현됐습니다
- large(`50`) note:
  - HV만 중요하면 `mutation_archive`를 cheap alternative로 볼 수 있습니다
  - 하지만 default path는 계속 pure NSGA-II입니다

## External Family Addendum

| Family | External reading |
| --- | --- |
| monotone bitstring | `hill_climb` practical default |
| deceptive trap | pure GA tested winner |
| `Jump_k` | 아직 broad rule 없음 |
| knapsack uncorrelated | `greedy_local_search` 유지 |
| knapsack weakly correlated | seed/repair hybrid worth trying |
| knapsack strongly correlated | seed/repair hybrid worth trying |
| knapsack subset-sum | 아직 broad rule 없음 |
| tsp TSPLIB subset | `nearest_neighbor_2opt` 유지 |
| zdt tested subset | pure NSGA-II default path 유지 |

## CLI Helpers

```bash
python scripts/recommend_preset.py --problem onemax --size 128 --priority robust
python scripts/recommend_solver.py --problem onemax --size 128 --priority default
python scripts/recommend_solver.py --problem tsp --size 20 --priority quality
python scripts/recommend_solver.py --problem zdt1 --size 50 --priority hv
```

`recommend_solver.py`는 여전히 problem / size / priority만 다룹니다.
family-conditioned external rule은 benchmark family semantics에 묶여 있어, 아직 stable CLI contract로 올리지 않았습니다.

## Related Docs

- [External validity](external_validity.md)
- [External family solver guide](external_family_solver_guide.md)
- [Ablation and claims freeze](ablation_and_claims.md)
- [Reproducibility and governance](reproducibility_and_governance.md)
