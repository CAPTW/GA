# Benchmark vs simple baselines

이 문서는 pure preset과 cheap baseline을 같은 function evaluation budget으로 비교한 기준 문서다.
주장은 validated size range 안에서만 한다.

## Official comparison scope

| Problem | Size key | Validated sizes |
| --- | --- | --- |
| onemax | `genome_length` | `32 / 64 / 128` |
| knapsack | `problem_options.num_items` | `20 / 30 / 80` |
| tsp | `problem_options.num_cities` | `10 / 20 / 50` |
| zdt1 | `genome_length` | `10 / 20 / 50` |

## Fairness policy

- 1차 비교 기준은 wall-clock time이 아니라 matched function evaluation budget이다.
- 초기 population 평가를 budget에 포함한다.
- 현재 runner의 final population 재평가도 budget에 포함한다.
- wall-clock runtime은 함께 기록하지만 2차 지표다.

## Budget definition

| Algorithm | Configured budget formula | Meaning |
| --- | --- | --- |
| `ga` | `population_size * (generations + 2)` | generation `0..G` 평가 + final population 재평가 |
| `nsga2` | `population_size * (3 * generations + 2)` | current population 평가 + combined parent/offspring 평가 + final population 재평가 |

Preset별 configured budget:

| Problem | Size | Preset | Configured budget |
| --- | --- | --- | --- |
| onemax | `32` | `onemax_small.json` | `2080` |
| onemax | `64` | `onemax_medium.json` | `3360` |
| onemax | `128` | `onemax_large.json` | `14640` |
| knapsack | `20 / 30 / 80` | `knapsack_{small,medium,large}.json` | `9760` |
| tsp | `10 / 20 / 50` | `tsp_{small,medium,large}.json` | `4100` |
| zdt1 | `10 / 20` | `zdt1_{small,medium}.json` | `28960` |
| zdt1 | `50` | `zdt1_large.json` | `48160` |

## Problem-by-problem results

### onemax

| Size | GA success | Random success | Hill-climb success | GA mean evals to target | Hill-climb mean evals to target | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| `32` | `1.00` | `0.00` | `1.00` | `603.4` | `123.3` | GA > random, hill climb faster |
| `64` | `1.00` | `0.00` | `1.00` | `2272.3` | `223.4` | GA > random, hill climb faster |
| `128` | `1.00` | `0.00` | `1.00` | `4783.3` | `538.5` | GA > random, hill climb faster |

해석:

- validated onemax size에서는 추천 preset이 random search보다 target reachability가 확실히 좋다.
- 하지만 same target hit rate 기준으로 hill climb가 훨씬 적은 evaluation으로 끝난다.
- practical default는 GA가 아니라 hill climb다.

### knapsack

| Size | GA best feasible fitness | Greedy local search | Random sampling | GA feasible rate | Random feasible rate | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| `20` | `265.36` | `265.26` | `254.70` | `0.850` | `0.501` | GA and greedy nearly tie |
| `30` | `354.52` | `357.19` | `312.12` | `0.797` | `0.501` | greedy slightly better |
| `80` | `1085.26` | `1098.57` | `844.87` | `0.764` | `0.502` | greedy better |

해석:

- current preset은 random sampling보다 확실히 좋다.
- 하지만 greedy ratio + local search baseline이 validated 범위에서 비슷하거나 약간 더 좋다.
- feasible value만 놓고 보면 pure GA가 universal default는 아니다.

### tsp

| Size | GA distance | Random tours | NN + 2-opt | GA vs random | GA vs NN+2opt | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| `10` | `339.98` | `367.17` | `339.98` | `7.4%` better | tie | cheap baseline enough |
| `20` | `459.73` | `736.20` | `442.17` | `37.6%` better | `4.0%` worse | cheap baseline stronger |
| `50` | `1120.81` | `2120.67` | `587.87` | `47.1%` better | `90.7%` worse | cheap baseline much stronger |

해석:

- pure GA preset은 random tours보다 좋다.
- practical route quality 기준으로는 nearest-neighbor + 2-opt가 더 강하다.
- TSP에서는 cheap heuristic이 baseline이 아니라 사실상 default practical solver에 가깝다.

### zdt1

| Size | GA HV | Mutation archive HV | Random archive HV | GA pareto_ratio | Mutation pareto_ratio | GA spread | Mutation spread | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `10` | `11.7506` | `11.2041` | `10.4565` | `1.0000` | `1.0000` | `0.3342` | `1.8285` | GA clear win |
| `20` | `11.6851` | `11.3011` | `9.7007` | `0.9938` | `1.0000` | `0.4947` | `1.6807` | GA clear win |
| `50` | `11.5797` | `11.6978` | `9.0314` | `0.9400` | `1.0000` | `0.5281` | `0.6510` | mixed |

해석:

- NSGA-II preset은 validated 범위에서 random archive baseline보다 HV가 높다.
- small/medium에서는 mutation-only archive보다도 multi-metric quality가 더 좋다.
- large(`50`)에서는 mutation-only archive가 HV와 pareto_ratio에서 약간 앞서고, NSGA-II preset은 spread가 더 좋다.
- zdt1는 single scalar best 문제처럼 쓰면 안 된다.

## Safe capability claims

아래 수준만 capability claim으로 쓰는 것이 안전하다.

- "For onemax at validated sizes, the recommended preset reaches target more reliably than random search under matched evaluation budgets."
- "For tsp at validated sizes, the recommended preset improves route distance over random tours under matched evaluation budgets."
- "For zdt1 at validated sizes, the recommended NSGA-II preset improves hypervolume over the random archive baseline."

반드시 함께 붙여야 하는 제한:

- onemax practical default는 hill climb다.
- knapsack practical default는 greedy local search다.
- tsp practical default는 nearest-neighbor + 2-opt다.
- zdt1 large는 metric priority tradeoff를 인정해야 한다.

## Related docs

- [recommended_presets_by_problem.md](./recommended_presets_by_problem.md)
- [large_preset_decision_guide.md](./large_preset_decision_guide.md)
- [hybrid_vs_baselines.md](./hybrid_vs_baselines.md)
- [solver_choice_guide.md](./solver_choice_guide.md)
