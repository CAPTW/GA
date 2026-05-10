# Large preset decision guide

이 문서는 pure preset 관점에서 large tier 결정을 요약한다.
solver family 선택은 [solver_choice_guide.md](./solver_choice_guide.md)에서 따로 다룬다.

## Current gap recap

| Problem | Starting large preset question | Final pure-preset decision |
| --- | --- | --- |
| onemax | large에서 target hit 안정성을 다시 닫을 수 있는가 | `onemax_large.json`으로 closed |
| knapsack | large pure preset PASS를 유지할 수 있는가 | current preset 유지 |
| tsp | large pure preset PASS를 유지할 수 있는가 | current preset 유지 |
| zdt1 | large에서 metric priority를 인정하면서도 practical preset을 닫을 수 있는가 | `zdt1_large.json`으로 closed |

## Final large decisions

### onemax large

- preset: `configs/presets/onemax_large.json`
- validated size: `genome_length=128`
- why:
  - 15 seeds confirm에서 target hit rate `1.0`
  - legacy `pop80 / gen40` large shape는 closure 기준을 못 넘겼다
- decision:
  - default/fast/robust를 따로 쪼개지 않고 one-preset large로 유지

### knapsack large

- preset: `configs/presets/knapsack_large.json`
- validated size: `num_items=80`
- why:
  - large pure preset PASS를 유지했다
  - 이번 단계의 목적은 pure preset closure가 아니라 solver family hardening이었기 때문에 불필요한 retune은 하지 않았다
- note:
  - practical solver choice는 pure GA와 다를 수 있다

### tsp large

- preset: `configs/presets/tsp_large.json`
- validated size: `num_cities=50`
- why:
  - pure preset 자체는 PASS 유지
  - large hybrid candidate도 실험했지만 practical default를 바꿀 만큼의 우위는 확보하지 못했다
- note:
  - practical solver choice는 pure GA large preset과 다를 수 있다

### zdt1 large

- preset: `configs/presets/zdt1_large.json`
- validated size: `genome_length=50`
- why:
  - large에서도 같은 preset이 HV-first와 coverage/diversity-first 사이의 실사용 타협점으로 유지됐다
  - faster challenger들은 runtime은 줄였지만 HV와 pareto_ratio를 함께 깎았다
- decision:
  - balanced를 폐기하지 않고 `zdt1_large.json` 하나로 유지

## What large means

- onemax: `128`
- knapsack: `80`
- tsp: `50`
- zdt1: `50`

즉, `large`는 각 문제에서 현재 확인된 가장 큰 validated tier다.
그보다 큰 size에 대한 자동 일반화는 하지 않는다.

## Related docs

- [preset_scale_guide.md](./preset_scale_guide.md)
- [recommended_presets_by_problem.md](./recommended_presets_by_problem.md)
- [solver_choice_guide.md](./solver_choice_guide.md)
