# Genetic Algorithm Audit Report

## 1. Executive Summary

- 현재 수준: **Level 4에 가까운 Level 3.5~4 로컬 실험 프레임워크**
- 가장 큰 강점:
  - bit / real / permutation 표현과 single-objective GA / NSGA-II / hybrid GA가 모두 실제 실행 가능하다.
  - `GAConfig`, JSON config, `history.csv`, `run_metadata.json`, local baseline snapshot, candidate ledger까지 이어지는 **재현성/거버넌스 체계**가 강하다.
  - 코어 테스트와 로컬 운영/후보 검증 테스트가 넓게 깔려 있다. 이번 감사 기준 `tests/` 아래 **40개 테스트 파일 / 265개 테스트 함수**가 존재했고, 핵심 경로 63개 + 로컬 거버넌스 31개 테스트가 모두 통과했다.
- 가장 큰 리스크:
  - `run_single_objective_ga()`와 `run_nsga2()`가 **주입된 `mutation_fn`을 실제로 사용하지 않고** `apply_adaptive_mutation()`로 우회한다. 이 때문에 mutation operator 실험/확장 경로의 신뢰도가 떨어진다.
  - 비정상 문제 함수가 `NaN` fitness를 반환해도 엔진이 fail-fast 하지 않고 **NaN 결과를 정상 summary처럼 기록**한다.
  - 병렬 평가, 일반화된 체크포인트/복구, 범용 제약조건 처리, 외부 문제 플러그인 구조는 아직 제한적이다.
- 실험 적용 가능 범위:
  - OneMax/bitstring benchmark
  - 작은/중간 규모 knapsack
  - 작은/중간 규모 TSP
  - box-bounded continuous toy function
  - ZDT1 계열의 간단한 multi-objective 실험
- 적용 비추천 범위:
  - VRP / Job-shop / Timetabling / NAS / Symbolic regression / 실시간 최적화 / 대규모 산업 최적화

이번 감사에서 **내장 지원 경로의 즉시 동작 불가(Critical) 문제는 발견되지 않았다.** 다만, 확장 실험과 실패 안전성 쪽에 **High 2건**이 있어 “좋아 보이는 GA”를 넘어서 “신뢰 가능한 실험 엔진”으로 가려면 우선 보완이 필요하다.

## 2. Repository Overview

### 저장소 구조 파악

| 항목 | 발견 위치 | 설명 | 감사상 중요도 |
|---|---|---|---|
| 코어 알고리즘 | `src/ga_lab/algorithms/` | `single_objective.py`, `nsga2.py`, `hybrid_ga.py`가 핵심 진화 루프를 담당한다. | 매우 높음 |
| 표현/연산자 | `src/ga_lab/core/` | representation, selection, crossover, mutation 플러그인 레지스트리와 구현이 있다. | 매우 높음 |
| 문제 정의 | `src/ga_lab/problems/` | OneMax, Knapsack, TSP, ZDT1 문제 정의와 metadata contract가 있다. | 매우 높음 |
| 실행 경로 | `src/ga_lab/runner.py`, `src/ga_lab/factory.py`, `src/ga_lab/consumer_cli.py` | config 검증, runtime context 구성, output artifact 작성, CLI 진입점을 담당한다. | 매우 높음 |
| 실험/비교 인프라 | `src/ga_lab/experiment/` | baseline comparison, suite loader, grid summary, reporting, retention 등 실험 집계 코드가 있다. | 높음 |
| 로컬 운영 프로토콜 | `configs/local_protocols/`, `scripts/run_local_protocol.py` | 문제별 Q/F 운영 규칙과 local protocol 의사결정 러너가 존재한다. | 높음 |
| 설정 파일 | `configs/` 전반 | smoke, presets, baselines, benchmarks, local_profiles, local_studies, local_candidates로 세분화되어 있다. | 높음 |
| 테스트 | `tests/` | 연산자/문제/러너/API/로컬 baseline/candidate/change-control까지 포괄한다. | 매우 높음 |
| 예제/데모 | `configs/demo/`, `scripts/run_demo_suite.py`, `examples/README.md` | 데모/예제 실행 경로와 운영 예시가 문서화되어 있다. | 중간 |
| 실행 결과/아티팩트 | `outputs/`, `artifacts/` | run summary, local studies, baseline snapshot, candidate ledger, reopen criteria 등이 저장된다. | 높음 |
| 문서화 | `README.md`, `docs/` | quickstart, protocol, candidate workflow, reopen criteria, benchmark guidance가 있다. | 높음 |
| 의존성/패키징 | `pyproject.toml` | 코어는 비교적 가볍고, viz/dashboard/tracking/ops는 optional extras로 분리되어 있다. | 중간 |
| 운영 서비스 | `services/` | `ga_ops` 기반 ops/ingestion 계층이 존재한다. | 중간 |

### 핵심 구조

- **문제 레지스트리**: `src/ga_lab/problems/registry.py`
- **연산자 레지스트리**: `src/ga_lab/core/selection.py`, `crossover.py`, `mutation.py`, `representation.py`
- **실험 실행**: `src/ga_lab/runner.py::run_experiment()`
- **runtime contract 검증**: `src/ga_lab/factory.py::build_runtime_context()`
- **single-objective GA**: `src/ga_lab/algorithms/single_objective.py::run_single_objective_ga()`
- **multi-objective NSGA-II**: `src/ga_lab/algorithms/nsga2.py::run_nsga2()`
- **hybrid / memetic GA**: `src/ga_lab/algorithms/hybrid_ga.py::run_hybrid_ga()`

### 실행 방법

- 단일 config 실행: `python scripts/run_experiment.py --config configs/smoke/onemax_smoke.json`
- baseline freeze 검증: `python scripts/check_local_baseline.py`
- candidate backlog 요약: `python scripts/summarize_local_candidates.py`
- 로컬 protocol 제안: `python scripts/run_local_protocol.py --problem tsp --mode final`

## 3. GA Component Audit

### 구성요소 식별 표

| 구성요소 | 구현 여부 | 구현 위치 | 구현 품질 평가 | 리스크 |
|---|---:|---|---|---|
| Binary 표현 | 예 | `src/ga_lab/core/representation.py` | 비트 초기화/repair/validate가 명확하다. | 낮음 |
| Permutation 표현 | 예 | `src/ga_lab/core/representation.py`, `src/ga_lab/problems/tsp.py` | permutation repair와 validate가 있어 TSP 해 공간 보존이 비교적 안전하다. | 낮음 |
| Real-valued 표현 | 예 | `src/ga_lab/core/representation.py` | bounds clipping/validation이 구현되어 있다. | 낮음 |
| Mixed 표현 | 아니오 | - | 내장 mixed/integer 복합 표현은 없다. | 중간 |
| Custom 표현 | 부분적 | code-level only | adapter/algorithm contract는 있으나 외부 plugin loader는 없다. | 중간 |
| 초기 개체군 생성 | 예 | `representation.py`, `hybrid_ga.py` | random init + hybrid용 seeded initializer가 있다. | 낮음 |
| 적합도 함수 인터페이스 | 예 | `src/ga_lab/problems/base.py` | scalar/multi-objective 모두 허용된다. | 낮음 |
| 선택 연산자 | 예 | `selection.py` | tournament, rank, roulette, crowded tournament 지원. | 낮음 |
| 교차 연산자 | 예 | `crossover.py` | one-point, uniform, arithmetic, order 지원. | 낮음 |
| 돌연변이 연산자 | 예 | `mutation.py` | bit_flip, gaussian, swap, inversion 지원. | **중간~높음**: GA/NSGA-II 본체에서 injected `mutation_fn`을 무시 |
| 엘리트 보존 | 예 | `single_objective.py`, `hybrid_ga.py`, `nsga2.py` | single/hybrid는 explicit elitism, NSGA-II는 survivor selection으로 보존. | 낮음 |
| 제약조건 처리 | 부분적 | `problems/knapsack.py`, `hybrid_ga.py` | knapsack penalty와 repair/local search는 있으나 범용 constraint framework는 없다. | 중간 |
| 중복 개체 처리 | 부분적 | `representation.py` | permutation repair는 있으나 generic duplicate suppression은 없다. | 중간 |
| 다양성 유지 | 예 | `adaptive_policies.py`, `convergence_diagnostics.py` | entropy/edge diversity/spread 기반 adaptive refresh가 있다. | 중간 |
| 종료 조건 | 예 | `single_objective.py`, `nsga2.py`, `_shared.py` | max generation, target fitness, plateau early stop 지원. | 낮음 |
| 랜덤 시드 제어 | 예 | `utils/seed.py`, `runner.py` | `random.Random(seed)`로 단순하고 재현 가능하다. | 낮음 |
| 세대별 로그/히스토리 | 예 | `runner.py`, algorithm files | `history.csv`, `summary.json`, `config.canonical.json`, `run_metadata.json` 생성. | 낮음 |
| 수렴 추적 | 예 | `convergence_diagnostics.py`, `_shared.py` | progress metric, diversity slope, convergence speed, hypervolume normalization 지원. | 낮음 |
| 병렬 평가 지원 | 아니오 | - | serial loop만 존재한다. | 높음(고비용 objective에서) |
| 체크포인트 저장/복구 | 부분적 | `nsga2.py`, `hybrid_ga.py`의 `_initial_population` | 숨겨진 in-memory resume만 있고 persisted checkpoint는 없다. | 중간 |
| 외부 문제 플러그인 구조 | 부분적 | `problems/registry.py` | registry 기반 확장은 가능하지만 code edit이 필요하다. | 중간 |
| baseline 비교 프레임 | 예 | `experiment/budget_baseline_comparison.py`, `scripts/run_baselines.py` | random/hill-climb/greedy/nearest-neighbor baseline까지 포함. | 낮음 |
| local governance / drift guard | 예 | `artifacts/local_baseline_snapshot.json`, `scripts/check_local_baseline.py`, candidate workflow | 실험 프레임워크 성숙도를 끌어올리는 강점. | 낮음 |

### 빠진 구성요소

- mixed / integer / tree / graph 등 **범용 표현**
- persisted checkpoint + RNG state restore
- 병렬/비동기 fitness evaluation
- generic constraint-handling policy layer (feasible-first ranking, generic repair API)
- 외부 패키지 수준의 문제/연산자 플러그인 시스템

## 4. Correctness Findings

감사 기준으로 **Critical은 발견하지 못했다.** 내장 지원 경로(OneMax / Knapsack / TSP / ZDT1 / custom Sphere)는 모두 실제 실행되었다. 다만 아래 이슈는 결과 신뢰도와 확장 실험의 해석을 왜곡할 수 있다.

| 심각도 | 문제 | 근거 위치 | 영향 | 재현 방법 | 권장 수정 방향 |
|---|---|---|---|---|---|
| High | GA/NSGA-II가 주입된 `mutation_fn`을 실제로 사용하지 않는다. | `src/ga_lab/algorithms/single_objective.py` (`del mutation_fn`, 이후 `apply_adaptive_mutation()` 호출), `src/ga_lab/algorithms/nsga2.py` 동일 패턴 | custom mutation plugin, ablation, operator override 실험이 **표면 config와 실제 실행 경로가 다를 수 있음** | 감사 probe에서 `exploding_mutation`을 넘겼는데 예외 없이 run이 완료됐다. 즉 supplied mutation contract가 무시됐다. | `mutation_fn`을 기본 경로로 사용하고, adaptive rate는 wrapper나 operator option으로 주입해야 한다. 현재처럼 algorithm 본체에서 직접 branching 하지 말 것. |
| High | 비정상 fitness(`NaN`, `inf`)를 fail-fast 하지 않는다. | `single_objective.py::evaluate_population()`, `nsga2.py::evaluate_population()`, `_shared.py::log_summary_row()` | 잘못된 문제 정의나 수치 폭주가 발생해도 엔진이 실패하지 않고 **NaN summary/history를 정상 결과처럼 남긴다** | 감사 probe에서 `NaNProblem.fitness()`가 `nan`을 반환했을 때 summary/history에 `NaN`이 기록되고 `stop_reason=max_generations`로 종료됐다. | evaluation loop에서 `math.isfinite()` 검사를 넣고, non-finite fitness는 즉시 `ValueError` 또는 실패 run 상태로 처리해야 한다. |
| Medium | checkpoint/resume 지원이 숨겨진 내부 옵션에만 부분 구현되어 있고 single-objective GA에는 없다. | `src/ga_lab/algorithms/nsga2.py`, `src/ga_lab/algorithms/hybrid_ga.py`의 `_initial_population`; single-objective에는 해당 경로 없음 | 긴 실험 재개, 장애 복구, 실험 reproducibility replay가 불완전하다. | `_initial_population` 검색 시 NSGA-II/hybrid에만 등장하고 persisted checkpoint artifact는 없다. | 공통 checkpoint schema(population, generation, RNG state, config hash)를 만들고 `run_experiment()` 차원에서 저장/복구를 지원할 것. |
| Medium | 제약조건 처리가 problem-specific penalty/repair에 묶여 있다. | `src/ga_lab/problems/knapsack.py`, `src/ga_lab/algorithms/hybrid_ga.py` | knapsack 밖의 constrained domain으로 확장할 때 실험 코드가 빨리 domain-specific hack로 흐를 수 있다. | 현재 범용 feasible-first comparator, generic repair interface, constraint violation vector가 없다. | problem contract에 제약 메타데이터/violation API를 추가하고, selection/survivor 쪽에서 reusable constraint policy를 지원할 것. |
| Low | `optimal_fitness()` 메타데이터가 일부 문제에서 실제 optimum을 의미하지 않는다. | `src/ga_lab/problems/tsp.py`는 `0.0`, `src/ga_lab/problems/knapsack.py`는 `sum(values)` 반환 | 잘못 사용하면 “이론 최적값”으로 오해할 수 있다. 현재 엔진은 이 값을 적극 활용하지 않아 직접적인 오동작은 적다. | 코드 검토로 확인 가능. | exact optimum을 모를 때는 `None`을 반환하거나 `upper_bound` / `sentinel`을 별도 필드로 분리할 것. |
| Info | 표현 repair/validate는 built-in 경로에서 해 공간을 비교적 잘 보존한다. | `src/ga_lab/core/representation.py`, `src/ga_lab/factory.py` | permutation 중복, real bound 이탈, bit coercion을 상당 부분 방지한다. | `tests/test_operators.py`, `tests/test_problems.py`에서 확인 | 현재 방향 유지. 다만 built-in mutation bypass 이슈를 먼저 수정해야 한다. |

## 5. Reproducibility and Experiment Readiness

### Software Quality Audit

| 평가 항목 | 점수 | 근거 | 개선 필요성 |
|---|---:|---|---|
| 모듈화 | 4 | `algorithms/`, `core/`, `problems/`, `experiment/`, `governance/`로 계층이 나뉘어 있다. | core/operator contract와 adaptive policy 경계를 조금 더 분리할 필요가 있다. |
| 인터페이스 명확성 | 4 | `GAConfig`, registry, `run_experiment()` 경로가 비교적 일관적이다. | mutation contract 우회 문제를 해결해야 인터페이스 신뢰도가 올라간다. |
| 테스트 가능성 | 5 | 40 test files / 265 tests, operator/problem/runner/local workflow까지 테스트된다. | 비정상 fitness / failure-mode 테스트를 더 넣으면 좋다. |
| 설정 가능성 | 5 | `configs/`가 preset/smoke/benchmarks/local_profiles/local_candidates로 잘 분리되어 있다. | mixed representation, generic constraint config는 아직 없다. |
| 타입 힌트 또는 입력 검증 | 4 | dataclass와 `GAConfig.validate()`가 잘 되어 있다. | non-finite numeric validation, hidden internal options 정리가 필요하다. |
| 에러 처리 | 3 | operator/representation/config validation은 양호하다. | NaN fail-fast, checkpoint mismatch, baseline drift messaging 개선 여지가 있다. |
| 로그/메트릭 기록 | 5 | `summary.json`, `history.csv`, `run_metadata.json`, local baseline/candidate artifacts가 풍부하다. | 대규모 실험용 schema versioning을 더 엄격히 하면 좋다. |
| 실험 재현성 | 5 | seed control, canonical config, baseline snapshot, candidate ledger, reopen criteria까지 존재한다. | persisted checkpoint가 있으면 더 좋아진다. |
| 의존성 관리 | 4 | `pyproject.toml`이 코어/옵션 의존성을 분리한다. | editable install smoke나 lockfile 계층이 있으면 더 안정적이다. |
| 문서화 | 4 | README + `docs/local_protocol_guide.md` + `docs/local_candidate_workflow.md` 등 풍부하다. | audit/benchmark-level usage guide를 더 추가할 수 있다. |
| 성능 병목 | 2 | 모든 fitness evaluation이 serial이고 caching이 없다. | 병렬 평가 훅, expensive objective용 caching 필요 |
| 메모리 사용 | 3 | 현재 규모에서는 무난하나 NSGA-II combined population/front metrics가 커질 수 있다. | 큰 pareto front, 대규모 pop에 대한 profiling 필요 |
| 대규모 population 또는 고차원 문제 대응 가능성 | 2 | small/medium local study에 최적화되어 있고 scale-out 증거가 부족하다. | 병렬화, checkpoint, large-instance benchmark 추가 필요 |

### Seed / Logging / Config / Benchmark / Result Tracking

- **의존성 / 실행 환경**
  - `pyproject.toml` 기준 코어 runtime은 비교적 가볍고, viz/dashboard/tracking/ops는 optional extras로 분리되어 있다.
  - 이번 감사에서 editable install 자체를 강제하지는 않았지만, 로컬 source mode에서 pytest/CLI/script 경로가 모두 정상 실행되어 **현 환경에서는 실행 가능성**이 확인되었다.
- **Seed 제어**
  - `src/ga_lab/utils/seed.py::make_rng()`가 `random.Random(seed)`를 사용한다.
  - `tests/test_ga_onemax.py::test_seed_reproducibility`가 동일 seed에서 summary/history 동일성을 검증한다.
- **Config 정규화/검증**
  - `src/ga_lab/config.py::normalize_config_data()`와 `GAConfig.validate()`가 alias, bounds, operator/problem compatibility를 검증한다.
- **실행 결과 기록**
  - `src/ga_lab/runner.py::run_experiment()`가 `config.json`, `config.canonical.json`, `summary.json`, `run_metadata.json`, `history.csv`를 항상 쓴다.
- **실험 비교/집계**
  - `src/ga_lab/experiment/comparison.py`, `budget_baseline_comparison.py`, `scripts/run_baselines.py`, `scripts/run_baseline_comparison.py`가 baseline 비교를 지원한다.
- **로컬 거버넌스**
  - `artifacts/local_baseline_snapshot.json`, `artifacts/local_candidate_ledger.json`, `artifacts/local_optimization_status.json`, `outputs/local_studies/future_optimization_targets.json`가 현재 운영 기준과 candidate 상태를 고정한다.

요약하면, **재현성과 실험 추적성은 이 저장소의 가장 강한 면**이다. 반면, “엔진 내부 contract가 실제로 지켜지는가?”와 “비정상 fitness를 얼마나 안전하게 거부하는가?”는 그에 비해 약하다.

## 6. Benchmark Results

### 실행한 테스트/스모크

- 코어 테스트: `python -m pytest tests/test_config.py tests/test_factory.py tests/test_operators.py tests/test_ga_onemax.py tests/test_nsga2.py tests/test_problems.py tests/test_direction_contracts.py -q`
  - **63개 테스트 통과**
- 로컬 baseline/candidate 거버넌스 테스트: `python -m pytest tests/test_local_baseline.py tests/test_local_candidate.py tests/test_local_candidate_ledger.py tests/test_local_change_request.py tests/test_local_optimization_status.py tests/test_local_maintenance_audit.py -q`
  - **31개 테스트 통과**
- baseline check: `python scripts/check_local_baseline.py`
  - **PASS**
- candidate summary: `python scripts/summarize_local_candidates.py`
  - 총 3개 candidate, ready-for-change-request 0개
- CLI smoke:
  - `python scripts/run_experiment.py --config configs/smoke/onemax_smoke.json ...`
  - `python scripts/run_experiment.py --config configs/smoke/zdt1_nsga2_smoke.json ...`
- 감사 benchmark runner:
  - `python audit/ga_execution_audit.py`

### 세부 benchmark 표

| 문제 | pop | gen | mutation rate | crossover rate | seed 수 | 핵심 metric | best | mean | std | 성공률 | 평균 실행 시간(s) | 실패/예외 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| OneMax | 40 | 40 | 0.02 | 0.90 | 10 | best fitness | 32.0000 | 32.0000 | 0.0000 | 100% | 0.1621 | 없음 |
| Sphere (custom audit problem) | 32 | 40 | 0.12 | 0.90 | 10 | best fitness (min) | 0.0034 | 0.0076 | 0.0024 | 90% (`<=1e-2`) | 0.0559 | 없음 |
| Knapsack | 40 | 30 | 0.05 | 0.90 | 10 | best feasible total value (feasible 시) | 117.6135 | 117.5725 | 0.1297 | 100% (best feasible) | 0.1951 | 없음 |
| TSP | 30 | 30 | 0.25 | 0.90 | 10 | best route distance (min) | 339.9767 | 348.9215 | 9.7604 | 100% (valid route) | 0.1833 | 없음 |
| ZDT1 (추가 multi-objective smoke) | 24 | 15 | 0.12 | 0.90 | 5 | hypervolume | 9.8596 | 9.3869 | 0.3104 | 100% (finite HV) | 0.2552 | 없음 |

### 해석용 요약 표

| 문제 | 실행 가능 여부 | 성능 요약 | 안정성 | 해석 |
|---|---|---|---|---|
| OneMax | 가능 | 10/10 seed에서 목표 32 도달 | 매우 안정적 | bitstring GA 경로는 현재 구현에서 가장 건강하다. |
| Sphere | 가능 | 10 seed 중 9 seed에서 `1e-2` 이하 도달 | 안정적 | real-coded single-objective 경로는 단순 continuous minimization에 충분히 동작한다. |
| Knapsack | 가능 | 10/10 seed에서 feasible best 확보 | 안정적 | 작은 제약 이진 문제는 현재 penalty 기반 구현으로 실험 가능하다. |
| TSP | 가능 | 유효한 permutation route 10/10, 분산은 존재 | 안정적이지만 quality variance 존재 | permutation 경로는 작동하지만 절대 성능 평가는 외부 optimum 비교가 더 필요하다. |
| ZDT1 | 가능 | hypervolume/pareto/spread 산출 정상 | 안정적 | NSGA-II 경로는 동작하나 multi-objective quality 해석에는 baseline 비교가 계속 필요하다. |

### 추가 해석

- `OneMax`는 **실행 안정성 + 수렴 속도 + 재현성** 모두 우수했다.
- `Sphere`를 별도 custom problem으로 붙였을 때도 동작했다는 점은 **실수형 single-objective 엔진 자체는 usable**하다는 근거다. 다만 이것은 registry 기반 “공식 문제 지원”이 아니라 감사용 custom hookup이다.
- `Knapsack`은 작은 문제에서 매우 안정적이었지만, 이는 현재 penalty scale과 instance가 비교적 온건했기 때문일 수 있다. 더 빡빡한 용량/구조적 제약으로 가면 별도 감사가 필요하다.
- `TSP`는 실행은 안정적이지만 seed별 편차가 눈에 띈다. 현재 로컬 protocol이 TSP를 Q/F split과 anti-case caution 중심으로 운영하는 이유와 잘 맞는다.
- `ZDT1`는 NSGA-II path가 실제 작동하고 hypervolume/pareto_ratio/spread를 정상 산출한다. 다만 이 결과만으로 범용 multi-objective engine이라고 보기엔 benchmark 폭이 좁다.

### 비교 기준 설정

| 비교 대상 | 현재 비교 가능 여부 | 필요한 작업 | 우선순위 |
|---|---|---|---|
| Random Search | 가능 | `src/ga_lab/experiment/budget_baseline_comparison.py`에 onemax `random_search`, knapsack `random_sampling`, tsp `random_tours`, zdt1 `random_archive`가 이미 있다. | 높음 |
| Hill Climbing | 부분 가능 | 현재 onemax `hill_climb`는 있으나 다른 문제군에는 공통 hill climbing baseline이 없다. | 중간 |
| Simulated Annealing | 불가 | 현재 구현/manifest가 없다. TSP/continuous용 최소 SA baseline을 추가할 필요가 있다. | 중간~높음 |
| 문제별 단순 greedy baseline | 부분 가능 | knapsack `greedy_local_search`, tsp `nearest_neighbor_2opt`가 있다. continuous / multi-objective에는 별도 greedy 정의가 필요하다. | 높음 |
| 기존 라이브러리/표준 알고리즘과의 비교 | 제한적 | external benchmark suite는 있으나 `pymoo`, `DEAP`, OR-Tools 같은 외부 표준 구현과의 직접 비교는 아직 부족하다. | 높음 |

## 7. Applicability Matrix

| 문제 영역 | 적용 등급 | 가능한 이유 | 막히는 이유 | 필요한 개선 |
|---|---|---|---|---|
| OneMax, bitstring benchmark | A | built-in problem, bit representation, target-hitting tests, benchmark 성공 | 거의 없음 | 현재 경로 유지 |
| Knapsack | A | built-in problem, penalty fitness, hybrid repair/local search, benchmark 성공 | 범용 constraint policy는 없음 | family별 baseline/constraint abstraction 보강 |
| Feature selection | B | bit representation + custom scalar fitness로 근접 가능 | dataset pipeline, sparsity/cost constraint, baseline 부재 | dataset adapter + sparse objective + baseline 추가 |
| Hyperparameter optimization | C | real representation과 bounded search는 가능 | mixed/discrete vars, expensive objective, parallel eval 부재 | mixed representation, async/parallel evaluation |
| Continuous function optimization | B | real-valued genome + arithmetic/gaussian + custom Sphere 동작 | built-in continuous benchmark가 부족 | Sphere/Rastrigin/Rosenbrock 정식 문제 등록 |
| TSP | A | built-in permutation problem, order crossover, swap/inversion, benchmark 성공 | 외부 optimum/TSPLIB 비교가 아직 제한적 | standard instance benchmark 확대 |
| Vehicle Routing Problem | D | TSP 일부 재사용 가능성은 있음 | route partition, capacity/time-window constraints, domain operators 부재 | 전용 encoding/repair/operator 설계 필요 |
| Job-shop scheduling | D | 현재 직접 지원 없음 | 표현, feasibility, specialized crossover 전부 부족 | 전용 representation/constraint handling |
| Timetabling | D | 현재 직접 지원 없음 | hard/soft constraints와 repair framework 부재 | generic constraint policy + scheduling representation |
| Portfolio optimization | C | real-valued bounded search, multi-objective 여지 | budget/cardinality/risk constraints, covariance-aware operators 부재 | constraint handling + domain fitness + baselines |
| Neural architecture search | D | 실질적 기반 없음 | graph/tree/mixed encoding, huge eval cost, parallel/distributed absent | 별도 search space/parallel infra 필요 |
| Symbolic regression | D | 실질적 기반 없음 | tree encoding/crossover/mutation 없음 | GP/tree subsystem 필요 |
| Rule discovery | C | bit/real surrogate로 단순 버전 가능 | rule semantics, mixed discrete encoding 부족 | custom representation + explainability metric |
| Multi-objective optimization | B | NSGA-II, hypervolume, spread, pareto_ratio 존재 | built-in family가 ZDT 계열 위주, benchmark 폭이 좁음 | DTLZ/WFG류와 standard baseline 추가 |
| Constrained optimization | C | knapsack penalty/repair 사례는 있음 | generic constraint API 없음 | reusable constraint comparator/repair policy |
| Real-time optimization | D | 현재는 오프라인 local study 위주 | serial eval, no checkpoint, no latency guarantee | incremental / anytime / async support |
| 대규모 산업 최적화 문제 | D | local experimentation discipline은 좋음 | scale evidence, parallelism, checkpoint, domain operators 부족 | scale benchmark + ops + domain abstraction |

## 8. Maturity Score

### 현재 판정: Level 4

판정 근거:

- 재현 가능한 실험과 비교 평가를 위한 기반(`run_experiment`, baseline comparison, local baseline snapshot, candidate ledger, reopen criteria)이 이미 존재한다.
- bit / real / permutation + single-objective / NSGA-II / hybrid 경로가 실제로 실행되고, OneMax / Knapsack / TSP / ZDT1 / custom Sphere에서 동작 증거를 확보했다.
- 다만 범용 최적화 엔진이라기보다는 **로컬 실험에 매우 강한 연구용 toolkit**이며, mutation contract와 non-finite fitness 처리 같은 engine-level 결함 때문에 “완전히 robust”하다고 보기는 아직 어렵다.

Level 4로 올리기 위한 최소 조건(현재 Level 4 하한으로 본 이유):

- 주입된 operator contract(`mutation_fn`)가 실제로 지켜져야 한다.
- non-finite fitness를 즉시 실패로 처리해야 한다.
- checkpoint/parallel/constraint abstraction 중 최소 2개 이상이 공통 계층으로 올라와야 한다.

| 항목 | 배점 | 점수 | 근거 |
|---|---:|---:|---|
| 알고리즘 완성도 | 20 | 15 | single-objective, NSGA-II, hybrid GA, bit/real/permutation, baseline comparison까지 갖췄다. 다만 mixed representation, generic constraint layer, persisted checkpoint는 없다. |
| 정확성 | 20 | 14 | built-in 경로는 실행 가능하고 permutation/bounds 보존도 양호하다. 그러나 mutation contract 무시와 NaN fail-fast 부재가 감점 요인이다. |
| 재현성 | 15 | 15 | seed control, canonical config, summary/history/run_metadata, baseline snapshot, candidate ledger까지 있다. |
| 확장성 | 15 | 9 | registry 구조는 있으나 code-edit 기반이고, 병렬/외부 plugin/mixed encoding이 부족하다. |
| 실험 설계 준비도 | 15 | 13 | baseline comparison, local studies, protocol matrix, candidate workflow가 강하다. 다만 표준 external benchmark와 library comparison은 더 필요하다. |
| 코드 품질 | 10 | 8 | 모듈 분리가 좋고 테스트도 풍부하다. 다만 adaptive path가 operator contract를 침범한다. |
| 문서화 | 5 | 4 | README/docs/examples가 풍부하다. audit 관점의 usage guide는 더 보완 가능하다. |
| 총점 | 100 | 78 | 연구용 실험 프레임워크 후보 수준 |

총점 해석:

- 0~20: 아이디어 수준
- 21~40: toy prototype
- 41~60: 제한적 실험 가능
- 61~75: 일반 benchmark 실험 가능
- 76~90: 연구용 실험 프레임워크 후보
- 91~100: 고신뢰 범용/제품화 후보

## 9. Go / No-Go Decision

- **Toy benchmark 적용: Go**
  - 근거: OneMax / Sphere / Knapsack / TSP / ZDT1 모두 실제 실행되었고, core tests 63개가 통과했다.
- **연구용 benchmark 적용: Conditional Go**
  - 근거: baseline comparison, experiment tracking, local protocol/candidate governance가 이미 있다.
  - 조건: mutation contract와 NaN fail-fast를 먼저 고치고, external/standard benchmark 비교를 추가할 것.
- **실제 도메인 문제 적용: Conditional Go**
  - 근거: small/medium TSP, knapsack-like, bounded continuous, ZDT-like 실험은 가능하다.
  - 조건: 문제별 baseline, domain-specific feasibility handling, non-regression protocol을 함께 둘 것.
- **제품 또는 자동 의사결정 적용: No-Go**
  - 근거: 병렬성/체크포인트/실패 안전성/범용 제약 처리/대규모 검증이 부족하고, 현재는 어디까지나 local experimentation toolkit이다.

## 10. Prioritized Action Plan

### 1단계: Sanity Benchmark

- 목적: bit / real / permutation / multi-objective 핵심 경로의 engine sanity를 계속 확인
- 문제:
  - OneMax
  - Sphere
  - Small Knapsack
  - Small TSP
  - ZDT1
- 성공 기준:
  - no exception
  - same seed reproducibility
  - history/summary artifact 생성
  - non-finite fitness fail-fast
- 필요한 코드:
  - 현재 작성한 `audit/ga_execution_audit.py` 유지
  - non-finite fitness negative test 추가

### 2단계: Standard Benchmark

- 목적: “작동한다”를 넘어 “의미 있다”를 baseline과 비교
- 문제:
  - OneMax / LeadingOnes / Trap / Jump
  - Knapsack small/medium
  - TSPLIB small/medium
  - ZDT1/ZDT2/ZDT3
- baseline:
  - Random Search
  - Hill Climbing
  - Greedy / nearest-neighbor + 2opt
  - library baseline (예: `pymoo` NSGA-II, `DEAP`, 혹은 OR-Tools/문제별 표준)
- metric:
  - success rate
  - evaluations to target
  - best distance / hypervolume / feasible fitness
  - runtime
- 반복 횟수:
  - 최소 10 seed, 표준 benchmark는 20 seed 이상 권장
- 성공 기준:
  - baseline 대비 명확한 gain 또는 최소한 honest tradeoff 설명 가능

### 3단계: Target Domain Experiment

- 적용 가능한 실제 문제 후보:
  - small/medium TSP-like routing
  - small constrained selection/packing
  - bounded continuous parameter tuning
- 데이터 요구사항:
  - reproducible instance file
  - train/holdout split 또는 seed block
  - problem-specific baseline
- 위험 요소:
  - domain-specific operator 부재
  - constraint 처리의 일반화 부족
  - expensive objective에서 serial eval bottleneck
- 성공 기준:
  - candidate가 baseline snapshot을 이기고
  - non-regression/stress slice를 통과하며
  - change-request pack을 만들 수 있을 것
- 중단 기준:
  - same micro-tuning 반복
  - baseline drift 미해결
  - 새 mechanism hypothesis 없이 frozen target 재개

### 우선순위 작업 목록

| 우선순위 | 작업 | 이유 | 예상 효과 | 난이도 |
|---:|---|---|---|---|
| 1 | `mutation_fn` contract 복구 (`single_objective.py`, `nsga2.py`) | 현재 operator 실험 신뢰도를 직접 훼손하는 High 이슈 | custom mutation/ablation 결과 신뢰 회복 | 중간 |
| 2 | non-finite fitness fail-fast 추가 | NaN 결과가 조용히 summary에 남는 것은 감사상 큰 리스크 | 잘못된 문제 정의를 즉시 차단 | 낮음 |
| 3 | 공통 checkpoint/resume 설계 및 persisted artifact 추가 | long run 복구와 재현성 보강 | 실험 신뢰도/운영성 향상 | 중간 |
| 4 | 병렬 evaluation hook 또는 evaluator abstraction 도입 | expensive objective, 실험 확장성의 핵심 병목 | 실제 도메인 적용 범위 확대 | 중간~높음 |
| 5 | generic constraint handling layer 추가 | knapsack 밖 constrained domain 확장을 막는 핵심 제한 | feature selection / portfolio / scheduling 실험 토대 확보 | 높음 |
| 6 | standard baseline/library comparison 추가 | 현재는 자체 baseline은 있으나 외부 표준 대비 설득력이 아직 약함 | 연구용 benchmark 신뢰도 상승 | 중간 |
| 7 | built-in continuous benchmark(Sphere/Rastrigin 등) 정식 등록 | 현재 Sphere는 audit용 custom problem으로만 확인됨 | continuous GA 지원 증거 강화 | 낮음 |
| 8 | external plugin / mixed representation 전략 정리 | 확장성/적용 영역 확장을 위해 필요 | HPO, richer domains 진입 가능 | 높음 |

마지막 한 문장 요약:

**“현재 이 GA는 로컬 연구용 실험 프레임워크 후보 수준이며, OneMax/Knapsack/TSP/ZDT1 및 단순 continuous toy 영역까지는 실험 적용 가능하지만, VRP·스케줄링·대규모 산업 최적화 영역에 적용하기 전에는 mutation contract 복구, non-finite fitness fail-fast, checkpoint/constraint/parallel 평가 개선이 필요하다.”**
