# Dorm Assignment GA Adaptation

이 문서는 기존 GA/NSGA-II 연구 Repo를 유지하면서, 약 90-100명의 학생을 4인 1실로 배정하는 Rule-aware Dorm Assignment 문제로 확장하기 위한 사전 감사와 설계 기록이다.

현재 단계는 구현 전 감사 및 설계 단계다. Dorm Assignment 구현 코드, 실제 학생 데이터, 개인정보 포함 파일은 생성하지 않았다.

## 목적

- 기존 GA/NSGA-II 알고리즘 자산을 삭제하지 않고 보존한다.
- Basic Mode는 해석 가능한 단일 목적 Rule-aware GA로 설계한다.
- Advanced Mode는 기존 NSGA-II 확장 경로를 보존해 추후 Pareto 후보안 비교에 활용한다.
- 4인실 capacity mismatch는 `EMPTY` placeholder를 포함한 slot-level chromosome으로 처리하도록 설계한다.

## Source-Only Baseline

이번 baseline은 source/config/docs/tests 중심으로 구성했다. 저장공간과 Git history 크기를 보호하기 위해 generated outputs, experiment artifacts, checkpoints는 삭제하지 않고 Git 추적에서 제외했다.

제외한 대표 경로:

- `.pytest-local-tmp/`
- `artifacts/`
- `outputs*/`
- `output/`
- `results/`
- `logs/`

추후 꼭 필요한 artifact 문서가 있으면 별도 branch 또는 별도 commit에서 선별적으로 포함한다.

## Mode Strategy

Basic Mode:

- Single-objective Rule-aware GA
- weighted `TotalCost`를 최소화하고 `Fitness = -TotalCost`로 기존 maximize 흐름에 맞춘다.

Advanced Mode:

- Existing NSGA-II extension preserved
- 생활지도 리스크, 반복 동호실, 학기별 정책 만족도, 운영 효율성을 분리 목적함수로 두는 후보안 비교 모드로 유지한다.
