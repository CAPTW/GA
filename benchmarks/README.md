# External Benchmark Cache

이 디렉터리는 external validity 실험에 쓰는 benchmark 메타데이터와 fetch cache를 위한 자리다.

## Provenance

- TSPLIB
  - 원출처: Reinelt, "TSPLIB -- A Traveling Salesman Problem Library" (1991)
  - 자동화 fetch 경로: `mastqe/tsplib` public mirror
  - 정책: mirror와 원출처 모두 repo-친화적인 license 고지가 명확하지 않아 raw `.tsp` 파일은 `benchmarks/cache/tsplib/` 아래 fetch-only cache로만 둔다.
- kplib
  - 원출처: `likr/kplib`
  - 생성 근거: Kellerer, Pferschy, Pisinger (2004) 계열 0/1 knapsack benchmark family
  - 라이선스: README 기준 CC BY 4.0
  - 정책: raw `.kp` 파일도 fetch cache로 내려받되, provenance와 license note를 `benchmarks/metadata.json`에 기록한다.
- Synthetic families
  - bitstring: `LeadingOnes`, `deceptive trap`
  - multi-objective: `ZDT2`, `ZDT3`
  - 별도 raw dataset은 없고 문제 정의만 로컬 구현으로 유지한다.

## Files

- `metadata.json`
  - fetch script가 쓰는 benchmark source / instance registry snapshot
- `cache/`
  - 다운로드된 raw benchmark cache

## Fetch

```bash
python scripts/fetch_benchmarks.py --dry-run
python scripts/fetch_benchmarks.py
```

## Notes

- `benchmarks/cache/`는 `.gitignore`에 포함되어 있다.
- external claim은 internal validated range와 분리해서 문서화한다.
