# Reproducibility And Governance

이 문서는 이 저장소의 benchmark, summary, claim이 시간이 지나도 다시 만들 수 있고, drift가 생기면 자동으로 드러나도록 하는 운영 규칙을 정리합니다.

핵심 원칙:

- validated range 밖으로 일반화하지 않습니다.
- 1차 공정성 기준은 function evaluation budget입니다.
- external benchmark provenance / fetch / redistribution policy를 문서와 metadata에 함께 남깁니다.
- 공식 claim이 깨지면 문서를 먼저 믿지 않고 drift report를 `WARN` 또는 `FAIL`로 남깁니다.

## Evidence Scope

| Scope | Meaning |
| --- | --- |
| Internal official | 현재 repo의 validated internal range에서 공식으로 유지하는 claim |
| External supported | public / canonical benchmark subset에서 재현된 claim |
| Family-conditional external | 전체 family 일반론은 아니지만, 특정 tested family에서는 성립하는 claim |
| Experimental / insufficient | 신호는 있으나 promotion하기엔 범위가 좁거나 결과가 혼재한 claim |

## Machine-Readable Sources Of Truth

- claim registry: `claims/claim_registry.json`
- release artifact manifest: `configs/release/release_artifacts_manifest.json`
- internal / external summary artifacts:
  - `outputs/benchmark_summary/baseline_comparison_summary.json`
  - `outputs/benchmark_summary/hybrid_comparison_summary.json`
  - `outputs/benchmark_summary/ablation_summary.json`
  - `outputs/benchmark_summary/external_benchmark_summary.json`
  - `outputs/benchmark_summary/external_family_summary.json`
- drift artifacts:
  - `outputs/benchmark_summary/claim_drift_report.json`
  - `outputs/benchmark_summary/claim_drift_report.md`
  - `outputs/benchmark_summary/run_metadata.json`
- public-facing release artifacts:
  - `artifacts/claim_matrix.json`
  - `artifacts/solver_matrix.json`
  - `artifacts/release_snapshot.json`

README와 solver-choice 문서는 이 registry와 summary가 말할 수 있는 범위 안에서만 유지합니다.

## Benchmark Tiers

| Tier | Purpose | Included manifests / configs | Runtime class | Cadence | Artifact outputs | Failure policy |
| --- | --- | --- | --- | --- | --- | --- |
| Tier 0 | registry/schema validation, drift-check logic, governance smoke | `tests/test_claim_governance.py`, `scripts/check_claim_drift.py` against checked-in summaries | seconds | pre-commit / PR | `claim_drift_report.json`, `claim_drift_report.md`, `run_metadata.json` | `ci_gated` claim `FAIL`이면 실패 |
| Tier 1 | fast runner smoke | `configs/ci/baseline_smoke.json`, `scripts/fetch_benchmarks.py --dry-run` | low minutes | PR | smoke outputs, `ci_benchmark_plan.json` | runner 실패 시 실패 |
| Tier 2 | internal validated regression subset | `configs/benchmarks/ablation_manifest.json` | minutes | nightly | `ablation_summary.*`, drift report | internal official claim drift가 `FAIL`이면 실패 |
| Tier 3 | external support regression subset | `configs/benchmarks/external_family_manifest.json` | tens of minutes | manual / release-candidate | `external_family_summary.*`, drift report | external / family-conditional `ci_gated` claim drift가 `FAIL`이면 실패 |
| Tier 4 | deeper refresh for release-facing evidence | `baseline_confirm_manifest.json`, `hybrid_confirm_manifest.json`, `ablation_confirm_manifest.json`, `external_confirm_manifest.json`, `external_family_confirm_manifest.json` | heavy | weekly / manual | refreshed summaries + drift report | `ci_gated` claim `FAIL`이면 실패, report-only claim은 문서 검토용 경고로 남김 |

실행 helper:

```bash
python scripts/run_ci_benchmarks.py --tier tier0 --dry-run
python scripts/run_ci_benchmarks.py --tier tier1
python scripts/run_ci_benchmarks.py --tier tier2
python scripts/run_ci_benchmarks.py --tier tier3
python scripts/run_ci_benchmarks.py --tier tier4
```

## Reproducibility Inventory

현재 summary artifact와 drift artifact에는 아래 재현 정보가 같이 남습니다.

- `summary_schema_version`
- `run_metadata`
- Python version / executable / implementation
- OS / platform snapshot
- git SHA / branch / working tree status, 가능한 경우
- manifest SHA256, size, suite name, entry count
- benchmark metadata snapshot (`benchmarks/metadata.json`)
- installed package version snapshot
- output root / summary stem
- run-specific extra note (`suite_kind`, row count 등)

추가 companion file:

- `<summary_stem>_run_metadata.json`
- `outputs/benchmark_summary/run_metadata.json` for drift-check runs

## Claim Registry Shape

각 claim 항목은 최소 아래 필드를 가집니다.

- `claim_id`
- `label`
- `status`
- `evidence_scope`
- `problem_family`
- `validated_ranges`
- `comparators`
- `metrics`
- `pass_condition`
- `warning_threshold`
- `source_summary_paths`
- `doc_locations`
- `notes`
- `governance.mode`
- `governance.tier`
- `checks[]`

`checks[]`는 요약 파일의 특정 row를 읽어 `PASS / WARN / FAIL / NOT_EVALUATED`를 계산합니다.

예:

- `tsp_external_nn2opt_default`
- `zdt_external_nsga2_over_random_archive`
- `bitstring_monotone_hill_climb_default`
- `knapsack_correlated_seed_repair_worth_trying`
- `tsp_medium_hybrid_internal_quality_first`

## Drift Checker

drift checker entrypoint:

```bash
python scripts/check_claim_drift.py
python scripts/render_release_artifacts.py
```

주요 입력:

- `--registry`
- `--summary`
- `--summary-dir`
- `--reference-summary`
- `--output-json`
- `--output-md`
- `--output-metadata`
- `--fail-on`

판정 규칙:

- `PASS`: registry의 `pass_if` 조건을 만족
- `WARN`: `pass_if`는 깨졌지만 `warn_if`는 만족, 또는 일부 필요한 row가 빠짐
- `FAIL`: 공식 claim의 pass condition이 깨짐
- `NOT_EVALUATED`: 필요한 summary 또는 row가 없음

CI에서 기본적으로 `ci_gated` claim만 실패 기준에 포함하고, `report_only` claim은 문서 검토용으로만 남깁니다.

## CI And Automation

현재 automation 경로:

- PR CI: `.github/workflows/ci.yml`
  - lint / typecheck / tests
  - baseline regression
  - `tier1` benchmark smoke
  - `tier0` governance checks
- scheduled/manual governance: `.github/workflows/benchmark-governance.yml`
  - nightly: `tier2`
  - weekly: `tier4`
  - manual input: `tier2`, `tier3`, `tier4`

heavy external suite를 항상 PR에서 돌린다고 쓰지 않습니다. 비용이 큰 tier는 schedule 또는 수동 실행으로 분리합니다.

## External Benchmark Provenance

external benchmark source-of-truth:

- `benchmarks/metadata.json`
- `benchmarks/README.md`

정책:

- TSPLIB raw file는 fetch-only cache로만 두고 커밋하지 않습니다.
- kplib도 provenance와 cache path를 metadata에 남깁니다.
- redistribution이 애매한 source는 raw file를 canonical artifact로 다루지 않습니다.
- summary와 docs에는 어떤 instance / family를 썼는지 명시합니다.

## Reading The Reports

실무적으로는 이렇게 읽으면 됩니다.

1. README / solver guide에서 현재 추천 규칙을 봅니다.
2. `claim_drift_report.md`에서 그 규칙이 아직 `PASS`인지 확인합니다.
3. `*_run_metadata.json`에서 어떤 환경, 어떤 manifest, 어떤 benchmark metadata로 돌았는지 확인합니다.
4. `WARN` 또는 `FAIL`이 있으면 해당 claim의 `doc_locations`를 먼저 검토합니다.

## Related Docs

- [Solver choice guide](solver_choice_guide.md)
- [External validity](external_validity.md)
- [External family solver guide](external_family_solver_guide.md)
- [Ablation and claims freeze](ablation_and_claims.md)
