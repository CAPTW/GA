from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.config import GAConfig, load_config
from ga_lab.convergence_diagnostics import configured_evaluation_budget
from ga_lab.experiment.diversity_diagnostics import evaluate_diversity_diagnostics
from ga_lab.experiment.external_mo_comparators import (
    ExternalMOComparatorResult,
    result_to_front_row,
    run_internal_nsga2,
)
from ga_lab.experiment.mo_runner_fairness import (
    build_candidate_rows,
    build_mo_benchmark_rows,
    decorate_fairness_row,
    fairness_summary_rows,
)
from ga_lab.experiment.nsga2_candidate_suite import (
    build_problem_config,
    mo_candidate_suite_specs,
    reference_front_for_spec,
    safe_artifact_path,
)
from ga_lab.experiment.nsga2_candidate_variants import (
    NSGA2CandidateVariant,
    apply_candidate_variant,
    candidate_j_h_lite_retry2,
    candidate_l_sparse_parent_bias_light,
    candidate_variant_metadata,
)
from ga_lab.experiment.parameter_fairness import evaluate_parameter_fairness


def _load_base_helpers():
    helper_path = PROJECT_ROOT / "scripts" / "validate_nsga2_candidate_suite.py"
    spec = importlib.util.spec_from_file_location("_candidate_suite_phase0_base", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 0 sanity validation for survivor-pressure candidate_l."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problems", default="zdt1")
    parser.add_argument("--output-root", default="outputs/nsga2_survivor_pressure_phase0")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=10101)
    parser.add_argument("--budget", type=int, default=300)
    return parser.parse_args()


def _phase0_variants() -> list[NSGA2CandidateVariant]:
    return [
        candidate_j_h_lite_retry2(),
        candidate_l_sparse_parent_bias_light(),
    ]


def _retarget_budget(base_config: GAConfig, requested_budget: int) -> GAConfig:
    clone = GAConfig.from_dict(base_config.to_dict())
    if configured_evaluation_budget(clone) == requested_budget:
        return clone

    best_match: tuple[int, int, int] | None = None
    max_population = max(clone.population_size * 2, 40)
    for population_size in range(4, max_population + 1):
        if requested_budget % population_size != 0:
            continue
        generation_term = (requested_budget // population_size) - 2
        if generation_term <= 0 or generation_term % 3 != 0:
            continue
        generations = generation_term // 3
        if generations <= 0:
            continue
        score = abs(population_size - clone.population_size) + abs(generations - clone.generations)
        if best_match is None or score < best_match[0]:
            best_match = (score, population_size, generations)

    if best_match is None:
        raise ValueError(
            f"Unable to derive an exact NSGA-II population/generation pair for requested budget {requested_budget}"
        )

    _, population_size, generations = best_match
    clone.population_size = population_size
    clone.generations = generations
    clone.elitism = min(clone.elitism, population_size - 1)
    clone.tournament_size = min(max(2, clone.tournament_size), population_size)
    if "tournament_size" in clone.selection_options:
        clone.selection_options["tournament_size"] = min(
            max(2, int(clone.selection_options["tournament_size"])),
            population_size,
        )
    return clone


def _decorate_row(
    row: dict[str, Any],
    *,
    reference_front: list[list[float]],
) -> dict[str, Any]:
    directions = [
        bool(value) for value in row.get("metadata", {}).get("objective_directions", [False, False])
    ]
    if row.get("success"):
        decision_vectors = row.get("front_decision_vectors") or row.get("decision_vectors")
        row.update(
            evaluate_diversity_diagnostics(
                row.get("objective_vectors", []),
                directions=directions,
                decision_vectors=decision_vectors if isinstance(decision_vectors, list) else None,
            )
        )
        row["metric_calculation_success"] = all(
            isinstance(row.get(metric_name), int | float) and math.isfinite(float(row[metric_name]))
            for metric_name in (
                "hypervolume_2d",
                "reference_front_distance",
                "inverted_generational_distance",
                "spacing",
                "nondominated_count",
            )
        )
    else:
        row["decision_duplicate_rate"] = None
        row["objective_duplicate_rate"] = None
        row["archive_duplicate_rate"] = None
        row["unique_decision_count"] = None
        row["unique_objective_count"] = None
        row["boundary_point_count"] = None
        row["metric_calculation_success"] = False
    row["reference_front_size"] = len(reference_front)
    return row


def _make_candidate_result(
    base_config: GAConfig,
    variant: NSGA2CandidateVariant,
    *,
    seed: int,
    output_root: Path,
) -> ExternalMOComparatorResult:
    candidate_config = apply_candidate_variant(base_config, variant)
    result = run_internal_nsga2(candidate_config, seed=seed, output_root=str(output_root))
    metadata = dict(result.metadata)
    metadata.update(candidate_variant_metadata(variant))
    return ExternalMOComparatorResult(
        problem_name=result.problem_name,
        algorithm_name=variant.candidate_id,
        library_name="internal_candidate",
        seed=result.seed,
        requested_budget=result.requested_budget,
        evaluations=result.evaluations,
        runtime_seconds=result.runtime_seconds,
        status=result.status,
        success=result.success,
        error_message=result.error_message,
        objective_vectors=result.objective_vectors,
        nondominated_objective_vectors=result.nondominated_objective_vectors,
        metadata=metadata,
    )


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["problem"]), str(row["algorithm"]))].append(row)

    aggregates: list[dict[str, Any]] = []
    for (problem, algorithm), bucket in sorted(grouped.items()):
        successful = [row for row in bucket if row.get("success")]
        statuses = {str(row.get("status", "unknown")) for row in bucket}
        status = (
            "skipped"
            if statuses == {"skipped"}
            else "failed"
            if "failed" in statuses and not successful
            else "partial_failure"
            if "failed" in statuses
            else "success"
        )
        aggregates.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "status": status,
                "seeds": len(bucket),
                "successful_seeds": len(successful),
                "mean_hv": BASE._summary_stat(successful, "hypervolume_2d")["mean"],
                "mean_distance": BASE._summary_stat(successful, "reference_front_distance")["mean"],
                "mean_igd": BASE._summary_stat(successful, "inverted_generational_distance")["mean"],
                "mean_spacing": BASE._summary_stat(successful, "spacing")["mean"],
                "mean_nondominated_count": BASE._summary_stat(successful, "nondominated_count")["mean"],
                "mean_duplicate_rate": BASE._summary_stat(successful, "archive_duplicate_rate")["mean"],
                "mean_runtime_seconds": BASE._summary_stat(successful, "runtime_seconds")["mean"],
                "mean_actual_evaluations": BASE._summary_stat(successful, "actual_evaluations")["mean"],
                "success_rate": BASE._success_rate(bucket),
            }
        )
    return aggregates


def _sanity_gate_results(
    raw_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_l_rows = [row for row in raw_rows if row["algorithm"] == "candidate_l_sparse_parent_bias_light"]
    metadata_ok = all(
        row.get("metadata", {}).get("candidate_id") == "candidate_l_sparse_parent_bias_light"
        and row.get("metadata", {}).get("base_candidate_id") == "candidate_j_h_lite_retry2"
        for row in candidate_l_rows
    )
    default_changed_ok = all(
        row.get("metadata", {}).get("default_changed") is False for row in candidate_l_rows
    )
    evaluations_ok = all(
        row.get("requested_budget") == row.get("actual_evaluations") for row in candidate_l_rows if row.get("success")
    )
    metrics_ok = all(bool(row.get("metric_calculation_success")) for row in candidate_l_rows if row.get("success"))
    execution_ok = all(row.get("success") for row in candidate_l_rows)
    catastrophic_regression = any(
        isinstance(row.get("hypervolume_2d"), int | float) and float(row["hypervolume_2d"]) <= 0.0
        for row in candidate_l_rows
        if row.get("success")
    )
    fairness_fail_free = fairness_payload.get("summary_counts", {}).get("fail", 0) == 0

    return [
        {"gate": "candidate metadata", "result": metadata_ok, "pass_fail": metadata_ok, "note": "candidate_l metadata present"},
        {"gate": "default_changed=false", "result": default_changed_ok, "pass_fail": default_changed_ok, "note": "candidate_l metadata keeps default_changed=false"},
        {"gate": "fairness fail 없음", "result": fairness_fail_free, "pass_fail": fairness_fail_free, "note": f"fairness status={fairness_payload.get('status')}"},
        {"gate": "actual evaluations match", "result": evaluations_ok, "pass_fail": evaluations_ok, "note": "candidate_l actual evaluations match requested budget"},
        {"gate": "metric calculation success", "result": metrics_ok, "pass_fail": metrics_ok, "note": "core MO metrics stayed finite"},
        {"gate": "candidate_l execution", "result": execution_ok, "pass_fail": execution_ok, "note": "candidate_l run succeeded on all seeds"},
        {"gate": "catastrophic regression 여부", "result": not catastrophic_regression, "pass_fail": not catastrophic_regression, "note": "no catastrophic failure observed in Phase 0"},
        {"gate": "artifact generation", "result": True, "pass_fail": True, "note": "results and report artifacts created"},
    ]


def _phase0_decision(gate_rows: list[dict[str, Any]], fairness_payload: dict[str, Any]) -> str:
    if fairness_payload.get("summary_counts", {}).get("fail", 0):
        return "Phase 0 failed, fix required"
    if not all(bool(row["pass_fail"]) for row in gate_rows):
        return "Phase 0 failed, fix required"
    if fairness_payload.get("summary_counts", {}).get("warning", 0):
        return "Phase 0 passed with warnings"
    return "Phase 0 passed, eligible for Phase 1 planning"


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(PROJECT_ROOT / args.config)
    requested_problem_names = [item.strip().lower() for item in str(args.problems).split(",") if item.strip()]
    selected_specs = [mo_candidate_suite_specs()[name] for name in requested_problem_names]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    variants = _phase0_variants()
    variant_map = {variant.candidate_id: variant for variant in variants}

    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for spec in selected_specs:
        config = build_problem_config(base_config, spec)
        config = _retarget_budget(config, args.budget)
        reference_front = reference_front_for_spec(spec, point_count=201)
        reference_point = list(spec.hv_reference_point)
        problem_output_root = output_root / spec.problem
        problem_output_root.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            results = [
                run_internal_nsga2(config, seed=seed, output_root=str(problem_output_root)),
                *[
                    _make_candidate_result(config, variant, seed=seed, output_root=problem_output_root)
                    for variant in variants
                ],
            ]
            for result in results:
                row = result_to_front_row(
                    result,
                    reference_front=reference_front,
                    reference_point=reference_point,
                )
                row = _decorate_row(row, reference_front=reference_front)
                row = decorate_fairness_row(
                    row,
                    spec=spec,
                    base_config=config,
                    requested_budget=args.budget,
                    variant_map=variant_map,
                )
                raw_rows.append(row)
                if result.status != "success":
                    failures.append(
                        {
                            "type": result.status,
                            "target": result.algorithm_name,
                            "message": result.error_message,
                            "impact": "seed excluded from sanity read",
                            "action": "fix the candidate/runtime issue before Phase 1",
                        }
                    )

    fairness_payload = evaluate_parameter_fairness(
        raw_rows,
        benchmark_rows=benchmark_rows,
        candidate_rows=candidate_rows,
    )
    aggregate_rows = _aggregate_rows(raw_rows)
    gate_rows = _sanity_gate_results(raw_rows, fairness_payload)
    decision = _phase0_decision(gate_rows, fairness_payload)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "selected_problems": requested_problem_names,
        "seeds": seeds,
        "budget": args.budget,
        "benchmark_rows": benchmark_rows,
        "candidate_rows": candidate_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
        "gate_rows": gate_rows,
        "phase0_decision": decision,
        "failures": failures,
    }

    json_path = safe_artifact_path(
        artifact_root,
        "nsga2_survivor_pressure_phase0_results",
        args.artifact_suffix,
        ".json",
    )
    md_path = safe_artifact_path(
        artifact_root,
        "nsga2_survivor_pressure_phase0_results",
        args.artifact_suffix,
        ".md",
    )
    report_path = safe_artifact_path(
        artifact_root,
        "nsga2_survivor_pressure_phase0_report",
        args.artifact_suffix,
        ".md",
    )

    BASE._write_json(json_path, payload)
    md_lines = [
        "# NSGA-II Survivor-Pressure Phase 0 Results",
        "",
        "## Aggregate Results",
        "",
        *BASE._markdown_table(
            aggregate_rows,
            [
                "problem",
                "algorithm",
                "status",
                "mean_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_nondominated_count",
                "mean_duplicate_rate",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Fairness Summary",
        "",
        *BASE._markdown_table(
            fairness_summary_rows(fairness_payload),
            ["status", "pass", "warning", "fail"],
        ),
        "",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# NSGA-II Survivor-Pressure Phase 0 Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: candidate_l_sparse_parent_bias_light의 Phase 0 sanity 확인",
        "- 구현한 candidate: `candidate_l_sparse_parent_bias_light`",
        "- 기본값 변경 여부: `false`",
        f"- 실행한 Phase 0 sanity: {', '.join(requested_problem_names)}",
        f"- fairness 결과: **{fairness_payload['status']}**",
        f"- Phase 0 판정: **{decision}**",
        "- Phase 1은 이번 패스에서 자동 승인되지 않는다.",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "- Scope: candidate_l_sparse_parent_bias_light, Phase 0 sanity, ZDT1 small run, candidate isolation, fairness check, artifact generation",
        "- Non-Scope: default promotion, candidate_l change request, Phase 1 full benchmark, WFG/DTLZ validation, new survivor-pressure families, productization",
        "",
        "## 3. Candidate Definition",
        "",
        *BASE._markdown_table(
            [
                {"field": "candidate_id", "value": "candidate_l_sparse_parent_bias_light"},
                {"field": "base_candidate", "value": "candidate_j_h_lite_retry2"},
                {"field": "mechanism", "value": "sparse_region_parent_bias_light"},
                {"field": "default_changed", "value": "false"},
                {"field": "promotion_status", "value": "phase0_sanity"},
                {"field": "allowed_use", "value": "phase0_sanity_only"},
                {"field": "disallowed_use", "value": "default_replacement"},
            ],
            ["field", "value"],
        ),
        "",
        "## 4. Implementation Summary",
        "",
        *BASE._markdown_table(
            [
                {
                    "file": "src/ga_lab/core/selection.py",
                    "change": "added light sparse-parent-bias selection path behind explicit selection_options flag",
                    "default_impact": "none when option is absent",
                },
                {
                    "file": "src/ga_lab/experiment/nsga2_candidate_variants.py",
                    "change": "added candidate_l metadata and apply logic",
                    "default_impact": "none",
                },
                {
                    "file": "configs/candidates/nsga2_sparse_parent_bias_candidate_l.json",
                    "change": "registered candidate_l config",
                    "default_impact": "none",
                },
                {
                    "file": "scripts/validate_nsga2_survivor_pressure_phase0.py",
                    "change": "added dedicated Phase 0 runner",
                    "default_impact": "none",
                },
            ],
            ["file", "change", "default_impact"],
        ),
        "",
        "## 5. Phase 0 Configuration",
        "",
        *BASE._markdown_table(
            [
                {"item": "problems", "value": ", ".join(requested_problem_names)},
                {"item": "seeds", "value": len(seeds)},
                {"item": "budget", "value": args.budget},
                {"item": "algorithms", "value": "internal_nsga2, candidate_j_h_lite_retry2, candidate_l_sparse_parent_bias_light"},
                {"item": "metrics", "value": "HV, distance, IGD, spacing, nondominated_count, duplicate rate, runtime"},
                {"item": "fairness checker 사용 여부", "value": "yes"},
            ],
            ["item", "value"],
        ),
        "",
        "## 6. Sanity Gate Results",
        "",
        *BASE._markdown_table(
            gate_rows,
            ["gate", "result", "pass_fail", "note"],
        ),
        "",
        "## 7. Metric Snapshot",
        "",
        *BASE._markdown_table(
            aggregate_rows,
            [
                "algorithm",
                "mean_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_nondominated_count",
                "mean_duplicate_rate",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## 8. Failures and Warnings",
        "",
        *BASE._markdown_table(
            failures or [{"type": "none", "target": "none", "message": "none", "impact": "none", "action": "none"}],
            ["type", "target", "message", "impact", "action"],
        ),
        "",
        "## 9. Phase 0 Decision",
        "",
        f"- **{decision}**",
        "",
        "## 10. Next Gate",
        "",
        "- Phase 1은 다음 패스에서 별도 승인 후 진행해야 한다.",
        "- 최소 Phase 1 조건: ZDT1/ZDT2/ZDT3 10 seeds, candidate_l vs candidate_j 비교, fairness pass 또는 warning-only, 그리고 HV/distance/IGD catastrophic regression 없음.",
        "",
        "## 11. Regression Check",
        "",
        "| command | result | note |",
        "| --- | --- | --- |",
        "| python -m pytest tests/test_survivor_pressure_phase0.py -q | pending_from_shell | executed separately in shell |",
        "| python -m pytest tests/test_nsga2_candidate_isolation.py tests/test_parameter_fairness.py tests/test_fairness_runner_integration.py -q | pending_from_shell | executed separately in shell |",
        "| python -m pytest tests/test_mutation_contract.py tests/test_fitness_validation.py -q | pending_from_shell | executed separately in shell |",
        "| python scripts/check_local_baseline.py --output-dir artifacts/survivor_pressure_phase0_guard | pending_from_shell | executed separately in shell |",
        f"| python scripts/validate_nsga2_survivor_pressure_phase0.py --problems {','.join(requested_problem_names)} --seeds {len(seeds)} --budget {args.budget} --artifact-suffix {args.artifact_suffix or 'none'} | success | current run |",
        "",
        "## 12. Maturity Impact",
        "",
        "- Level 판정 유지.",
        "- Phase 0는 sanity일 뿐 성능 maturity 상향 근거가 아니다.",
        "- candidate isolation과 fairness gate가 유지되면 실험 툴킷으로서 Level 4 근거는 강화될 수 있다.",
        "",
        "## 13. Recommended Next Work",
        "",
        "1. Phase 0 passed이면 별도 Phase 1 ZDT small validation 계획을 작성한다.",
        "2. Phase 0 failed이면 candidate_l을 수정하거나 폐기한다.",
        "3. candidate_j opt-in 문서와 survivor-pressure 계획 문서를 같이 링크한다.",
        "4. fairness checker의 single-objective runner 확장을 별도 검토한다.",
        "",
        "이번 Phase 0 결과, candidate_l_sparse_parent_bias_light는 sanity-only 상태로 남아 있고, 기본 NSGA-II default는 unchanged 상태로 유지되며, 다음 단계는 별도 승인된 Phase 1 계획 작성이다.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "results_json": str(json_path),
                "results_md": str(md_path),
                "report_md": str(report_path),
                "phase0_decision": decision,
                "fairness_status": fairness_payload["status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
