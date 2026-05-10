from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.config import GAConfig, load_config
from ga_lab.experiment.external_mo_comparators import (
    ExternalMOComparatorResult,
    result_to_front_row,
    run_deap_nsga2,
    run_internal_nsga2,
    run_pymoo_nsga2,
    run_random_archive_anchor,
)
from ga_lab.experiment.mo_baselines import run_random_pareto_archive
from ga_lab.experiment.mo_metrics import coverage_indicator
from ga_lab.experiment.mo_runner_fairness import (
    build_candidate_rows,
    build_mo_benchmark_rows,
    fairness_summary_rows,
)
from ga_lab.experiment.nsga2_candidate_suite import (
    MOBenchmarkSpec,
    build_problem_config,
    mo_candidate_suite_specs,
    reference_front_for_spec,
    safe_artifact_path,
)
from ga_lab.experiment.nsga2_candidate_variants import (
    NSGA2CandidateVariant,
    candidate_d_uniform_crossover,
    candidate_h_uniform_dedup_mutation_boost,
    candidate_j_h_lite_retry2,
    candidate_l_sparse_parent_bias_light,
    candidate_m_boundary_preservation_light,
)
from ga_lab.experiment.parameter_fairness import evaluate_parameter_fairness


def _load_phase1_helpers():
    helper_path = PROJECT_ROOT / "scripts" / "validate_nsga2_survivor_pressure_phase1.py"
    spec = importlib.util.spec_from_file_location("_survivor_pressure_phase1_boundary_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE1 = _load_phase1_helpers()
BASE = PHASE1.BASE
METRIC_SPECS = PHASE1.METRIC_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 ZDT repeated-seed validation for boundary-preservation candidate_m."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problems", default="zdt1,zdt2,zdt3")
    parser.add_argument("--output-root", default="outputs/nsga2_boundary_phase1")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=10401)
    parser.add_argument("--budget", type=int, default=760)
    parser.add_argument(
        "--drift-audit-json",
        default="artifacts/nsga2_candidate_m_phase1_default_drift_audit_results.json",
        help="Optional drift-audit JSON included in the Phase 1 report if it exists.",
    )
    parser.add_argument(
        "--local-baseline-status",
        default="not_run",
        help="Optional local baseline governance status recorded in the report.",
    )
    parser.add_argument(
        "--local-baseline-note",
        default="see regression check section",
        help="Optional note for the local baseline governance row in the report.",
    )
    parser.add_argument("--skip-pymoo", action="store_true")
    parser.add_argument("--skip-deap", action="store_true")
    parser.add_argument("--skip-random-archive", action="store_true")
    return parser.parse_args()


def _candidate_variants() -> list[NSGA2CandidateVariant]:
    return [
        candidate_d_uniform_crossover(),
        candidate_h_uniform_dedup_mutation_boost(),
        candidate_j_h_lite_retry2(),
        candidate_l_sparse_parent_bias_light(),
        candidate_m_boundary_preservation_light(),
    ]


def _candidate_result(
    config: GAConfig,
    variant: NSGA2CandidateVariant,
    *,
    seed: int,
    output_root: Path,
) -> ExternalMOComparatorResult:
    result = PHASE1._candidate_result(config, variant, seed=seed, output_root=output_root)
    if variant.candidate_id == "candidate_m_boundary_preservation_light":
        result.metadata["promotion_status"] = "phase1_validation"
    return result


def _decorate_row(
    row: dict[str, Any],
    *,
    spec: MOBenchmarkSpec,
    base_config: GAConfig,
    reference_front: list[list[float]],
    requested_budget: int,
    variant_map: dict[str, NSGA2CandidateVariant],
) -> dict[str, Any]:
    return PHASE1._decorate_row(
        row,
        spec=spec,
        base_config=base_config,
        reference_front=reference_front,
        requested_budget=requested_budget,
        variant_map=variant_map,
    )


def _pick(
    paired_rows: list[dict[str, Any]],
    problem: str,
    left: str,
    right: str,
    metric: str,
) -> dict[str, Any] | None:
    return PHASE1._pick(paired_rows, problem, left, right, metric)


def _metric_wins(row: dict[str, Any] | None) -> bool:
    return PHASE1._metric_wins(row)


def _metric_losses(row: dict[str, Any] | None) -> bool:
    return PHASE1._metric_losses(row)


def _candidate_gate_rows(
    rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    default_rows = [row for row in rows if row["algorithm"] == "internal_nsga2"]
    candidate_rows = [
        row for row in rows if row["algorithm"] == "candidate_m_boundary_preservation_light"
    ]

    metadata_ok = all(
        row.get("metadata", {}).get("candidate_id") == "candidate_m_boundary_preservation_light"
        and row.get("metadata", {}).get("base_candidate_id") == "candidate_j_h_lite_retry2"
        and row.get("metadata", {}).get("mechanism") == "boundary_preservation_light"
        and row.get("metadata", {}).get("allowed_use") == "phase0_sanity_only"
        and row.get("metadata", {}).get("disallowed_use") == "default_replacement"
        for row in candidate_rows
    )
    default_changed_ok = all(
        row.get("metadata", {}).get("default_changed") is False for row in candidate_rows
    )
    candidate_isolation_ok = all(
        "candidate_id" not in row.get("metadata", {})
        and "default_changed" not in row.get("metadata", {})
        for row in default_rows
    )
    evaluations_ok = all(
        row.get("requested_budget") == row.get("actual_evaluations")
        for row in candidate_rows
        if row.get("success")
    )
    def _row_metric_gate_ok(row: dict[str, Any]) -> bool:
        if bool(row.get("metric_calculation_success")):
            return True
        if not row.get("success"):
            return False
        core_metrics = (
            "hypervolume_2d",
            "reference_front_distance",
            "inverted_generational_distance",
            "nondominated_count",
        )
        core_finite = all(
            isinstance(row.get(metric_name), int | float)
            and math.isfinite(float(row[metric_name]))
            for metric_name in core_metrics
        )
        spacing_value = row.get("spacing")
        nondominated_count = row.get("nondominated_count")
        spacing_degenerate = (
            isinstance(nondominated_count, int | float)
            and float(nondominated_count) <= 1.0
            and isinstance(spacing_value, int | float)
            and math.isnan(float(spacing_value))
        )
        return core_finite and spacing_degenerate

    metric_ok = all(_row_metric_gate_ok(row) for row in candidate_rows if row.get("success"))
    fairness_fail_free = fairness_payload.get("summary_counts", {}).get("fail", 0) == 0
    catastrophic_regression = any(
        isinstance(row.get("hypervolume_2d"), int | float) and float(row["hypervolume_2d"]) <= 0.0
        for row in candidate_rows
        if row.get("success")
    )

    return [
        {
            "gate": "candidate metadata",
            "result": metadata_ok,
            "evidence": "candidate_m raw_rows metadata",
            "interpretation": "candidate_m metadata should preserve candidate_id/base_candidate/mechanism and remain opt-in only",
        },
        {
            "gate": "default_changed=false",
            "result": default_changed_ok,
            "evidence": "candidate_m raw_rows metadata",
            "interpretation": "candidate_m rows must keep default_changed=false",
        },
        {
            "gate": "candidate isolation",
            "result": candidate_isolation_ok,
            "evidence": "default internal rows + candidate rows",
            "interpretation": "default internal NSGA-II rows must remain candidate-metadata free",
        },
        {
            "gate": "fairness fail 없음",
            "result": fairness_fail_free,
            "evidence": f"fairness summary={fairness_payload.get('summary_counts', {})}",
            "interpretation": "Phase 1 promotion logic is blocked if any fairness fail appears",
        },
        {
            "gate": "actual evaluations",
            "result": evaluations_ok,
            "evidence": "requested_budget vs actual_evaluations",
            "interpretation": "candidate_m actual evaluations must match the requested budget",
        },
        {
            "gate": "metric calculation success",
            "result": metric_ok,
            "evidence": "core MO metrics",
            "interpretation": (
                "Phase 1 core MO metrics must stay finite for candidate_m; a singleton-front "
                "spacing NaN is treated as a degenerate outcome, not a runner failure"
            ),
        },
        {
            "gate": "catastrophic regression",
            "result": not catastrophic_regression,
            "evidence": "candidate_m hypervolume_2d",
            "interpretation": "no zero-or-negative hypervolume collapse should appear",
        },
    ]


def _problem_tradeoff_score(
    paired_rows: list[dict[str, Any]],
    *,
    problem: str,
    left: str,
    right: str,
) -> tuple[int, int]:
    metrics = (
        "spacing",
        "nondominated_count",
        "hypervolume_2d",
        "reference_front_distance",
        "inverted_generational_distance",
        "coverage_indicator",
        "archive_duplicate_rate",
    )
    wins = 0
    losses = 0
    for metric in metrics:
        row = _pick(paired_rows, problem, left, right, metric)
        if _metric_wins(row):
            wins += 1
        elif _metric_losses(row):
            losses += 1
    return wins, losses


def _phase1_problem_rows(
    paired_rows: list[dict[str, Any]],
    problems: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for problem in problems:
        spacing_j = _pick(
            paired_rows,
            problem,
            "candidate_m_boundary_preservation_light",
            "candidate_j_h_lite_retry2",
            "spacing",
        )
        count_j = _pick(
            paired_rows,
            problem,
            "candidate_m_boundary_preservation_light",
            "candidate_j_h_lite_retry2",
            "nondominated_count",
        )
        hv_j = _pick(
            paired_rows,
            problem,
            "candidate_m_boundary_preservation_light",
            "candidate_j_h_lite_retry2",
            "hypervolume_2d",
        )
        distance_j = _pick(
            paired_rows,
            problem,
            "candidate_m_boundary_preservation_light",
            "candidate_j_h_lite_retry2",
            "reference_front_distance",
        )
        igd_j = _pick(
            paired_rows,
            problem,
            "candidate_m_boundary_preservation_light",
            "candidate_j_h_lite_retry2",
            "inverted_generational_distance",
        )
        coverage_j = _pick(
            paired_rows,
            problem,
            "candidate_m_boundary_preservation_light",
            "candidate_j_h_lite_retry2",
            "coverage_indicator",
        )
        duplicate_j = _pick(
            paired_rows,
            problem,
            "candidate_m_boundary_preservation_light",
            "candidate_j_h_lite_retry2",
            "archive_duplicate_rate",
        )

        diversity_positive = _metric_wins(spacing_j) or _metric_wins(count_j)
        duplicate_positive = _metric_wins(duplicate_j)
        core_regressions = sum(
            _metric_losses(row) for row in (hv_j, distance_j, igd_j, coverage_j)
        )

        wins_vs_l, losses_vs_l = _problem_tradeoff_score(
            paired_rows,
            problem=problem,
            left="candidate_m_boundary_preservation_light",
            right="candidate_l_sparse_parent_bias_light",
        )
        stable_vs_l = wins_vs_l >= losses_vs_l

        if diversity_positive and core_regressions == 0 and stable_vs_l:
            interpretation = "candidate_j 대비 spacing 또는 nondominated_count 개선이 있고, core regression 없이 candidate_l보다도 안정적이다"
        elif diversity_positive and core_regressions > 0:
            interpretation = "diversity 개선 신호는 있지만 HV/distance/IGD/coverage 일부 후퇴가 함께 나타난다"
        elif duplicate_positive and core_regressions == 0:
            interpretation = "duplicate rate는 개선됐지만 spacing/nondominated_count 신호는 약하다"
        elif core_regressions >= 3:
            interpretation = "candidate_j 대비 core convergence metric 후퇴가 커서 부정 신호다"
        else:
            interpretation = "candidate_j 대비 차이가 작거나 혼합 신호이며, candidate_l 대비 우위도 약하다"

        rows.append(
            {
                "problem": problem,
                "diversity_positive": diversity_positive,
                "duplicate_positive": duplicate_positive,
                "core_regressions": core_regressions,
                "stable_vs_l": stable_vs_l,
                "wins_vs_l": wins_vs_l,
                "losses_vs_l": losses_vs_l,
                "interpretation": interpretation,
            }
        )
    return rows


def _phase1_decision(
    gate_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    problem_rows: list[dict[str, Any]],
) -> str:
    if fairness_payload.get("summary_counts", {}).get("fail", 0) > 0:
        return "Needs fairness rerun"
    if not all(bool(row["result"]) for row in gate_rows):
        return "Fix required"

    positive_problems = sum(bool(row["diversity_positive"]) for row in problem_rows)
    catastrophic_problems = sum(int(row["core_regressions"]) >= 3 for row in problem_rows)
    stable_vs_l_problems = sum(bool(row["stable_vs_l"]) for row in problem_rows)
    mixed_problems = sum(
        bool(row["diversity_positive"]) and int(row["core_regressions"]) > 0 for row in problem_rows
    )

    if positive_problems >= 2 and catastrophic_problems == 0 and stable_vs_l_problems >= 2:
        return "Phase 1 passed, eligible for Phase 2 planning"
    if positive_problems >= 1 and catastrophic_problems == 0 and stable_vs_l_problems >= 2:
        return "Phase 1 passed with trade-offs"
    if catastrophic_problems >= 2 and positive_problems == 0 and stable_vs_l_problems <= 1:
        return "Reject"
    if mixed_problems > 0 or positive_problems == 1:
        return "Hold for more evidence"
    return "Hold for more evidence"


def _results_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# NSGA-II Boundary Preservation Phase 1 Results",
        "",
        "## Aggregate Summary",
        "",
        *BASE._markdown_table(
            payload["aggregate_rows"],
            [
                "problem",
                "algorithm",
                "status",
                "seeds",
                "mean_hv",
                "mean_distance",
                "mean_gd",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_duplicate_rate",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Paired Summary",
        "",
        *BASE._markdown_table(
            payload["paired_rows"],
            [
                "problem",
                "comparison",
                "metric",
                "win",
                "tie",
                "loss",
                "mean_delta",
                "median_delta",
                "comparable_seeds",
            ],
        ),
        "",
        "## Fairness Summary",
        "",
        *BASE._markdown_table(fairness_summary_rows(payload["fairness"]), ["status", "pass", "warning", "fail"]),
        "",
        "## Decision",
        "",
        f"- `{payload['phase1_decision']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def _fairness_report_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# NSGA-II Boundary Preservation Phase 1 Fairness Report",
        "",
        "## Fairness Summary",
        "",
        *BASE._markdown_table(fairness_summary_rows(payload["fairness"]), ["status", "pass", "warning", "fail"]),
        "",
        "## Fairness Issues",
        "",
        *BASE._markdown_table(
            payload["fairness"]["issues"],
            ["status", "issue_type", "algorithm", "problem", "message", "severity", "recommended_action"],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _phase1_report_markdown(payload: dict[str, Any]) -> str:
    drift_payload = payload.get("drift_audit")
    drift_rows = drift_payload.get("summary_rows", []) if isinstance(drift_payload, dict) else []
    drift_detected = (
        bool(drift_payload.get("overall", {}).get("drift_detected"))
        if isinstance(drift_payload, dict)
        else None
    )
    local_baseline_status = str(payload.get("local_baseline_status", "not_run"))
    candidate_definition = {
        "candidate_id": "candidate_m_boundary_preservation_light",
        "base_candidate": "candidate_j_h_lite_retry2",
        "mechanism": "boundary_preservation_light",
        "default_changed": False,
        "promotion_status": "phase1_validation",
        "allowed_use": "phase0_sanity_only",
        "disallowed_use": "default_replacement",
    }

    drift_table_rows = [
        {
            "gate": "default metadata contamination",
            "result": "pass" if drift_detected is False else "fail" if drift_detected else "n/a",
            "evidence": payload.get("drift_audit_path") or "n/a",
            "interpretation": "default path should remain free of candidate metadata after nsga2.py changes",
        },
        {
            "gate": "default_changed=false",
            "result": "pass"
            if any(row["gate"] == "default_changed=false" and row["result"] for row in payload["gate_rows"])
            else "fail",
            "evidence": "candidate_m metadata",
            "interpretation": "candidate_m must keep default_changed=false",
        },
        {
            "gate": "actual evaluations",
            "result": "pass"
            if any(row["gate"] == "actual evaluations" and row["result"] for row in payload["gate_rows"])
            else "fail",
            "evidence": "requested_budget vs actual_evaluations",
            "interpretation": "candidate_m actual evaluations should match the requested budget",
        },
        {
            "gate": "candidate isolation",
            "result": "pass"
            if any(row["gate"] == "candidate isolation" and row["result"] for row in payload["gate_rows"])
            else "fail",
            "evidence": "default internal rows + candidate rows",
            "interpretation": "candidate_m metadata should appear only on explicit opt-in runs",
        },
        {
            "gate": "nsga2.py drift",
            "result": "pass" if drift_detected is False else "fail" if drift_detected else "n/a",
            "evidence": payload.get("drift_audit_path") or "n/a",
            "interpretation": "nsga2.py boundary-preservation support must not perturb the default internal baseline",
        },
        {
            "gate": "local baseline governance",
            "result": local_baseline_status,
            "evidence": "python scripts/check_local_baseline.py ...",
            "interpretation": str(payload.get("local_baseline_note", "see regression check section")),
        },
    ]

    paired_focus = {
        "candidate_m vs candidate_j",
        "candidate_m vs candidate_l",
        "candidate_m vs candidate_h",
        "candidate_m vs candidate_d",
        "candidate_m vs pymoo",
        "candidate_m vs deap",
    }

    lines: list[str] = [
        "# NSGA-II Boundary Preservation Phase 1 ZDT Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: `candidate_m_boundary_preservation_light`의 ZDT Phase 1 반복 검증",
        f"- default drift audit 결과: `{('DRIFT DETECTED' if drift_detected else 'NO DRIFT') if drift_detected is not None else 'not available'}`",
        f"- candidate isolation 결과: `{'pass' if any(row['gate'] == 'candidate isolation' and row['result'] for row in payload['gate_rows']) else 'fail'}`",
        f"- 실행한 benchmark: {', '.join(payload['selected_problems'])}",
        f"- candidate_m vs candidate_j 핵심 결과: `{payload['phase1_decision']}` 판단 이전의 상세 신호는 paired summary와 problem summary에 기록했다",
        f"- candidate_m vs candidate_l 핵심 결과: boundary-preservation 가설이 sparse-parent-bias 대비 더 안정적인지 paired summary에 별도 기록했다",
        f"- fairness 결과: pass={payload['fairness_summary'].get('pass', 0)}, warning={payload['fairness_summary'].get('warning', 0)}, fail={payload['fairness_summary'].get('fail', 0)}",
        f"- Phase 1 decision: **{payload['phase1_decision']}**",
        "- 기본값 변경 여부: 없음",
        "- Level 판정 변화 여부: Level 판정 유지 또는 실험 툴킷 근거 강화 범위로만 해석한다",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "- Scope: candidate_m_boundary_preservation_light, ZDT1/ZDT2/ZDT3, 10-seed repeated validation, candidate_m vs candidate_j, candidate_m vs candidate_l, fairness check, default drift check",
        "- Non-Scope: default promotion, candidate_m change request, DTLZ/WFG validation, new survivor-pressure families, productization",
        "",
        "## 3. Default Drift and Candidate Isolation",
        "",
        *BASE._markdown_table(
            drift_table_rows,
            ["gate", "result", "evidence", "interpretation"],
        ),
        "",
        "## 4. Candidate Definition",
        "",
        *BASE._markdown_table(
            [{"field": key, "value": value} for key, value in candidate_definition.items()],
            ["field", "value"],
        ),
        "",
        "## 5. Experiment Configuration",
        "",
        *BASE._markdown_table(
            [
                {
                    "문제": row["problem"],
                    "알고리즘": row["algorithm"],
                    "seeds": row["seeds"],
                    "requested_budget": payload["budget"],
                    "actual_evaluations_summary": row["mean_actual_evaluations"],
                    "주요_설정": row["status"],
                }
                for row in payload["aggregate_rows"]
            ],
            ["문제", "알고리즘", "seeds", "requested_budget", "actual_evaluations_summary", "주요_설정"],
        ),
        "",
        "## 6. Results Summary",
        "",
        *BASE._markdown_table(
            payload["aggregate_rows"],
            [
                "problem",
                "algorithm",
                "mean_hv",
                "mean_distance",
                "mean_gd",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_duplicate_rate",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## 7. Paired Comparisons",
        "",
        *BASE._markdown_table(
            [row for row in payload["paired_rows"] if row["comparison"] in paired_focus],
            [
                "problem",
                "comparison",
                "metric",
                "win",
                "tie",
                "loss",
                "mean_delta",
                "median_delta",
            ],
        ),
        "",
        "## 8. Candidate M Decision",
        "",
        f"- **{payload['phase1_decision']}**",
        "",
        "## 9. What We Learned",
        "",
        *[f"- {row['problem']}: {row['interpretation']}" for row in payload["problem_rows"]],
        "- boundary preservation이 spacing / nondominated_count를 개선했는지는 candidate_m vs candidate_j paired rows에서 문제별로 따로 읽어야 한다.",
        "- candidate_j 대비 HV / distance / IGD regression이 커지면 boundary preservation 가설은 Phase 2 근거를 얻지 못한다.",
        "- candidate_l 대비 trade-off가 나아졌는지는 candidate_m vs candidate_l paired rows로 확인한다.",
        "- pymoo / DEAP 대비 약점이 남으면 그대로 기록하고, ZDT Phase 1만으로 external parity 또는 우세를 주장하지 않는다.",
        "- Phase 2로 가려면 최소한 fairness fail이 없어야 하고, candidate_j 대비 diversity 신호가 ZDT 2개 이상에서 반복되어야 한다.",
        "",
        "## 10. Failures and Warnings",
        "",
        *BASE._markdown_table(
            payload["failures"]
            or [{"type": "none", "target": "none", "seed": None, "message": "none", "impact": "none", "action": "none"}],
            ["type", "target", "seed", "message", "impact", "action"],
        ),
        "",
        "## 11. Regression Check",
        "",
        *BASE._markdown_table(
            [
                {
                    "command": "python scripts/audit_nsga2_default_drift.py ...",
                    "result": "see drift artifact",
                    "note": payload.get("drift_audit_path") or "n/a",
                },
                {
                    "command": "python scripts/check_local_baseline.py ...",
                    "result": local_baseline_status,
                    "note": str(payload.get("local_baseline_note", "see regression check section")),
                },
                {
                    "command": "python scripts/validate_nsga2_boundary_phase1.py ...",
                    "result": "success",
                    "note": "current run",
                },
            ],
            ["command", "result", "note"],
        ),
        "",
        "## 12. Maturity Impact",
        "",
        "- Level 판정 유지.",
        "- Phase 1은 candidate evidence이지 default algorithm maturity 상향 근거가 아니다.",
        "- candidate_m이 좋아도 기본값이 바뀌지 않았으므로 default NSGA-II maturity 상향은 금지한다.",
        "- fairness / isolation / default drift gate가 유지되면 실험 툴킷으로서의 Level 4 근거는 강화 가능하다.",
        "- pymoo / DEAP 대비 약점이 남아 있으면 범용 optimizer 성숙도 상향은 금지한다.",
        "",
        "## 13. Recommended Next Work",
        "",
        "1. Phase 1 passed이면 Phase 2 DTLZ planning 작성",
        "2. Phase 1 passed with trade-offs이면 candidate_m 설계 조정 여부 검토",
        "3. Hold이면 seed 수 또는 ZDT stress 재검토",
        "4. Reject이면 boundary preservation family 폐기 또는 backlog로 회수",
        "5. candidate_l은 Hold 상태 유지",
        "6. candidate_j opt-in documentation 유지",
        "7. fairness checker single-objective runner 확장 검토",
        "8. constrained multi-objective contract",
        "9. checkpoint/resume",
        "10. parallel evaluation",
        "",
        f"이번 Phase 1 결과, candidate_m_boundary_preservation_light는 ZDT 계열에서 {payload['phase1_decision']} 신호를 보였고, candidate_j 대비 mixed signal 여부는 paired rows에 기록되었으며, 최종 판정은 {payload['phase1_decision']}이다.",
        "",
    ]

    if drift_rows:
        lines.extend(
            [
                "## Drift Excerpt",
                "",
                *BASE._markdown_table(
                    drift_rows,
                    [
                        "problem",
                        "metric",
                        "previous_mean",
                        "current_mean",
                        "delta",
                        "drift_detected",
                        "interpretation",
                    ],
                ),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(PROJECT_ROOT / args.config)
    selected_problems = [item.strip().lower() for item in str(args.problems).split(",") if item.strip()]
    selected_specs = [mo_candidate_suite_specs()[name] for name in selected_problems]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    variants = _candidate_variants()
    variant_map = {variant.candidate_id: variant for variant in variants}

    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    comparison_specs = [
        ("candidate_m_boundary_preservation_light", "candidate_j_h_lite_retry2", "candidate_m vs candidate_j"),
        ("candidate_m_boundary_preservation_light", "candidate_l_sparse_parent_bias_light", "candidate_m vs candidate_l"),
        ("candidate_m_boundary_preservation_light", "candidate_h_uniform_dedup_mutation_boost", "candidate_m vs candidate_h"),
        ("candidate_m_boundary_preservation_light", "candidate_d_uniform_crossover", "candidate_m vs candidate_d"),
        ("candidate_m_boundary_preservation_light", "internal_nsga2", "candidate_m vs internal baseline"),
        ("candidate_m_boundary_preservation_light", "pymoo_nsga2", "candidate_m vs pymoo"),
        ("candidate_m_boundary_preservation_light", "deap_nsga2", "candidate_m vs deap"),
        ("candidate_m_boundary_preservation_light", "random_pareto_archive", "candidate_m vs random archive"),
        ("candidate_j_h_lite_retry2", "pymoo_nsga2", "candidate_j vs pymoo"),
        ("candidate_l_sparse_parent_bias_light", "candidate_j_h_lite_retry2", "candidate_l vs candidate_j"),
    ]

    for spec in selected_specs:
        config = build_problem_config(base_config, spec)
        config = PHASE1._retarget_budget(config, args.budget)
        reference_front = reference_front_for_spec(spec, point_count=201)
        reference_point = list(spec.hv_reference_point)
        problem_output_root = output_root / spec.problem
        problem_output_root.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            results: list[ExternalMOComparatorResult] = [
                run_internal_nsga2(config, seed=seed, output_root=str(problem_output_root)),
                *[
                    _candidate_result(config, variant, seed=seed, output_root=problem_output_root)
                    for variant in variants
                ],
            ]
            if not args.skip_pymoo:
                results.append(run_pymoo_nsga2(config, seed=seed, budget=args.budget))
            if not args.skip_deap:
                results.append(run_deap_nsga2(config, seed=seed, budget=args.budget))
            if not args.skip_random_archive:
                results.append(
                    run_random_archive_anchor(
                        run_random_pareto_archive(config, seed=seed, budget=args.budget)
                    )
                )

            for result in results:
                row = result_to_front_row(
                    result,
                    reference_front=reference_front,
                    reference_point=reference_point,
                )
                row = _decorate_row(
                    row,
                    spec=spec,
                    base_config=config,
                    reference_front=reference_front,
                    requested_budget=args.budget,
                    variant_map=variant_map,
                )
                raw_rows.append(row)
                if result.status != "success":
                    failures.append(
                        {
                            "type": result.status,
                            "target": result.algorithm_name,
                            "seed": seed,
                            "message": result.error_message,
                            "impact": "seed excluded from paired comparison",
                            "action": "review comparator/runtime failure before any Phase 1 conclusion",
                        }
                    )

    fairness_payload = evaluate_parameter_fairness(
        raw_rows,
        benchmark_rows=benchmark_rows,
        candidate_rows=candidate_rows,
    )
    aggregate_rows = PHASE1._aggregate_rows(raw_rows)
    paired_rows = PHASE1._paired_rows(raw_rows, comparison_specs)
    gate_rows = _candidate_gate_rows(raw_rows, fairness_payload)
    problem_rows = _phase1_problem_rows(paired_rows, selected_problems)
    phase1_decision = _phase1_decision(gate_rows, fairness_payload, problem_rows)
    drift_audit_path = PROJECT_ROOT / args.drift_audit_json
    drift_payload = PHASE1._load_drift_payload(drift_audit_path, selected_problems)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "selected_problems": selected_problems,
        "seeds": seeds,
        "budget": args.budget,
        "benchmark_rows": benchmark_rows,
        "candidate_rows": candidate_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "gate_rows": gate_rows,
        "problem_rows": problem_rows,
        "phase1_decision": phase1_decision,
        "failures": failures,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
        "drift_audit_path": str(drift_audit_path) if drift_payload is not None else None,
        "drift_audit": drift_payload,
        "local_baseline_status": args.local_baseline_status,
        "local_baseline_note": args.local_baseline_note,
    }

    results_json = safe_artifact_path(
        artifact_root,
        "nsga2_boundary_preservation_phase1_results",
        args.artifact_suffix,
        ".json",
    )
    results_csv = safe_artifact_path(
        artifact_root,
        "nsga2_boundary_preservation_phase1_results",
        args.artifact_suffix,
        ".csv",
    )
    results_md = safe_artifact_path(
        artifact_root,
        "nsga2_boundary_preservation_phase1_results",
        args.artifact_suffix,
        ".md",
    )
    report_md = safe_artifact_path(
        artifact_root,
        "nsga2_boundary_preservation_phase1_report",
        args.artifact_suffix,
        ".md",
    )
    fairness_md = safe_artifact_path(
        artifact_root,
        "nsga2_boundary_preservation_phase1_fairness_report",
        args.artifact_suffix,
        ".md",
    )

    BASE._write_json(results_json, payload)
    BASE._write_csv(
        results_csv,
        aggregate_rows,
        [
            "problem",
            "algorithm",
            "library",
            "status",
            "seeds",
            "successful_seeds",
            "mean_hv",
            "mean_distance",
            "mean_gd",
            "mean_igd",
            "mean_spacing",
            "mean_coverage",
            "mean_nondominated_count",
            "mean_duplicate_rate",
            "mean_archive_duplicate_rate",
            "mean_objective_duplicate_rate",
            "mean_decision_duplicate_rate",
            "mean_unique_decision_count",
            "mean_unique_objective_count",
            "mean_runtime_seconds",
            "mean_actual_evaluations",
            "success_rate",
        ],
    )
    results_md.write_text(_results_markdown(payload), encoding="utf-8")
    fairness_md.write_text(_fairness_report_markdown(payload), encoding="utf-8")
    report_md.write_text(_phase1_report_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "results_json": str(results_json),
                "results_csv": str(results_csv),
                "results_md": str(results_md),
                "report_md": str(report_md),
                "fairness_md": str(fairness_md),
                "decision": phase1_decision,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
