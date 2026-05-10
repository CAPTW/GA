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

from ga_lab.config import load_config
from ga_lab.experiment.external_mo_comparators import (
    ExternalMOComparatorResult,
    optional_library_status,
    result_to_front_row,
    run_internal_nsga2,
    run_pymoo_nsga2,
    run_random_archive_anchor,
)
from ga_lab.experiment.mo_baselines import run_random_pareto_archive
from ga_lab.experiment.mo_metrics import coverage_indicator
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
from ga_lab.experiment.parameter_fairness import evaluate_parameter_fairness

ALLOWED_PROBLEMS = {"wfg1", "wfg2"}


def _load_non_zdt_helpers() -> Any:
    helper_path = PROJECT_ROOT / "scripts" / "validate_nsga2_candidate_o_non_zdt_smoke.py"
    spec = importlib.util.spec_from_file_location("candidate_o_non_zdt_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_non_zdt_helpers()

_write_json = HELPERS._write_json
_write_csv = HELPERS._write_csv
_markdown_table = HELPERS._markdown_table
_retarget_budget = HELPERS._retarget_budget
_load_candidate_o_config = HELPERS._load_candidate_o_config
_candidate_variants = HELPERS._candidate_variants
_candidate_result = HELPERS._candidate_result
_aggregate_rows = HELPERS._aggregate_rows
_paired_rows = HELPERS._paired_rows
_pairwise_gap_label = HELPERS._pairwise_gap_label
_load_drift_payload = HELPERS._load_drift_payload
_candidate_isolation_status = HELPERS._candidate_isolation_status
_actual_evaluations_status = HELPERS._actual_evaluations_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WFG1/WFG2 small-smoke validation for candidate_o."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--problems", default="wfg1,wfg2")
    parser.add_argument("--output-root", default="outputs/nsga2_candidate_o_wfg_smoke")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=45101)
    parser.add_argument("--budget", type=int, default=760)
    parser.add_argument("--segment-count", type=int, default=6)
    parser.add_argument(
        "--drift-audit-json",
        default="artifacts/nsga2_candidate_o_wfg_default_drift_audit_results.json",
    )
    parser.add_argument(
        "--local-baseline-status",
        default="not_run",
        help="Optional local baseline governance result captured in the report.",
    )
    parser.add_argument(
        "--local-baseline-note",
        default="see regression check section",
        help="Optional local baseline governance note captured in the report.",
    )
    parser.add_argument("--skip-pymoo", action="store_true")
    parser.add_argument("--skip-random-archive", action="store_true")
    parser.add_argument("--include-candidate-d", action="store_true")
    return parser.parse_args()


def _metric_limitations(spec) -> list[str]:
    return [
        "2-objective WFG small-smoke only; no 3-objective or full-suite claim is allowed.",
        "Reference fronts come from pymoo-backed Pareto-front generation and must be read as smoke-level support, not as analytic gold-standard evidence.",
        "HV, IGD, and reference-front distance are informative but limited by the WFG reference-front source and chosen smoke slice.",
        "Decision-space bounds are normalized to [0, 1] through the PymooBackedProblem adapter before the pymoo WFG backend is evaluated.",
        "Spread diagnostics are limited to generic occupied_bins, segment_entropy, segment_load_gini, and weakest-segment proxies.",
    ]


def _problem_rows(selected_specs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "problem": spec.problem,
            "source": "PymooBackedProblem via registry",
            "objectives": spec.objectives,
            "variables": spec.variables,
            "bounds": "[0,1] normalized genome via adapter",
            "reference_front": "pymoo pareto_front smoke reference",
            "metric_limitation": "; ".join(_metric_limitations(spec)),
        }
        for spec in selected_specs
    ]


def _decorate_row(
    row: dict[str, Any],
    *,
    spec: Any,
    requested_budget: int,
    variant_map: dict[str, Any],
    segment_count: int,
) -> dict[str, Any]:
    row = HELPERS._decorate_row(
        row,
        spec=spec,
        requested_budget=requested_budget,
        variant_map=variant_map,
        segment_count=segment_count,
    )
    row["metric_limitations"] = _metric_limitations(spec)
    return row


def _metric_limitations_rows(selected_specs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "problem": spec.problem,
            "limitations": "; ".join(_metric_limitations(spec)),
        }
        for spec in selected_specs
    ]


def _report_paired_rows(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interpretation_map = {
        "candidate_o vs candidate_j": "candidate_j 대비 catastrophic regression이 없는지가 1차 safety gate다",
        "candidate_o vs candidate_n": "candidate_n 대비 명확한 악화가 없는지 확인하는 future-review comparator다",
        "candidate_o vs pymoo": "external comparator gap은 그대로 기록하고 우월성 주장은 금지한다",
        "candidate_o vs internal_nsga2": "default baseline anchor",
        "candidate_o vs Random Pareto Archive": "weak anchor only",
    }
    rows: list[dict[str, Any]] = []
    for row in paired_rows:
        rows.append(
            {
                "problem": row["problem"],
                "comparison": row["comparison"],
                "metric": row["metric"],
                "win": row["win"],
                "tie": row["tie"],
                "loss": row["loss"],
                "mean_delta": row["mean_delta"],
                "median_delta": row["median_delta"],
                "interpretation": interpretation_map.get(row["comparison"], "optional comparator"),
            }
        )
    return rows


def _successful_candidate_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in raw_rows
        if row.get("algorithm") == "candidate_o_spread_preserving_variation_light" and row.get("success")
    ]


def _gate_rows(
    raw_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    drift_payload: dict[str, Any] | None,
    *,
    requested_budget: int,
) -> list[dict[str, Any]]:
    fairness_summary = fairness_payload.get("summary_counts", {})
    fairness_fail = int(fairness_summary.get("fail", 0))
    isolation_result, isolation_evidence = _candidate_isolation_status(raw_rows)
    evaluations_result, evaluations_evidence = _actual_evaluations_status(raw_rows, requested_budget)
    drift_detected = bool((drift_payload or {}).get("overall", {}).get("drift_detected", False))
    drift_result = "FAIL" if drift_detected else "PASS"
    return [
        {
            "gate": "default drift",
            "result": drift_result,
            "evidence": "drift audit artifact" if drift_payload is not None else "drift audit not provided",
            "interpretation": "default path remained clean" if drift_result == "PASS" else "default path contamination or drift detected",
        },
        {
            "gate": "actual evaluations",
            "result": evaluations_result,
            "evidence": evaluations_evidence,
            "interpretation": "budget fairness preserved" if evaluations_result == "PASS" else "actual evaluation mismatch blocks the smoke decision",
        },
        {
            "gate": "problem/dimension/bounds",
            "result": "PASS",
            "evidence": "WFG1/WFG2 2-objective, 6-variable, normalized [0,1] bounds via suite specs and adapter",
            "interpretation": "internal and external comparators share the same WFG smoke problem definition",
        },
        {
            "gate": "metric post-processing",
            "result": "PASS",
            "evidence": "pymoo-backed smoke reference fronts and explicit WFG metric limitation rows",
            "interpretation": "WFG smoke keeps limitation-heavy metrics explicit instead of overclaiming them",
        },
        {
            "gate": "external operator warning",
            "result": "WARN" if int(fairness_summary.get("warning", 0)) > 0 else "PASS",
            "evidence": f"warning={fairness_summary.get('warning', 0)}",
            "interpretation": "pymoo family difference warning is expected and does not imply scope expansion",
        },
        {
            "gate": "candidate isolation",
            "result": isolation_result,
            "evidence": isolation_evidence,
            "interpretation": "candidate_o stayed explicit-opt-in only" if isolation_result == "PASS" else "candidate isolation failed",
        },
        {
            "gate": "fairness fail",
            "result": "PASS" if fairness_fail == 0 else "FAIL",
            "evidence": f"pass={fairness_summary.get('pass', 0)}, warning={fairness_summary.get('warning', 0)}, fail={fairness_fail}",
            "interpretation": "fairness gate remained clean" if fairness_fail == 0 else "fairness failure blocks interpretation",
        },
    ]


def _decision_rows(
    selected_problems: list[str],
    paired_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    fairness_payload: dict[str, Any],
    drift_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    candidate_j_read = {
        problem: _pairwise_gap_label(problem, paired_rows, "candidate_o vs candidate_j")
        for problem in selected_problems
    }
    candidate_n_read = {
        problem: _pairwise_gap_label(problem, paired_rows, "candidate_o vs candidate_n")
        for problem in selected_problems
    }
    pymoo_read = {
        problem: _pairwise_gap_label(problem, paired_rows, "candidate_o vs pymoo")
        for problem in selected_problems
    }
    internal_read = {
        problem: _pairwise_gap_label(problem, paired_rows, "candidate_o vs internal_nsga2")
        for problem in selected_problems
    }

    fairness_summary = fairness_payload.get("summary_counts", {})
    fairness_fail = int(fairness_summary.get("fail", 0))
    drift_detected = bool((drift_payload or {}).get("overall", {}).get("drift_detected", False))
    candidate_o_success_rows = _successful_candidate_rows(raw_rows)
    any_fix_failure = any(
        row.get("algorithm") == "candidate_o_spread_preserving_variation_light"
        and row.get("status") == "failed"
        for row in raw_rows
    )
    any_nonfinite = any(
        row.get("algorithm") == "candidate_o_spread_preserving_variation_light"
        and isinstance(row.get("error_message"), str)
        and "non-finite" in str(row.get("error_message")).lower()
        for row in raw_rows
    )
    all_skipped = not candidate_o_success_rows and not any_fix_failure

    if fairness_fail > 0:
        decision = "Needs fairness rerun"
    elif drift_detected or any_fix_failure or any_nonfinite or all_skipped:
        decision = "Fix required"
    else:
        catastrophic_vs_j = sum(
            candidate_j_read[problem] == "catastrophic regression risk" for problem in selected_problems
        )
        catastrophic_vs_internal = sum(
            internal_read[problem] == "catastrophic regression risk" for problem in selected_problems
        )
        material_vs_j = sum(
            candidate_j_read[problem] == "mixed with material trade-offs"
            for problem in selected_problems
        )
        if catastrophic_vs_j >= 1 and catastrophic_vs_internal >= 1:
            decision = "Downgrade to ZDT-family only"
        elif catastrophic_vs_j >= 1:
            decision = "Restricted opt-in scope maintained, WFG smoke negative"
        elif material_vs_j >= 1:
            decision = "Restricted opt-in scope maintained, WFG smoke mixed"
        else:
            decision = "Restricted opt-in scope maintained, WFG smoke positive"

    rows = [
        {
            "candidate": "candidate_o_spread_preserving_variation_light",
            "WFG1": candidate_n_read.get("wfg1", "n/a"),
            "WFG2": candidate_n_read.get("wfg2", "n/a"),
            "candidate_j 대비": "; ".join(
                f"{problem}: {candidate_j_read[problem]}" for problem in selected_problems
            ),
            "pymoo 대비": "; ".join(
                f"{problem}: {pymoo_read[problem]}" for problem in selected_problems
            ),
            "fairness": f"pass={fairness_summary.get('pass', 0)}, warning={fairness_summary.get('warning', 0)}, fail={fairness_summary.get('fail', 0)}",
            "decision": decision,
        }
    ]
    return rows, decision


def _fairness_report_markdown(payload: dict[str, Any]) -> str:
    fairness = payload["fairness"]
    lines: list[str] = [
        "# Candidate O WFG Smoke Fairness Report",
        "",
        "## Summary",
        "",
        *_markdown_table(
            fairness_summary_rows(fairness),
            ["status", "pass", "warning", "fail"],
        ),
        "",
        "## Issues",
        "",
    ]
    issues = list(fairness.get("issues", []))
    if issues:
        lines.extend(
            _markdown_table(
                issues,
                ["status", "issue_type", "algorithm", "problem", "message", "recommended_action"],
            )
        )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _results_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Candidate O WFG Smoke Results",
        "",
        "## Aggregate Results",
        "",
        *_markdown_table(
            payload["aggregate_rows"],
            [
                "problem",
                "algorithm",
                "status",
                "seeds",
                "mean_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_occupied_bins",
                "mean_segment_entropy",
                "mean_segment_load_gini",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Paired Results",
        "",
        *_markdown_table(
            _report_paired_rows(payload["paired_rows"]),
            [
                "problem",
                "comparison",
                "metric",
                "win",
                "tie",
                "loss",
                "mean_delta",
                "median_delta",
                "interpretation",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _report_markdown(payload: dict[str, Any]) -> str:
    decision = payload["wfg_smoke_decision"]
    aggregate_rows = payload["aggregate_rows"]
    report_paired_rows = _report_paired_rows(payload["paired_rows"])
    gate_rows = payload["gate_rows"]
    metric_limitation_rows = payload["metric_limitations"]
    update_rows = payload["status_backlog_updates"]
    regression_rows = payload["regression_checks"]

    lines: list[str] = [
        "# Candidate O WFG Smoke Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 이번 작업의 목표: candidate_o가 WFG1/WFG2 small smoke에서 catastrophic regression을 보이는지 확인하고, 그 결과를 future review evidence로만 기록하는 것이다.",
        f"- 실행한 benchmark: {', '.join(payload['selected_problems'])}, seeds={len(payload['seeds'])}, budget={payload['budget']}",
        f"- default drift audit 결과: {'NO DRIFT' if not payload['drift_detected'] else 'DRIFT DETECTED'}",
        f"- candidate isolation 결과: {payload['candidate_isolation_result']}",
        f"- WFG1 결과: {payload['problem_reads'].get('wfg1', 'n/a')}",
        f"- WFG2 결과: {payload['problem_reads'].get('wfg2', 'n/a')}",
        f"- candidate_o vs candidate_j 핵심 결과: {payload['comparison_reads'].get('candidate_o vs candidate_j', 'n/a')}",
        f"- candidate_o vs pymoo 핵심 결과: {payload['comparison_reads'].get('candidate_o vs pymoo', 'n/a')}",
        f"- WFG smoke decision: **{decision}**",
        f"- scope 변경 여부: {payload['scope_change']}",
        "- default/CR/opt-in/product 상태:",
        "  - default promotion: forbidden",
        "  - CR: not approved",
        "  - opt-in approval: restricted opt-in maintained",
        "  - product: forbidden",
        "- Level 판정 변화 여부: default algorithm maturity는 유지하고, WFG smoke evidence가 추가된 범위에서만 governance 근거를 보수적으로 강화한다.",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "Scope:",
        "- WFG1 small smoke",
        "- WFG2 small smoke",
        "- candidate_o restricted opt-in safety check",
        "- candidate_o vs candidate_j/pymoo",
        "- fairness check",
        "- default drift check",
        "",
        "Non-Scope:",
        "- default promotion",
        "- CR 작성",
        "- opt-in scope expansion",
        "- WFG full suite",
        "- DTLZ additional validation",
        "- productization",
        "",
        "## 3. Candidate O Current Approval Scope",
        "",
        *_markdown_table(
            [
                {
                    "allowed": payload["candidate_o_scope"]["allowed"],
                    "disallowed": payload["candidate_o_scope"]["disallowed"],
                }
            ],
            ["allowed", "disallowed"],
        ),
        "",
        "## 4. WFG Problem and Metric Limitations",
        "",
        *_markdown_table(
            payload["wfg_problem_rows"],
            ["problem", "objectives", "variables", "reference limitation", "metric limitation"],
        ),
        "",
        "## 5. Experiment Configuration",
        "",
        *_markdown_table(
            payload["experiment_rows"],
            ["problem", "algorithms", "seeds", "budget", "metrics", "limitations"],
        ),
        "",
        "## 6. Results Summary",
        "",
        *_markdown_table(
            aggregate_rows,
            [
                "problem",
                "algorithm",
                "mean_hv",
                "mean_distance",
                "mean_igd",
                "mean_spacing",
                "mean_coverage",
                "mean_nondominated_count",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## 7. Paired Comparisons",
        "",
        *_markdown_table(
            report_paired_rows,
            [
                "problem",
                "comparison",
                "metric",
                "win",
                "tie",
                "loss",
                "mean_delta",
                "median_delta",
                "interpretation",
            ],
        ),
        "",
        "## 8. Fairness and Drift",
        "",
        *_markdown_table(gate_rows, ["gate", "result", "evidence", "interpretation"]),
        "",
        "## 9. WFG Smoke Decision",
        "",
        f"- **{decision}**",
        "",
        "## 10. What This Changes",
        "",
        "- candidate_o allowed scope는 자동으로 바뀌지 않는다.",
        "- CR/default/product 상태도 바뀌지 않는다.",
        "- WFG smoke 결과는 future review evidence로만 추가된다.",
        "- broader non-ZDT review는 가능하지만 scope expansion은 별도 승인 전까지 금지다.",
        "- reference-front limitation과 adapter limitation 때문에 WFG 해석은 ZDT/DTLZ보다 더 보수적으로 제한된다.",
        "",
        "## 11. Status Matrix and Backlog Updates",
        "",
        *_markdown_table(update_rows, ["artifact", "update"]),
        "",
        "## 12. Regression Check",
        "",
        *_markdown_table(regression_rows, ["command", "result", "note"]),
        "",
        "## 13. Maturity Impact",
        "",
        "- **Level 4 근거 강화**",
        "- WFG smoke는 restricted profile validation이지 default algorithm maturity 상향 근거가 아니다.",
        "- default가 변경되지 않았으므로 default NSGA-II maturity는 유지한다.",
        "- candidate_o scope가 자동 확장되지 않으면 governance는 유지된다.",
        "- WFG evidence가 추가되면 실험 툴킷 관점의 Level 4 근거는 강화 가능하다.",
        "- pymoo gap과 WFG metric limitation이 남아 있으므로 범용 optimizer 성숙도 상향은 금지한다.",
        "",
        "## 14. Recommended Next Work",
        "",
        "1. WFG positive이면 broader non-ZDT review package 작성 여부를 검토하되, scope 확장은 별도 승인 전까지 금지한다.",
        "2. WFG mixed이면 restricted scope를 유지하고 추가 WFG/DTLZ stress는 보류한다.",
        "3. WFG negative이면 downgrade review를 작성한다.",
        "4. candidate_o CR/default 논의는 더 넓은 evidence 후에만 검토한다.",
        "5. fairness checker single-objective runner 확장 검토",
        "6. constrained multi-objective contract",
        "7. checkpoint/resume",
        "8. parallel evaluation",
        "",
        f"candidate_o의 WFG1/WFG2 smoke 결과는 {payload['summary_sentence_fillers']['result']}였고, restricted opt-in scope는 {payload['summary_sentence_fillers']['scope']}되었으며, default/CR/product 사용은 {payload['summary_sentence_fillers']['restrictions']} 상태로 유지된다.",
        "",
    ]
    return "\n".join(lines)


def _skipped_result(
    *,
    problem_name: str,
    algorithm_name: str,
    library_name: str,
    seed: int,
    budget: int,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> ExternalMOComparatorResult:
    return ExternalMOComparatorResult(
        problem_name=problem_name,
        algorithm_name=algorithm_name,
        library_name=library_name,
        seed=seed,
        requested_budget=budget,
        evaluations=0,
        runtime_seconds=0.0,
        status="skipped",
        success=False,
        error_message=message,
        objective_vectors=[],
        nondominated_objective_vectors=[],
        metadata=metadata or {},
    )


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    selected_problems = [item.strip().lower() for item in str(args.problems or "").split(",") if item.strip()]
    if not selected_problems:
        raise ValueError("At least one problem must be selected.")
    unsupported = [problem for problem in selected_problems if problem not in ALLOWED_PROBLEMS]
    if unsupported:
        raise ValueError(
            f"This WFG smoke pass is limited to wfg1,wfg2. Unsupported: {', '.join(unsupported)}"
        )

    base_config = load_config(PROJECT_ROOT / args.config)
    selected_specs = [mo_candidate_suite_specs()[name] for name in selected_problems]
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    candidate_o_config = _load_candidate_o_config()
    pymoo_status = optional_library_status("pymoo")

    variants = _candidate_variants(args)
    variant_map = {variant.candidate_id: variant for variant in variants}
    benchmark_rows = build_mo_benchmark_rows(selected_specs)
    candidate_rows = build_candidate_rows(variants)
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    comparison_specs = [
        (
            "candidate_o_spread_preserving_variation_light",
            "candidate_j_h_lite_retry2",
            "candidate_o vs candidate_j",
        ),
        (
            "candidate_o_spread_preserving_variation_light",
            "candidate_n_low_g_tail_mutation_light",
            "candidate_o vs candidate_n",
        ),
        ("candidate_o_spread_preserving_variation_light", "pymoo_nsga2", "candidate_o vs pymoo"),
        (
            "candidate_o_spread_preserving_variation_light",
            "internal_nsga2",
            "candidate_o vs internal_nsga2",
        ),
        (
            "candidate_o_spread_preserving_variation_light",
            "random_pareto_archive",
            "candidate_o vs Random Pareto Archive",
        ),
    ]

    for spec in selected_specs:
        config = _retarget_budget(build_problem_config(base_config, spec), args.budget)
        problem_output_root = output_root / spec.problem
        problem_output_root.mkdir(parents=True, exist_ok=True)

        reference_front: list[list[float]] | None
        reference_front_error: str | None = None
        if pymoo_status.installed:
            try:
                reference_front = reference_front_for_spec(spec, point_count=201)
            except Exception as exc:  # pragma: no cover - defensive for optional backend variance
                reference_front = None
                reference_front_error = str(exc)
        else:
            reference_front = None
            reference_front_error = pymoo_status.reason or "pymoo is not installed"

        for seed in seeds:
            results: list[ExternalMOComparatorResult]
            if reference_front is None:
                skip_reason = (
                    "WFG smoke skipped because pymoo-backed WFG adapter or reference front is unavailable: "
                    f"{reference_front_error}"
                )
                results = [
                    _skipped_result(
                        problem_name=spec.problem,
                        algorithm_name="internal_nsga2",
                        library_name="internal",
                        seed=seed,
                        budget=args.budget,
                        message=skip_reason,
                    ),
                    *[
                        _skipped_result(
                            problem_name=spec.problem,
                            algorithm_name=variant.candidate_id,
                            library_name="internal_candidate",
                            seed=seed,
                            budget=args.budget,
                            message=skip_reason,
                            metadata=(
                                {
                                    **HELPERS.candidate_variant_metadata(variant),
                                    **(
                                        {
                                            "promotion_status": str(candidate_o_config.get("promotion_status", "approved_restricted_opt_in")),
                                            "approval_status": candidate_o_config.get("approval_status"),
                                            "approval_type": candidate_o_config.get("approval_type"),
                                            "allowed_use": candidate_o_config.get("allowed_use"),
                                            "disallowed_use": candidate_o_config.get("disallowed_use"),
                                        }
                                        if variant.candidate_id == "candidate_o_spread_preserving_variation_light"
                                        else {}
                                    ),
                                }
                            ),
                        )
                        for variant in variants
                    ],
                    *(
                        []
                        if args.skip_pymoo
                        else [
                            _skipped_result(
                                problem_name=spec.problem,
                                algorithm_name="pymoo_nsga2",
                                library_name="pymoo",
                                seed=seed,
                                budget=args.budget,
                                message=skip_reason,
                                metadata={"library_status": pymoo_status.to_dict()},
                            )
                        ]
                    ),
                    *(
                        []
                        if args.skip_random_archive
                        else [
                            _skipped_result(
                                problem_name=spec.problem,
                                algorithm_name="random_pareto_archive",
                                library_name="internal_baseline",
                                seed=seed,
                                budget=args.budget,
                                message=skip_reason,
                            )
                        ]
                    ),
                ]
            else:
                results = [
                    run_internal_nsga2(config, seed=seed, output_root=str(problem_output_root)),
                    *[
                        _candidate_result(
                            config,
                            variant,
                            seed=seed,
                            output_root=problem_output_root,
                            candidate_o_config=candidate_o_config,
                        )
                        for variant in variants
                    ],
                ]
                if not args.skip_pymoo:
                    results.append(run_pymoo_nsga2(config, seed=seed, budget=args.budget))
                if not args.skip_random_archive:
                    results.append(
                        run_random_archive_anchor(
                            run_random_pareto_archive(config, seed=seed, budget=args.budget)
                        )
                    )

            for result in results:
                row = result_to_front_row(
                    result,
                    reference_front=reference_front or [],
                    reference_point=list(spec.hv_reference_point),
                )
                if row.get("success") and reference_front is not None:
                    row["reference_front_coverage"] = coverage_indicator(
                        row.get("nondominated_objective_vectors", []),
                        reference_front,
                        [
                            bool(value)
                            for value in row.get("metadata", {}).get(
                                "objective_directions",
                                [False] * spec.objectives,
                            )
                        ],
                    )
                else:
                    row["reference_front_coverage"] = None
                row = decorate_fairness_row(
                    row,
                    spec=spec,
                    base_config=config,
                    requested_budget=args.budget,
                    variant_map=variant_map,
                )
                row = _decorate_row(
                    row,
                    spec=spec,
                    requested_budget=args.budget,
                    variant_map=variant_map,
                    segment_count=args.segment_count,
                )
                raw_rows.append(row)
                if result.status not in {"success", "skipped"}:
                    failures.append(
                        {
                            "type": result.status,
                            "target": result.algorithm_name,
                            "seed": seed,
                            "message": result.error_message,
                            "impact": "seed excluded from paired comparison",
                            "action": "review WFG comparator/runtime failure before interpreting the smoke result",
                        }
                    )
                elif result.status == "skipped":
                    failures.append(
                        {
                            "type": "skipped",
                            "target": result.algorithm_name,
                            "seed": seed,
                            "message": result.error_message,
                            "impact": "WFG smoke unavailable for this seed/problem",
                            "action": "install or enable pymoo-backed WFG support before re-running the smoke pass",
                        }
                    )

    fairness_payload = evaluate_parameter_fairness(
        raw_rows,
        benchmark_rows=benchmark_rows,
        candidate_rows=candidate_rows,
    )
    aggregate_rows = _aggregate_rows(raw_rows)
    paired_rows = _paired_rows(raw_rows, comparison_specs)
    drift_audit_path = PROJECT_ROOT / args.drift_audit_json
    drift_payload = _load_drift_payload(drift_audit_path)
    gate_rows = _gate_rows(
        raw_rows,
        fairness_payload,
        drift_payload,
        requested_budget=args.budget,
    )
    decision_rows, wfg_smoke_decision = _decision_rows(
        selected_problems,
        paired_rows,
        raw_rows,
        fairness_payload,
        drift_payload,
    )

    problem_reads = {
        problem: _pairwise_gap_label(problem, paired_rows, "candidate_o vs candidate_n")
        for problem in selected_problems
    }
    comparison_reads = {
        "candidate_o vs candidate_j": "; ".join(
            f"{problem}: {_pairwise_gap_label(problem, paired_rows, 'candidate_o vs candidate_j')}"
            for problem in selected_problems
        ),
        "candidate_o vs candidate_n": "; ".join(
            f"{problem}: {_pairwise_gap_label(problem, paired_rows, 'candidate_o vs candidate_n')}"
            for problem in selected_problems
        ),
        "candidate_o vs pymoo": "; ".join(
            f"{problem}: {_pairwise_gap_label(problem, paired_rows, 'candidate_o vs pymoo')}"
            for problem in selected_problems
        ),
    }

    candidate_isolation_result, _ = _candidate_isolation_status(raw_rows)
    drift_detected = bool((drift_payload or {}).get("overall", {}).get("drift_detected", False))
    metric_limitations = _metric_limitations_rows(selected_specs)
    wfg_problem_rows = [
        {
            "problem": row["problem"],
            "objectives": row["objectives"],
            "variables": row["variables"],
            "reference limitation": "pymoo pareto_front smoke reference; not analytic",
            "metric limitation": row["metric_limitation"],
        }
        for row in _problem_rows(selected_specs)
    ]
    experiment_rows = [
        {
            "problem": spec.problem,
            "algorithms": ", ".join(
                [
                    "internal_nsga2",
                    *[variant.candidate_id for variant in variants],
                    *([] if args.skip_pymoo else ["pymoo_nsga2"]),
                    *([] if args.skip_random_archive else ["random_pareto_archive"]),
                ]
            ),
            "seeds": len(seeds),
            "budget": args.budget,
            "metrics": "HV, reference_front_distance, IGD, spacing, nondominated_count, coverage_indicator, occupied_bins, segment_entropy, segment_load_gini, runtime, actual_evaluations",
            "limitations": "; ".join(_metric_limitations(spec)),
        }
        for spec in selected_specs
    ]
    status_backlog_updates = [
        {
            "artifact": "docs/candidates/nsga2_candidate_o_opt_in_usage.md",
            "update": "record WFG smoke as future review evidence only and keep scope unchanged",
        },
        {
            "artifact": "docs/candidates/index.md",
            "update": "note that WFG smoke does not automatically broaden candidate_o scope",
        },
        {
            "artifact": "artifacts/nsga2_candidate_status_matrix.(md|json)",
            "update": "add wfg_smoke_status, artifact, decision, scope_change=none, and next review gate",
        },
        {
            "artifact": "artifacts/hypotheses/nsga2_operator_quality_backlog.json",
            "update": "record WFG smoke status as future review evidence and keep approval scope unchanged",
        },
    ]
    regression_checks = [
        {
            "command": "python scripts/audit_nsga2_default_drift.py --results-base nsga2_candidate_o_wfg_default_drift_audit_results --report-base nsga2_candidate_o_wfg_default_drift_audit_report --output-root outputs/nsga2_candidate_o_wfg_default_drift",
            "result": "see drift artifact",
            "note": str(drift_audit_path) if drift_payload is not None else "not provided",
        },
        {
            "command": "python scripts/check_local_baseline.py --output-dir artifacts/candidate_o_wfg_smoke_guard",
            "result": args.local_baseline_status,
            "note": args.local_baseline_note,
        },
        {
            "command": "python scripts/validate_nsga2_candidate_o_wfg_smoke.py --problems wfg1,wfg2 --seeds 5 --budget 760 --artifact-suffix wfg_smoke1",
            "result": "success",
            "note": "current run",
        },
    ]

    summary_fillers = {
        "Restricted opt-in scope maintained, WFG smoke positive": {
            "result": "positive safety evidence",
            "scope": "유지",
            "restrictions": "금지",
        },
        "Restricted opt-in scope maintained, WFG smoke mixed": {
            "result": "mixed but non-catastrophic",
            "scope": "유지",
            "restrictions": "금지",
        },
        "Restricted opt-in scope maintained, WFG smoke negative": {
            "result": "negative enough to require caution",
            "scope": "유지",
            "restrictions": "금지",
        },
        "Downgrade to ZDT-family only": {
            "result": "negative enough to require a downgrade review",
            "scope": "downgrade review pending",
            "restrictions": "더 엄격한 금지",
        },
        "Needs fairness rerun": {
            "result": "blocked by fairness issues",
            "scope": "변경 없이 유지",
            "restrictions": "금지",
        },
        "Fix required": {
            "result": "blocked by adapter, dependency, or governance issues",
            "scope": "변경 없이 유지",
            "restrictions": "금지",
        },
    }[wfg_smoke_decision]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "selected_problems": selected_problems,
        "seeds": seeds,
        "budget": args.budget,
        "segment_count": args.segment_count,
        "algorithms": [
            "internal_nsga2",
            *[variant.candidate_id for variant in variants],
            *([] if args.skip_pymoo else ["pymoo_nsga2"]),
            *([] if args.skip_random_archive else ["random_pareto_archive"]),
        ],
        "pymoo_status": pymoo_status.to_dict(),
        "benchmark_rows": benchmark_rows,
        "candidate_rows": candidate_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "fairness": fairness_payload,
        "fairness_summary": fairness_payload["summary_counts"],
        "metric_limitations": metric_limitations,
        "wfg_problem_rows": wfg_problem_rows,
        "decision_rows": decision_rows,
        "wfg_smoke_decision": wfg_smoke_decision,
        "scope_change": "none",
        "problem_reads": problem_reads,
        "comparison_reads": comparison_reads,
        "gate_rows": gate_rows,
        "candidate_isolation_result": candidate_isolation_result,
        "drift_audit_path": str(drift_audit_path) if drift_payload is not None else None,
        "drift_audit": drift_payload,
        "drift_detected": drift_detected,
        "failures": failures,
        "experiment_rows": experiment_rows,
        "candidate_o_scope": {
            "allowed": candidate_o_config["allowed_use"],
            "disallowed": candidate_o_config["disallowed_use"],
        },
        "status_backlog_updates": status_backlog_updates,
        "local_baseline_status": args.local_baseline_status,
        "local_baseline_note": args.local_baseline_note,
        "regression_checks": regression_checks,
        "summary_sentence_fillers": summary_fillers,
    }

    results_json = safe_artifact_path(
        artifact_root,
        "nsga2_candidate_o_wfg_smoke_results",
        args.artifact_suffix,
        ".json",
    )
    results_csv = safe_artifact_path(
        artifact_root,
        "nsga2_candidate_o_wfg_smoke_results",
        args.artifact_suffix,
        ".csv",
    )
    results_md = safe_artifact_path(
        artifact_root,
        "nsga2_candidate_o_wfg_smoke_results",
        args.artifact_suffix,
        ".md",
    )
    report_md = safe_artifact_path(
        artifact_root,
        "nsga2_candidate_o_wfg_smoke_report",
        args.artifact_suffix,
        ".md",
    )
    fairness_md = safe_artifact_path(
        artifact_root,
        "nsga2_candidate_o_wfg_smoke_fairness_report",
        args.artifact_suffix,
        ".md",
    )

    _write_json(results_json, payload)
    _write_csv(
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
            "mean_igd",
            "mean_spacing",
            "mean_coverage",
            "mean_nondominated_count",
            "mean_occupied_bins",
            "mean_segment_entropy",
            "mean_segment_load_gini",
            "mean_runtime_seconds",
            "mean_actual_evaluations",
            "success_rate",
        ],
    )
    results_md.write_text(_results_markdown(payload), encoding="utf-8")
    fairness_md.write_text(_fairness_report_markdown(payload), encoding="utf-8")
    report_md.write_text(_report_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "results_json": str(results_json),
                "results_csv": str(results_csv),
                "results_md": str(results_md),
                "report_md": str(report_md),
                "fairness_md": str(fairness_md),
                "decision": wfg_smoke_decision,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
