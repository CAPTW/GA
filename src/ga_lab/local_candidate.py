from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ga_lab.local_baseline import PROJECT_ROOT as DEFAULT_PROJECT_ROOT
from ga_lab.local_baseline import check_local_baseline
from ga_lab.local_experiments import load_local_study, run_local_study

CANDIDATE_SCHEMA_VERSION = 1
ALLOWED_DECISION_LABELS = {
    "reject_regression",
    "reject_no_material_gain",
    "note_only_stress_slice",
    "monitor_only",
    "candidate_promising_needs_confirm",
    "candidate_passes_local_guard",
    "candidate_requires_new_mechanism_hypothesis",
    "baseline_drift_detected",
    "intentional_baseline_change_required",
}
REQUIRED_FIELDS = (
    "candidate_id",
    "problem",
    "target_id",
    "hypothesis_id",
    "baseline_snapshot_path",
    "baseline_guard_study",
    "candidate_profile_or_overrides",
    "allowed_budget_policy",
    "expected_primary_metric",
    "expected_non_regression_metrics",
    "stress_slices_to_check",
    "seed_policy",
    "decision_policy",
    "notes",
)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "candidate"


def _resolve_path(path_ref: str | Path, *, project_root: Path) -> Path:
    path = Path(path_ref)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str] | None:
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items()):
            return row
    return None


def _coerce_manifest_value(value: Any, field_name: str) -> Any:
    if field_name in {
        "candidate_id",
        "problem",
        "target_id",
        "hypothesis_id",
        "baseline_snapshot_path",
        "baseline_guard_study",
        "allowed_budget_policy",
        "expected_primary_metric",
        "notes",
    }:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        return value.strip()
    if field_name in {
        "candidate_profile_or_overrides",
        "seed_policy",
        "decision_policy",
    }:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be a JSON object")
        return value
    if field_name in {
        "expected_non_regression_metrics",
        "stress_slices_to_check",
    }:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{field_name} must be a non-empty array")
        return value
    return value


def validate_candidate_manifest(
    payload: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT
    normalized: dict[str, Any] = {
        "schema_version": int(payload.get("schema_version", CANDIDATE_SCHEMA_VERSION))
    }
    if normalized["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CANDIDATE_SCHEMA_VERSION}, got {normalized['schema_version']}"
        )
    for field in REQUIRED_FIELDS:
        if field not in payload:
            raise ValueError(f"Missing required candidate field: {field}")
        normalized[field] = _coerce_manifest_value(payload[field], field)

    problem = normalized["problem"]
    if problem not in {"tsp", "zdt1", "knapsack", "onemax"}:
        raise ValueError(f"Unsupported candidate problem: {problem}")

    load_local_study(normalized["baseline_guard_study"])

    optional_fields = {
        "candidate_study",
        "existing_output_dir",
        "expected_decision_label",
        "baseline_variant",
        "candidate_variant",
        "quality_variant",
        "comparison_variants",
    }
    for field in optional_fields:
        value = payload.get(field)
        if value is None:
            continue
        if field == "comparison_variants":
            if not isinstance(value, list) or not value:
                raise ValueError("comparison_variants must be a non-empty array when provided")
            normalized[field] = [str(item) for item in value]
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string when provided")
        normalized[field] = value.strip()

    if "expected_decision_label" in normalized and (
        normalized["expected_decision_label"] not in ALLOWED_DECISION_LABELS
    ):
        raise ValueError(
            f"expected_decision_label must be one of {sorted(ALLOWED_DECISION_LABELS)}"
        )

    baseline_snapshot_path = _resolve_path(
        normalized["baseline_snapshot_path"],
        project_root=root,
    )
    if not baseline_snapshot_path.exists():
        raise FileNotFoundError(f"Baseline snapshot not found: {baseline_snapshot_path}")

    if "candidate_study" in normalized:
        load_local_study(normalized["candidate_study"])

    if "existing_output_dir" in normalized:
        existing_output_dir = _resolve_path(normalized["existing_output_dir"], project_root=root)
        normalized["existing_output_dir"] = str(existing_output_dir)

    normalized["baseline_snapshot_path"] = str(baseline_snapshot_path)
    return normalized


def load_candidate_manifest(
    candidate_ref: str | Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT
    candidate_path = _resolve_path(candidate_ref, project_root=root)
    payload = _load_json(candidate_path)
    if not isinstance(payload, dict):
        raise ValueError("Candidate manifest must contain a JSON object")
    normalized = validate_candidate_manifest(payload, project_root=root)
    normalized["source_path"] = str(candidate_path)
    return normalized


def _candidate_output_dir(
    candidate_id: str,
    *,
    project_root: Path,
    output_root: str | Path,
) -> Path:
    root = _resolve_path(output_root, project_root=project_root)
    return root / f"{_timestamp()}_{_safe_name(candidate_id)}"


def _is_blocking_baseline_drift(report: dict[str, Any]) -> bool:
    blocking_prefixes = (
        "profile:",
        "protocol_matrix",
        "target_registry_hash",
        "target_status:",
        "problem_rule:",
        "candidate_schema_hash",
        "candidate_rule:",
    )
    for check in report.get("checks", []):
        name = str(check.get("name", ""))
        status = str(check.get("status", ""))
        if status != "PASS" and name.startswith(blocking_prefixes):
            return True
    return False


def _load_or_run_candidate_output(
    manifest: dict[str, Any],
    *,
    project_root: Path,
    no_execute: bool,
    use_existing_output: bool,
) -> tuple[Path | None, bool]:
    existing_output = manifest.get("existing_output_dir")
    if existing_output:
        existing_path = _resolve_path(existing_output, project_root=project_root)
        if existing_path.exists():
            return existing_path, False
        if use_existing_output:
            raise FileNotFoundError(f"Existing candidate output not found: {existing_path}")
    if no_execute:
        return None, False
    candidate_study = manifest.get("candidate_study")
    if not candidate_study:
        if use_existing_output:
            raise ValueError(
                "Candidate manifest must provide existing_output_dir with --use-existing-output"
            )
        return None, False
    result = run_local_study(candidate_study, output_root="outputs/local_studies")
    return Path(result["study_dir"]), True


def _result_row(
    metric: str,
    baseline_value: Any,
    candidate_value: Any,
    *,
    higher_is_better: bool | None = None,
) -> dict[str, Any]:
    baseline_float = _float_or_none(baseline_value)
    candidate_float = _float_or_none(candidate_value)
    delta = None
    if baseline_float is not None and candidate_float is not None:
        delta = candidate_float - baseline_float
    return {
        "metric": metric,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "delta": delta,
        "higher_is_better": higher_is_better if higher_is_better is not None else "",
    }


def _evaluate_tsp_candidate(
    manifest: dict[str, Any],
    candidate_output_dir: Path,
) -> dict[str, Any]:
    rows = _read_csv_rows(candidate_output_dir / "tsp_fast_tail_summary.csv")
    baseline_variant = manifest.get("baseline_variant", "current_fast")
    candidate_variant = manifest.get("candidate_variant", "candidate")
    quality_variant = manifest.get("quality_variant", "quality_first")

    baseline_overall = _find_row(
        rows,
        scope="overall",
        case_group="overall",
        study_variant=baseline_variant,
    )
    candidate_overall = _find_row(
        rows,
        scope="overall",
        case_group="overall",
        study_variant=candidate_variant,
    )
    baseline_anti = _find_row(
        rows,
        scope="case_group",
        case_group="anti_case",
        study_variant=baseline_variant,
    )
    candidate_anti = _find_row(
        rows,
        scope="case_group",
        case_group="anti_case",
        study_variant=candidate_variant,
    )
    baseline_rescue = _find_row(
        rows,
        scope="case_group",
        case_group="rescue_target",
        study_variant=baseline_variant,
    )
    candidate_rescue = _find_row(
        rows,
        scope="case_group",
        case_group="rescue_target",
        study_variant=candidate_variant,
    )
    if not all(
        [
            baseline_overall,
            candidate_overall,
            baseline_anti,
            candidate_anti,
            baseline_rescue,
            candidate_rescue,
        ]
    ):
        raise ValueError("TSP candidate output is missing required comparison rows")

    anti_p95_candidate = _float_or_none(candidate_anti["p95_loss_pct"])
    anti_p95_baseline = _float_or_none(baseline_anti["p95_loss_pct"])
    anti_max_candidate = _float_or_none(candidate_anti["max_loss_pct"])
    anti_max_baseline = _float_or_none(baseline_anti["max_loss_pct"])
    rescue_mean_candidate = _float_or_none(candidate_rescue["mean_loss_pct"])
    rescue_mean_baseline = _float_or_none(baseline_rescue["mean_loss_pct"])

    p95_gain = (
        anti_p95_baseline - anti_p95_candidate
        if anti_p95_candidate is not None and anti_p95_baseline is not None
        else None
    )
    max_gain = (
        anti_max_baseline - anti_max_candidate
        if anti_max_candidate is not None and anti_max_baseline is not None
        else None
    )
    rescue_mean_delta = (
        rescue_mean_candidate - rescue_mean_baseline
        if rescue_mean_candidate is not None and rescue_mean_baseline is not None
        else None
    )

    metrics_rows = [
        _result_row(
            "overall_mean_loss_pct_vs_q",
            baseline_overall["mean_loss_pct"],
            candidate_overall["mean_loss_pct"],
            higher_is_better=False,
        ),
        _result_row(
            "anti_case_p90_loss_pct_vs_q",
            baseline_anti["p90_loss_pct"],
            candidate_anti["p90_loss_pct"],
            higher_is_better=False,
        ),
        _result_row(
            "anti_case_p95_loss_pct_vs_q",
            baseline_anti["p95_loss_pct"],
            candidate_anti["p95_loss_pct"],
            higher_is_better=False,
        ),
        _result_row(
            "anti_case_max_loss_pct_vs_q",
            baseline_anti["max_loss_pct"],
            candidate_anti["max_loss_pct"],
            higher_is_better=False,
        ),
        _result_row(
            "rescue_target_mean_loss_pct_vs_q",
            baseline_rescue["mean_loss_pct"],
            candidate_rescue["mean_loss_pct"],
            higher_is_better=False,
        ),
        _result_row(
            "actual_evaluations_used_mean",
            baseline_overall["actual_evaluations_used_mean"],
            candidate_overall["actual_evaluations_used_mean"],
            higher_is_better=False,
        ),
        _result_row(
            "runtime_seconds_mean",
            baseline_overall["runtime_seconds_mean"],
            candidate_overall["runtime_seconds_mean"],
            higher_is_better=False,
        ),
    ]

    decision_policy = manifest["decision_policy"]
    needs_new_mechanism = bool(decision_policy.get("requires_new_mechanism_hypothesis"))
    has_new_mechanism = bool(decision_policy.get("new_mechanism_hypothesis"))
    if needs_new_mechanism and not has_new_mechanism:
        decision_label = "candidate_requires_new_mechanism_hypothesis"
        rationale = (
            "TSP anti-case tail is frozen as a protocol limitation, so a same-budget contour candidate "
            "without a new mechanism hypothesis is not eligible for fast-default promotion."
        )
        next_action = "reopen_only_with_new_mechanism_hypothesis"
    else:
        rescue_worsen_limit = float(decision_policy.get("rescue_mean_max_worsen", 0.05))
        p95_min_gain = float(decision_policy.get("anti_case_p95_min_gain", 0.05))
        if (
            (p95_gain is not None and p95_gain < 0.0)
            or (max_gain is not None and max_gain < 0.0)
            or (rescue_mean_delta is not None and rescue_mean_delta > rescue_worsen_limit)
        ):
            decision_label = "reject_regression"
            rationale = (
                "The candidate regressed the frozen TSP baseline on anti-case p95/max or rescue-target preservation."
            )
            next_action = "keep_default_and_monitor"
        elif (p95_gain or 0.0) < p95_min_gain and (max_gain or 0.0) <= 0.0:
            decision_label = "reject_no_material_gain"
            rationale = (
                "The candidate did not deliver a material anti-case tail gain over the frozen fast baseline."
            )
            next_action = "keep_default_and_monitor"
        else:
            decision_label = "candidate_promising_needs_confirm"
            rationale = (
                "The candidate showed a same-budget anti-case tail gain without obvious rescue regression."
            )
            next_action = "intentional_baseline_change_review"

    return {
        "compared_profiles": {
            "quality_reference": quality_variant,
            "baseline_variant": baseline_variant,
            "candidate_variant": candidate_variant,
        },
        "primary_metric_delta": {
            "anti_case_p95_delta_vs_current_fast": p95_gain,
            "anti_case_max_delta_vs_current_fast": max_gain,
            "overall_mean_loss_delta_vs_current_fast": (
                _float_or_none(candidate_overall["mean_loss_pct"])
                - _float_or_none(baseline_overall["mean_loss_pct"])
                if _float_or_none(candidate_overall["mean_loss_pct"]) is not None
                and _float_or_none(baseline_overall["mean_loss_pct"]) is not None
                else None
            ),
        },
        "non_regression_results": {
            "rescue_target_mean_delta_vs_current_fast": rescue_mean_delta,
            "actual_evaluations_delta": (
                _float_or_none(candidate_overall["actual_evaluations_used_mean"])
                - _float_or_none(baseline_overall["actual_evaluations_used_mean"])
                if _float_or_none(candidate_overall["actual_evaluations_used_mean"]) is not None
                and _float_or_none(baseline_overall["actual_evaluations_used_mean"]) is not None
                else None
            ),
        },
        "stress_slice_results": {
            "anti_case": {
                "baseline_p95": baseline_anti["p95_loss_pct"],
                "candidate_p95": candidate_anti["p95_loss_pct"],
                "baseline_max": baseline_anti["max_loss_pct"],
                "candidate_max": candidate_anti["max_loss_pct"],
            },
            "rescue_target": {
                "baseline_mean": baseline_rescue["mean_loss_pct"],
                "candidate_mean": candidate_rescue["mean_loss_pct"],
            },
        },
        "actual_evaluations_used": _float_or_none(
            candidate_overall["actual_evaluations_used_mean"]
        ),
        "runtime_seconds": _float_or_none(candidate_overall["runtime_seconds_mean"]),
        "decision_label": decision_label,
        "rationale": rationale,
        "recommended_next_action": next_action,
        "metrics_rows": metrics_rows,
    }


def _evaluate_zdt1_candidate(
    manifest: dict[str, Any],
    candidate_output_dir: Path,
) -> dict[str, Any]:
    rows = _read_csv_rows(candidate_output_dir / "zdt1_spread_candidate_boundary_table.csv")
    baseline_variant = manifest.get("baseline_variant", "current_fast")
    candidate_variant = manifest.get("candidate_variant", "candidate")
    quality_variant = manifest.get("quality_variant", "quality_first")

    baseline_overall = _find_row(
        rows, scope="overall", slice="overall", study_variant=baseline_variant
    )
    candidate_overall = _find_row(
        rows, scope="overall", slice="overall", study_variant=candidate_variant
    )
    baseline_spread = _find_row(
        rows, scope="slice", slice="spread_stress", study_variant=baseline_variant
    )
    candidate_spread = _find_row(
        rows, scope="slice", slice="spread_stress", study_variant=candidate_variant
    )
    baseline_joint = _find_row(
        rows, scope="slice", slice="joint_non_regression", study_variant=baseline_variant
    )
    candidate_joint = _find_row(
        rows, scope="slice", slice="joint_non_regression", study_variant=candidate_variant
    )
    baseline_normal = _find_row(
        rows, scope="slice", slice="normal_holdout", study_variant=baseline_variant
    )
    candidate_normal = _find_row(
        rows, scope="slice", slice="normal_holdout", study_variant=candidate_variant
    )
    baseline_stable = _find_row(
        rows, scope="slice", slice="stable_contrast", study_variant=baseline_variant
    )
    candidate_stable = _find_row(
        rows, scope="slice", slice="stable_contrast", study_variant=candidate_variant
    )
    if not all(
        [
            baseline_overall,
            candidate_overall,
            baseline_spread,
            candidate_spread,
            baseline_joint,
            candidate_joint,
            baseline_normal,
            candidate_normal,
            baseline_stable,
            candidate_stable,
        ]
    ):
        raise ValueError("ZDT1 candidate output is missing required boundary rows")

    spread_fail_gain = (
        _float_or_none(baseline_spread["spread_fail_rate"])
        - _float_or_none(candidate_spread["spread_fail_rate"])
        if _float_or_none(baseline_spread["spread_fail_rate"]) is not None
        and _float_or_none(candidate_spread["spread_fail_rate"]) is not None
        else None
    )
    spread_tail_gain = (
        _float_or_none(baseline_spread["spread_delta_p95"])
        - _float_or_none(candidate_spread["spread_delta_p95"])
        if _float_or_none(baseline_spread["spread_delta_p95"]) is not None
        and _float_or_none(candidate_spread["spread_delta_p95"]) is not None
        else None
    )
    hv_mean_delta = (
        _float_or_none(candidate_overall["mean_loss_pct"])
        - _float_or_none(baseline_overall["mean_loss_pct"])
        if _float_or_none(candidate_overall["mean_loss_pct"]) is not None
        and _float_or_none(baseline_overall["mean_loss_pct"]) is not None
        else None
    )
    hv_p90_delta = (
        _float_or_none(candidate_overall["p90_loss_pct"])
        - _float_or_none(baseline_overall["p90_loss_pct"])
        if _float_or_none(candidate_overall["p90_loss_pct"]) is not None
        and _float_or_none(baseline_overall["p90_loss_pct"]) is not None
        else None
    )
    joint_delta = (
        _float_or_none(candidate_joint["joint_safety_fail_rate"])
        - _float_or_none(baseline_joint["joint_safety_fail_rate"])
        if _float_or_none(candidate_joint["joint_safety_fail_rate"]) is not None
        and _float_or_none(baseline_joint["joint_safety_fail_rate"]) is not None
        else None
    )

    metrics_rows = [
        _result_row(
            "overall_hv_loss_pct_vs_q",
            baseline_overall["mean_loss_pct"],
            candidate_overall["mean_loss_pct"],
            higher_is_better=False,
        ),
        _result_row(
            "overall_hv_p90_loss_pct_vs_q",
            baseline_overall["p90_loss_pct"],
            candidate_overall["p90_loss_pct"],
            higher_is_better=False,
        ),
        _result_row(
            "spread_stress_fail_rate",
            baseline_spread["spread_fail_rate"],
            candidate_spread["spread_fail_rate"],
            higher_is_better=False,
        ),
        _result_row(
            "spread_stress_delta_p95",
            baseline_spread["spread_delta_p95"],
            candidate_spread["spread_delta_p95"],
            higher_is_better=False,
        ),
        _result_row(
            "joint_non_regression_fail_rate",
            baseline_joint["joint_safety_fail_rate"],
            candidate_joint["joint_safety_fail_rate"],
            higher_is_better=False,
        ),
        _result_row(
            "normal_holdout_hv_loss_pct_vs_q",
            baseline_normal["mean_loss_pct"],
            candidate_normal["mean_loss_pct"],
            higher_is_better=False,
        ),
        _result_row(
            "stable_contrast_hv_loss_pct_vs_q",
            baseline_stable["mean_loss_pct"],
            candidate_stable["mean_loss_pct"],
            higher_is_better=False,
        ),
    ]

    stable_score = _float_or_none(candidate_overall["stable_slice_non_regression_score"])
    normal_score = _float_or_none(candidate_overall["normal_slice_non_regression_score"])
    pareto_score = _float_or_none(candidate_overall["pareto_non_regression_score"])
    decision_policy = manifest["decision_policy"]
    hv_tail_worsen_limit = float(decision_policy.get("hv_p90_max_worsen", 0.05))
    if (spread_fail_gain or 0.0) <= 0.0 and (spread_tail_gain or 0.0) <= 0.0:
        decision_label = "reject_no_material_gain"
        rationale = (
            "The spread candidate did not produce a material gain on the pinned spread-stress slice."
        )
        next_action = "keep_default_and_monitor"
    elif (
        (stable_score is not None and stable_score < 0.0)
        or (normal_score is not None and normal_score < 0.0)
        or (pareto_score is not None and pareto_score < 0.0)
        or (joint_delta is not None and joint_delta > 0.0)
        or (hv_p90_delta is not None and hv_p90_delta > hv_tail_worsen_limit)
    ):
        decision_label = "note_only_stress_slice"
        rationale = (
            "The candidate improves the pinned spread-stress slice, but stable/normal or joint non-regression is not strong enough for a fast-default replacement."
        )
        next_action = "only_validate_if_candidate_generalizes_beyond_stress_slice"
    else:
        decision_label = "candidate_passes_local_guard"
        rationale = (
            "The candidate cleared the spread-stress gain and non-regression checks against the frozen fast baseline."
        )
        next_action = "intentional_baseline_change_review"

    return {
        "compared_profiles": {
            "quality_reference": quality_variant,
            "baseline_variant": baseline_variant,
            "candidate_variant": candidate_variant,
        },
        "primary_metric_delta": {
            "spread_fail_rate_delta_vs_current_fast": spread_fail_gain,
            "spread_delta_p95_delta_vs_current_fast": spread_tail_gain,
            "overall_hv_mean_loss_delta_vs_current_fast": hv_mean_delta,
        },
        "non_regression_results": {
            "overall_hv_p90_delta_vs_current_fast": hv_p90_delta,
            "joint_fail_rate_delta_vs_current_fast": joint_delta,
            "stable_slice_non_regression_score": stable_score,
            "normal_slice_non_regression_score": normal_score,
            "pareto_non_regression_score": pareto_score,
        },
        "stress_slice_results": {
            "spread_stress": {
                "baseline_fail_rate": baseline_spread["spread_fail_rate"],
                "candidate_fail_rate": candidate_spread["spread_fail_rate"],
                "baseline_p95": baseline_spread["spread_delta_p95"],
                "candidate_p95": candidate_spread["spread_delta_p95"],
            },
            "joint_non_regression": {
                "baseline_fail_rate": baseline_joint["joint_safety_fail_rate"],
                "candidate_fail_rate": candidate_joint["joint_safety_fail_rate"],
            },
            "normal_holdout": {
                "baseline_hv_loss": baseline_normal["mean_loss_pct"],
                "candidate_hv_loss": candidate_normal["mean_loss_pct"],
            },
            "stable_contrast": {
                "baseline_hv_loss": baseline_stable["mean_loss_pct"],
                "candidate_hv_loss": candidate_stable["mean_loss_pct"],
            },
        },
        "actual_evaluations_used": _float_or_none(
            candidate_overall["actual_evaluations_used_mean"]
        ),
        "runtime_seconds": _float_or_none(candidate_overall["runtime_seconds_mean"]),
        "decision_label": decision_label,
        "rationale": rationale,
        "recommended_next_action": next_action,
        "metrics_rows": metrics_rows,
    }


def _evaluate_knapsack_candidate(
    manifest: dict[str, Any],
    candidate_output_dir: Path,
) -> dict[str, Any]:
    rows = _read_csv_rows(candidate_output_dir / "summary.csv")
    baseline_variant = manifest.get("baseline_variant", "greedy_local_search")
    candidate_variant = manifest.get("candidate_variant", "repair_only")
    comparison_variants = manifest.get("comparison_variants", ["none"])

    baseline_row = _find_row(rows, study_variant=baseline_variant)
    candidate_row = _find_row(rows, study_variant=candidate_variant)
    if not baseline_row or not candidate_row:
        raise ValueError("Knapsack candidate output is missing required baseline/candidate rows")
    comparison_rows = {
        variant: _find_row(rows, study_variant=variant) for variant in comparison_variants
    }

    best_delta_vs_baseline = (
        _float_or_none(candidate_row["best_feasible_fitness_mean"])
        - _float_or_none(baseline_row["best_feasible_fitness_mean"])
        if _float_or_none(candidate_row["best_feasible_fitness_mean"]) is not None
        and _float_or_none(baseline_row["best_feasible_fitness_mean"]) is not None
        else None
    )
    note_only = bool(manifest["decision_policy"].get("family_conditioned_only", True))
    broad_forbidden = bool(
        manifest["decision_policy"].get("broad_default_promotion_forbidden", True)
    )
    metrics_rows = [
        _result_row(
            "best_feasible_fitness_vs_greedy",
            baseline_row["best_feasible_fitness_mean"],
            candidate_row["best_feasible_fitness_mean"],
            higher_is_better=True,
        ),
        _result_row(
            "feasible_rate",
            baseline_row["feasible_rate"],
            candidate_row["feasible_rate"],
            higher_is_better=True,
        ),
        _result_row(
            "generations_to_first_feasible_mean",
            baseline_row["generations_to_first_feasible_mean"],
            candidate_row["generations_to_first_feasible_mean"],
            higher_is_better=False,
        ),
    ]
    for variant, row in comparison_rows.items():
        if row is not None:
            metrics_rows.append(
                _result_row(
                    f"best_feasible_fitness_vs_{variant}",
                    row["best_feasible_fitness_mean"],
                    candidate_row["best_feasible_fitness_mean"],
                    higher_is_better=True,
                )
            )

    if broad_forbidden and not note_only:
        decision_label = "reject_regression"
        rationale = (
            "Knapsack candidates may not claim a broad default inside the frozen local workflow."
        )
        next_action = "keep_narrow_note_on_subset_sum_tight_capacity_only"
    elif (best_delta_vs_baseline or 0.0) < 0.0:
        decision_label = "reject_regression"
        rationale = (
            "The candidate underperformed the greedy local baseline on the narrow guard slice."
        )
        next_action = "keep_narrow_note_on_subset_sum_tight_capacity_only"
    elif (
        comparison_rows.get("none") is not None
        and _float_or_none(candidate_row["best_feasible_fitness_mean"]) is not None
        and _float_or_none(comparison_rows["none"]["best_feasible_fitness_mean"]) is not None
        and _float_or_none(candidate_row["best_feasible_fitness_mean"])
        <= _float_or_none(comparison_rows["none"]["best_feasible_fitness_mean"])
    ):
        decision_label = "reject_no_material_gain"
        rationale = (
            "The candidate did not beat the plain none reference on the narrow repair-note slice."
        )
        next_action = "keep_narrow_note_on_subset_sum_tight_capacity_only"
    else:
        decision_label = "note_only_stress_slice"
        rationale = (
            "The candidate remains useful only as a narrow family-conditioned repair note, not as a broad default."
        )
        next_action = "keep_narrow_note_on_subset_sum_tight_capacity_only"

    return {
        "compared_profiles": {
            "baseline_variant": baseline_variant,
            "candidate_variant": candidate_variant,
            "comparison_variants": comparison_variants,
        },
        "primary_metric_delta": {
            "best_feasible_fitness_delta_vs_baseline": best_delta_vs_baseline,
        },
        "non_regression_results": {
            "feasible_rate_candidate": _float_or_none(candidate_row["feasible_rate"]),
            "generations_to_first_feasible_candidate": _float_or_none(
                candidate_row["generations_to_first_feasible_mean"]
            ),
        },
        "stress_slice_results": {
            "narrow_repair_note": {
                "baseline_best_feasible": baseline_row["best_feasible_fitness_mean"],
                "candidate_best_feasible": candidate_row["best_feasible_fitness_mean"],
            }
        },
        "actual_evaluations_used": _float_or_none(
            candidate_row["actual_evaluations_used_mean"]
        ),
        "runtime_seconds": _float_or_none(candidate_row["runtime_seconds_mean"]),
        "decision_label": decision_label,
        "rationale": rationale,
        "recommended_next_action": next_action,
        "metrics_rows": metrics_rows,
    }


def _evaluate_onemax_candidate(
    manifest: dict[str, Any],
    candidate_output_dir: Path,
) -> dict[str, Any]:
    rows = _read_csv_rows(candidate_output_dir / "summary.csv")
    baseline_variant = manifest.get("baseline_variant", "none")
    candidate_variant = manifest.get("candidate_variant", baseline_variant)
    baseline_row = _find_row(rows, study_variant=baseline_variant)
    candidate_row = _find_row(rows, study_variant=candidate_variant)
    if not baseline_row or not candidate_row:
        raise ValueError("OneMax candidate output is missing required rows")
    metrics_rows = [
        _result_row(
            "evaluations_to_target_mean",
            baseline_row["evaluations_to_target_mean"],
            candidate_row["evaluations_to_target_mean"],
            higher_is_better=False,
        ),
        _result_row(
            "target_hit_rate",
            baseline_row["target_hit_rate"],
            candidate_row["target_hit_rate"],
            higher_is_better=True,
        ),
    ]
    return {
        "compared_profiles": {
            "baseline_variant": baseline_variant,
            "candidate_variant": candidate_variant,
        },
        "primary_metric_delta": {
            "evaluations_to_target_delta": (
                _float_or_none(candidate_row["evaluations_to_target_mean"])
                - _float_or_none(baseline_row["evaluations_to_target_mean"])
                if _float_or_none(candidate_row["evaluations_to_target_mean"]) is not None
                and _float_or_none(baseline_row["evaluations_to_target_mean"]) is not None
                else None
            ),
        },
        "non_regression_results": {
            "target_hit_rate": candidate_row["target_hit_rate"],
        },
        "stress_slice_results": {
            "control": {
                "baseline": baseline_variant,
                "candidate": candidate_variant,
            }
        },
        "actual_evaluations_used": _float_or_none(
            candidate_row["actual_evaluations_used_mean"]
        ),
        "runtime_seconds": _float_or_none(candidate_row["runtime_seconds_mean"]),
        "decision_label": "monitor_only",
        "rationale": "OneMax remains a control-only branch with no active candidate target.",
        "recommended_next_action": "no_active_target",
        "metrics_rows": metrics_rows,
    }


def _evaluate_candidate_metrics(
    manifest: dict[str, Any],
    candidate_output_dir: Path,
) -> dict[str, Any]:
    problem = manifest["problem"]
    if problem == "tsp":
        return _evaluate_tsp_candidate(manifest, candidate_output_dir)
    if problem == "zdt1":
        return _evaluate_zdt1_candidate(manifest, candidate_output_dir)
    if problem == "knapsack":
        return _evaluate_knapsack_candidate(manifest, candidate_output_dir)
    if problem == "onemax":
        return _evaluate_onemax_candidate(manifest, candidate_output_dir)
    raise ValueError(f"Unsupported candidate problem: {problem}")


def _render_candidate_report_markdown(report: dict[str, Any]) -> str:
    metrics_rows = report.get("candidate_vs_baseline_rows", [])
    lines = [
        "# Local Candidate Report",
        "",
        f"- Candidate ID: `{report['candidate_id']}`",
        f"- Problem: `{report['problem']}`",
        f"- Target: `{report['target_id']}`",
        f"- Baseline status: `{report['baseline_status']}`",
        f"- Decision label: `{report['decision_label']}`",
        "",
        "## Rationale",
        "",
        report["rationale"],
        "",
        "## Compared Profiles",
        "",
        f"```json\n{json.dumps(report['compared_profiles'], indent=2, ensure_ascii=False)}\n```",
        "",
        "## Primary Metric Delta",
        "",
        f"```json\n{json.dumps(report['primary_metric_delta'], indent=2, ensure_ascii=False)}\n```",
        "",
        "## Non-Regression Read",
        "",
        f"```json\n{json.dumps(report['non_regression_results'], indent=2, ensure_ascii=False)}\n```",
        "",
        "## Stress Slice Read",
        "",
        f"```json\n{json.dumps(report['stress_slice_results'], indent=2, ensure_ascii=False)}\n```",
        "",
    ]
    if metrics_rows:
        lines.extend(
            [
                "## Candidate vs Baseline",
                "",
                _markdown_table(
                    metrics_rows,
                    ["metric", "baseline_value", "candidate_value", "delta"],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Next Action",
            "",
            f"- `{report['recommended_next_action']}`",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_candidate_manifest(
    manifest_or_ref: dict[str, Any] | str | Path,
    *,
    project_root: Path | None = None,
    output_root: str | Path = "outputs/local_candidates",
    no_execute: bool = False,
    use_existing_output: bool = False,
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT
    manifest = (
        validate_candidate_manifest(manifest_or_ref, project_root=root)
        if isinstance(manifest_or_ref, dict)
        else load_candidate_manifest(manifest_or_ref, project_root=root)
    )
    output_dir = _candidate_output_dir(
        manifest["candidate_id"],
        project_root=root,
        output_root=output_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_snapshot_path = _resolve_path(manifest["baseline_snapshot_path"], project_root=root)
    baseline_report = check_local_baseline(
        project_root=root,
        snapshot_path=baseline_snapshot_path,
        output_json_path=output_dir / "baseline_check.json",
        output_md_path=output_dir / "baseline_check.md",
    )
    blocking_baseline_drift = _is_blocking_baseline_drift(baseline_report)

    candidate_output_dir, executed_study = _load_or_run_candidate_output(
        manifest,
        project_root=root,
        no_execute=no_execute,
        use_existing_output=use_existing_output,
    )

    if blocking_baseline_drift:
        comparison_payload = {
            "compared_profiles": manifest["candidate_profile_or_overrides"],
            "primary_metric_delta": {},
            "non_regression_results": {},
            "stress_slice_results": {},
            "actual_evaluations_used": None,
            "runtime_seconds": None,
            "decision_label": "baseline_drift_detected",
            "rationale": (
                "The frozen baseline snapshot does not match the current profiles/protocol/target registry, so candidate comparison is blocked until the drift is resolved."
            ),
            "recommended_next_action": "resolve_baseline_drift_before_candidate_review",
            "metrics_rows": [],
        }
    elif candidate_output_dir is None:
        comparison_payload = {
            "compared_profiles": manifest["candidate_profile_or_overrides"],
            "primary_metric_delta": {},
            "non_regression_results": {},
            "stress_slice_results": {},
            "actual_evaluations_used": None,
            "runtime_seconds": None,
            "decision_label": manifest.get("expected_decision_label", "monitor_only"),
            "rationale": (
                "No candidate output was executed in this run, so the report stays as a manifest-only check."
            ),
            "recommended_next_action": "provide_existing_output_or_run_candidate_study",
            "metrics_rows": [],
        }
    else:
        comparison_payload = _evaluate_candidate_metrics(manifest, candidate_output_dir)

    baseline_snapshot_hash = _sha256(baseline_snapshot_path)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_id": manifest["candidate_id"],
        "problem": manifest["problem"],
        "target_id": manifest["target_id"],
        "hypothesis_id": manifest["hypothesis_id"],
        "baseline_snapshot_hash": baseline_snapshot_hash,
        "baseline_status": baseline_report["status"],
        "baseline_guard_study": manifest["baseline_guard_study"],
        "candidate_study": manifest.get("candidate_study"),
        "candidate_output_dir": str(candidate_output_dir.resolve()) if candidate_output_dir else None,
        "candidate_source_path": manifest.get("source_path"),
        "compared_profiles": comparison_payload["compared_profiles"],
        "primary_metric_delta": comparison_payload["primary_metric_delta"],
        "non_regression_results": comparison_payload["non_regression_results"],
        "stress_slice_results": comparison_payload["stress_slice_results"],
        "actual_evaluations_used": comparison_payload["actual_evaluations_used"],
        "runtime_seconds": comparison_payload["runtime_seconds"],
        "decision_label": comparison_payload["decision_label"],
        "rationale": comparison_payload["rationale"],
        "recommended_next_action": comparison_payload["recommended_next_action"],
        "expected_decision_label": manifest.get("expected_decision_label"),
        "decision_matches_expectation": (
            comparison_payload["decision_label"] == manifest.get("expected_decision_label")
            if manifest.get("expected_decision_label")
            else None
        ),
        "executed_study": executed_study,
        "candidate_vs_baseline_rows": comparison_payload["metrics_rows"],
        "output_paths": {
            "candidate_report_json": str((output_dir / "candidate_report.json").resolve()),
            "candidate_report_md": str((output_dir / "candidate_report.md").resolve()),
            "candidate_summary_csv": str((output_dir / "candidate_summary.csv").resolve()),
            "candidate_vs_baseline_csv": str(
                (output_dir / "candidate_vs_baseline.csv").resolve()
            ),
            "baseline_check_json": str((output_dir / "baseline_check.json").resolve()),
            "baseline_check_md": str((output_dir / "baseline_check.md").resolve()),
        },
    }

    _write_json(output_dir / "candidate_report.json", report)
    (output_dir / "candidate_report.md").write_text(
        _render_candidate_report_markdown(report),
        encoding="utf-8",
    )

    summary_fields = [
        "candidate_id",
        "problem",
        "target_id",
        "baseline_status",
        "decision_label",
        "expected_decision_label",
        "decision_matches_expectation",
        "actual_evaluations_used",
        "runtime_seconds",
        "recommended_next_action",
    ]
    _write_rows_csv(
        output_dir / "candidate_summary.csv",
        [report],
        summary_fields,
    )
    _write_rows_csv(
        output_dir / "candidate_vs_baseline.csv",
        comparison_payload["metrics_rows"],
        ["metric", "baseline_value", "candidate_value", "delta", "higher_is_better"],
    )
    return report
