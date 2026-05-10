from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


CATALOG_COLUMNS = [
    "problem",
    "case_id",
    "instance_label",
    "seed",
    "budget_band",
    "compared_profiles",
    "regret_or_loss",
    "case_group",
    "why_selected",
    "future_target_label",
]

TARGET_COLUMNS = [
    "target_id",
    "problem",
    "current_default_affected",
    "failure_metric",
    "case_group",
    "severity",
    "estimated_optimization_value",
    "keep_as_regression_case",
    "suggested_next_pass",
    "notes",
    "evidence_count",
    "max_observed_loss_or_regret",
    "example_case_ids",
]


def _parse_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        columns = list(fieldnames or [])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if columns:
                writer.writeheader()
        return

    ordered_columns: list[str] = list(fieldnames or [])
    seen = set(ordered_columns)
    for row in rows:
        for key in row:
            if key not in seen:
                ordered_columns.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ordered_columns})


def _markdown_table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _infer_study_name(study_dir: Path) -> str:
    parts = study_dir.name.split("_", 1)
    return parts[1] if len(parts) == 2 else study_dir.name


def _infer_problem_from_study_name(study_name: str) -> str:
    prefix = study_name.split("_", 1)[0].lower()
    if prefix in {"tsp", "zdt1", "knapsack", "onemax"}:
        return prefix
    return "unknown"


def _infer_future_target_label(row: dict[str, Any]) -> str:
    problem = str(row.get("problem") or "").lower()
    case_group = str(row.get("case_group") or "").lower()
    why_selected = str(
        row.get("future_target_label")
        or row.get("why_selected")
        or row.get("why_selected_as_stress_case")
        or ""
    ).lower()
    if problem == "tsp":
        if "anti_case" in case_group:
            return "tsp_fast_anti_case_tail"
        if "rescue" in case_group or "ambigu" in why_selected or "decision" in why_selected:
            return "tsp_rescue_target_ambiguity"
        return "tsp_fast_general_tail"
    if problem == "zdt1":
        joint_fail = _parse_float(row.get("joint_safety_fail")) or 0.0
        spread_fail = _parse_float(row.get("spread_safety_fail")) or 0.0
        pareto_fail = _parse_float(row.get("pareto_ratio_safety_fail")) or 0.0
        if joint_fail > 0.0 or "joint" in why_selected:
            return "zdt1_fast_joint_safety_fail"
        if spread_fail > 0.0 or pareto_fail > 0.0 or "spread" in why_selected or "safety" in case_group:
            return "zdt1_fast_spread_safety_fail"
        if "ambigu" in why_selected or "decision" in why_selected:
            return "zdt1_fast_joint_safety_fail"
        return "zdt1_fast_hv_tail"
    if problem == "knapsack":
        if case_group in {"subset_sum_like_small", "tight_capacity_small"}:
            return "knapsack_repair_boundary_subset_sum_tight_capacity"
        if "borderline" in why_selected or case_group in {"weakly_correlated_small", "uncorrelated_small"}:
            return "knapsack_repair_borderline_family"
        return "knapsack_repair_note_regression"
    if problem == "onemax":
        return "onemax_no_active_target"
    return f"{problem}_future_target"


def discover_latest_study_dirs(search_root: Path, study_names: Sequence[str]) -> list[Path]:
    directories: list[Path] = []
    for study_name in study_names:
        candidates = sorted(
            path
            for path in search_root.iterdir()
            if path.is_dir() and path.name.endswith(f"_{study_name}")
        )
        if not candidates:
            raise FileNotFoundError(
                f"Could not find an output directory for study '{study_name}' under {search_root}."
            )
        directories.append(candidates[-1])
    return directories


def _load_study_bundle(study_dir: Path) -> dict[str, Any]:
    study_name = _infer_study_name(study_dir)
    problem = _infer_problem_from_study_name(study_name)
    catalog_path = study_dir / "stress_case_catalog.csv"
    tail_path = study_dir / "tail_risk_summary.csv"
    summary_path = study_dir / "summary.csv"
    return {
        "study_dir": study_dir,
        "study_name": study_name,
        "problem": problem,
        "catalog_rows": _read_csv_rows(catalog_path) if catalog_path.exists() else [],
        "tail_rows": _read_csv_rows(tail_path) if tail_path.exists() else [],
        "summary_rows": _read_csv_rows(summary_path) if summary_path.exists() else [],
    }


def _normalize_catalog_rows(study_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    study_name = str(study_bundle["study_name"])
    for row in study_bundle["catalog_rows"]:
        normalized = {
            "study_name": study_name,
            "problem": row.get("problem") or study_bundle["problem"],
            "case_id": row.get("case_id") or row.get("instance_label") or study_name,
            "instance_label": row.get("instance_label") or row.get("case_id") or study_name,
            "seed": row.get("seed") or "",
            "budget_band": row.get("budget_band") or "",
            "compared_profiles": row.get("compared_profiles") or row.get("profile_compared") or "",
            "regret_or_loss": _parse_float(row.get("regret_or_loss")),
            "case_group": row.get("case_group") or "overall",
            "why_selected": row.get("why_selected") or row.get("why_selected_as_stress_case") or "",
            "future_target_label": row.get("future_target_label") or _infer_future_target_label(row),
        }
        rows.append(normalized)
    rows.sort(
        key=lambda item: (
            str(item.get("problem") or ""),
            -abs(item.get("regret_or_loss") or 0.0),
            str(item.get("case_group") or ""),
            str(item.get("case_id") or ""),
        )
    )
    return rows


def _normalize_tail_rows(study_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    study_name = str(study_bundle["study_name"])
    for row in study_bundle["tail_rows"]:
        normalized = {
            "study_name": study_name,
            "problem": study_bundle["problem"],
            "scope": row.get("scope") or "",
            "case_group": row.get("case_group") or "overall",
            "study_variant": row.get("study_variant") or "",
            "mean_loss_pct": _parse_float(row.get("mean_loss_pct")),
            "p90_loss_pct": _parse_float(row.get("p90_loss_pct")),
            "p95_loss_pct": _parse_float(row.get("p95_loss_pct")),
            "max_loss_pct": _parse_float(row.get("max_loss_pct")),
            "decision_flip_rate": _parse_float(row.get("decision_flip_rate")),
            "spread_fail_rate": _parse_float(row.get("spread_fail_rate")),
            "joint_safety_fail_rate": _parse_float(row.get("joint_safety_fail_rate")),
            "runtime_seconds_mean": _parse_float(row.get("runtime_seconds_mean")),
            "actual_evaluations_used_mean": _parse_float(row.get("actual_evaluations_used_mean")),
        }
        rows.append(normalized)
    return rows


def _target_template(target_id: str) -> dict[str, Any]:
    templates: dict[str, dict[str, Any]] = {
        "tsp_fast_anti_case_tail": {
            "problem": "tsp",
            "current_default_affected": "configs/local_profiles/tsp_seeded_swap_local_fast.json",
            "failure_metric": "route_distance_loss_pct",
            "case_group": "anti_case",
            "severity": "high",
            "estimated_optimization_value": "high",
            "keep_as_regression_case": True,
            "suggested_next_pass": "Reduce corridor-like anti-case tail under the current TSP fast budget.",
            "notes": "Current hardened TSP F still saves budget well, but corridor-like anti-cases dominate the p90/p95 tail.",
        },
        "tsp_rescue_target_ambiguity": {
            "problem": "tsp",
            "current_default_affected": "configs/local_profiles/tsp_seeded_swap_local_fast.json",
            "failure_metric": "route_distance_loss_pct",
            "case_group": "rescue_target",
            "severity": "medium",
            "estimated_optimization_value": "medium",
            "keep_as_regression_case": True,
            "suggested_next_pass": "Trim rescue-target ambiguity without reopening the anti-case tail.",
            "notes": "Bridge/ring-like rescue targets still create compare-stage ambiguity even after hardening.",
        },
        "zdt1_fast_spread_safety_fail": {
            "problem": "zdt1",
            "current_default_affected": "configs/local_profiles/zdt1_diversity_injection_fast.json",
            "failure_metric": "spread_delta",
            "case_group": "safety_fail",
            "severity": "high",
            "estimated_optimization_value": "high",
            "keep_as_regression_case": True,
            "suggested_next_pass": "Reduce spread-tail degradation while preserving the current fast HV advantage.",
            "notes": "The hardened ZDT1 F still holds HV fairly well, but spread degradation remains the most visible safety failure.",
        },
        "zdt1_fast_joint_safety_fail": {
            "problem": "zdt1",
            "current_default_affected": "configs/local_profiles/zdt1_diversity_injection_fast.json",
            "failure_metric": "joint_safety_fail_rate",
            "case_group": "safety_fail",
            "severity": "high",
            "estimated_optimization_value": "high",
            "keep_as_regression_case": True,
            "suggested_next_pass": "Target joint safety failures before broadening the ZDT1 fast workflow.",
            "notes": "Joint safety failures are the main reason final safety still routes directly to Q.",
        },
        "zdt1_fast_hv_tail": {
            "problem": "zdt1",
            "current_default_affected": "configs/local_profiles/zdt1_diversity_injection_fast.json",
            "failure_metric": "hv_loss_pct",
            "case_group": "hv_tail",
            "severity": "medium",
            "estimated_optimization_value": "medium",
            "keep_as_regression_case": True,
            "suggested_next_pass": "Only revisit HV tail if spread/joint safety stops dominating the failure budget.",
            "notes": "HV tails exist, but they look secondary to spread and joint safety failures in the refreshed defaults.",
        },
        "knapsack_repair_boundary_subset_sum_tight_capacity": {
            "problem": "knapsack",
            "current_default_affected": "configs/local_profiles/knapsack_repair_local_experimental.json",
            "failure_metric": "best_feasible_fitness",
            "case_group": "subset_sum_like_small/tight_capacity_small",
            "severity": "medium",
            "estimated_optimization_value": "medium",
            "keep_as_regression_case": True,
            "suggested_next_pass": "Keep the repair-only note bounded to subset-sum-like and tight-capacity sanity reruns.",
            "notes": "This remains the narrow family slice where repair-only still shows enough value to remember.",
        },
        "knapsack_repair_borderline_family": {
            "problem": "knapsack",
            "current_default_affected": "configs/local_profiles/knapsack_repair_local_experimental.json",
            "failure_metric": "best_feasible_fitness",
            "case_group": "borderline",
            "severity": "low",
            "estimated_optimization_value": "low",
            "keep_as_regression_case": True,
            "suggested_next_pass": "Only revisit if a future repair tweak clearly helps weakly correlated borderline families.",
            "notes": "Borderline weakly correlated ties are worth cataloging, but not worth reopening as a broad knapsack default search.",
        },
        "knapsack_repair_note_regression": {
            "problem": "knapsack",
            "current_default_affected": "configs/local_profiles/knapsack_repair_local_experimental.json",
            "failure_metric": "best_feasible_fitness",
            "case_group": "repair_note",
            "severity": "low",
            "estimated_optimization_value": "low",
            "keep_as_regression_case": True,
            "suggested_next_pass": "Keep the note parked unless stress refresh evidence becomes materially stronger.",
            "notes": "Broad park still dominates the honest interpretation for knapsack.",
        },
        "onemax_no_active_target": {
            "problem": "onemax",
            "current_default_affected": "none",
            "failure_metric": "evaluations_to_target",
            "case_group": "control",
            "severity": "low",
            "estimated_optimization_value": "low",
            "keep_as_regression_case": False,
            "suggested_next_pass": "No active target; keep OneMax as a control-only branch.",
            "notes": "The control branch is stable enough that it should not drive future optimization work.",
        },
    }
    return dict(templates.get(target_id, {}))


def build_future_target_registry(catalog_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in catalog_rows:
        target_id = str(row.get("future_target_label") or "")
        grouped.setdefault(target_id, []).append(row)

    targets: list[dict[str, Any]] = []
    for target_id, rows in sorted(grouped.items()):
        template = _target_template(target_id)
        losses = [abs(float(row["regret_or_loss"])) for row in rows if row.get("regret_or_loss") is not None]
        max_loss = max(losses) if losses else 0.0
        unique_cases = sorted({str(row.get("case_id") or row.get("instance_label") or "") for row in rows if row.get("case_id") or row.get("instance_label")})
        target = {
            "target_id": target_id,
            "problem": template.get("problem") or (rows[0].get("problem") if rows else ""),
            "current_default_affected": template.get("current_default_affected", ""),
            "failure_metric": template.get("failure_metric", "regret_or_loss"),
            "case_group": template.get("case_group") or (rows[0].get("case_group") if rows else ""),
            "severity": template.get("severity", "medium"),
            "estimated_optimization_value": template.get("estimated_optimization_value", "medium"),
            "keep_as_regression_case": template.get("keep_as_regression_case", True),
            "suggested_next_pass": template.get("suggested_next_pass", ""),
            "notes": template.get("notes", ""),
            "evidence_count": len(rows),
            "max_observed_loss_or_regret": max_loss,
            "example_case_ids": ", ".join(unique_cases[:4]),
        }
        targets.append(target)

    order = {name: index for index, name in enumerate(
        [
            "tsp_fast_anti_case_tail",
            "tsp_rescue_target_ambiguity",
            "zdt1_fast_spread_safety_fail",
            "zdt1_fast_joint_safety_fail",
            "zdt1_fast_hv_tail",
            "knapsack_repair_boundary_subset_sum_tight_capacity",
            "knapsack_repair_borderline_family",
            "knapsack_repair_note_regression",
            "onemax_no_active_target",
        ]
    )}
    targets.sort(key=lambda row: (order.get(str(row["target_id"]), 999), str(row["target_id"])))
    return targets


def _render_catalog_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "problem",
        "case_id",
        "seed",
        "budget_band",
        "compared_profiles",
        "regret_or_loss",
        "case_group",
        "why_selected",
        "future_target_label",
    ]
    return "# Current Stress Case Catalog\n\n" + _markdown_table(rows, columns) + "\n"


def _render_targets_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "target_id",
        "problem",
        "failure_metric",
        "case_group",
        "severity",
        "estimated_optimization_value",
        "keep_as_regression_case",
        "suggested_next_pass",
    ]
    return "# Future Optimization Targets\n\n" + _markdown_table(rows, columns) + "\n"


def _render_notes_markdown(target_rows: list[dict[str, Any]]) -> str:
    target_map = {str(row["target_id"]): row for row in target_rows}
    lines = [
        "# Stress Refresh Notes",
        "",
        "## TSP",
        "- Current defaults stay fixed: Q remains quality-first, F remains the hardened budget-first default.",
        "- Anti-case corridor-like tails still dominate the future target list; rescue-target ambiguity stays secondary.",
        "- Protocol implication: keep the descriptive split. Anti-case suspicion still justifies routing straight to Q.",
        "",
        "## ZDT1",
        "- Current defaults stay fixed: hardened F is still useful for exploratory HV sanity, but final safety still belongs to Q.",
        "- The refreshed target list shows spread and joint safety failures ahead of plain HV tail.",
        "- Protocol implication: the 'final safety is Q' wording stays correct and should stay explicit.",
        "",
        "## Knapsack",
        "- Broad default remains parked. The only note worth keeping is the narrow repair-only sanity check.",
        "- Protocol implication: keep repair_only bounded to subset-sum-like and tight-capacity sanity cases; leave everything else parked.",
        "",
        "## Onemax",
        "- OneMax remains a control-only branch with no active optimization target.",
    ]
    if target_map:
        lines.extend(
            [
                "",
                "## Top Target Summary",
                f"- Primary TSP target: `{next((key for key in target_map if key.startswith('tsp_')), 'none')}`",
                f"- Primary ZDT1 target: `{next((key for key in target_map if key.startswith('zdt1_')), 'none')}`",
                f"- Knapsack note target: `{next((key for key in target_map if key.startswith('knapsack_')), 'none')}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def build_stress_refresh_registry(
    study_dirs: Sequence[Path],
    output_dir: Path,
) -> dict[str, str]:
    bundles = [_load_study_bundle(path) for path in study_dirs]
    catalog_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        catalog_rows.extend(_normalize_catalog_rows(bundle))
        tail_rows.extend(_normalize_tail_rows(bundle))

    target_rows = build_future_target_registry(catalog_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    current_catalog_csv = output_dir / "current_stress_case_catalog.csv"
    current_catalog_md = output_dir / "current_stress_case_catalog.md"
    tail_summary_csv = output_dir / "tail_risk_refresh_summary.csv"
    targets_csv = output_dir / "future_optimization_targets.csv"
    targets_json = output_dir / "future_optimization_targets.json"
    targets_md = output_dir / "future_optimization_targets.md"
    notes_md = output_dir / "stress_refresh_notes.md"

    _write_rows_csv(current_catalog_csv, catalog_rows, ["study_name", *CATALOG_COLUMNS])
    current_catalog_md.write_text(_render_catalog_markdown(catalog_rows), encoding="utf-8")
    _write_rows_csv(tail_summary_csv, tail_rows)
    _write_rows_csv(targets_csv, target_rows, TARGET_COLUMNS)
    targets_json.write_text(json.dumps(target_rows, indent=2), encoding="utf-8")
    targets_md.write_text(_render_targets_markdown(target_rows), encoding="utf-8")
    notes_md.write_text(_render_notes_markdown(target_rows), encoding="utf-8")

    return {
        "current_stress_case_catalog_csv": str(current_catalog_csv.resolve()),
        "current_stress_case_catalog_md": str(current_catalog_md.resolve()),
        "tail_risk_refresh_summary_csv": str(tail_summary_csv.resolve()),
        "future_optimization_targets_csv": str(targets_csv.resolve()),
        "future_optimization_targets_json": str(targets_json.resolve()),
        "future_optimization_targets_md": str(targets_md.resolve()),
        "stress_refresh_notes_md": str(notes_md.resolve()),
    }
