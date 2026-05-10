from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence


TARGET_COLUMNS = [
    "target_id",
    "problem",
    "current_default_affected",
    "failure_metric",
    "case_group",
    "previous_severity",
    "current_severity",
    "changed_or_not",
    "recommended_next_action",
    "keep_as_regression_case",
    "estimated_optimization_value",
    "notes",
]

PRIORITY_ORDER = {
    "tsp_fast_anti_case_tail": 0,
    "zdt1_fast_spread_safety_fail": 1,
    "zdt1_fast_joint_safety_fail": 2,
    "tsp_rescue_target_ambiguity": 3,
    "knapsack_repair_boundary_subset_sum_tight_capacity": 4,
    "knapsack_repair_borderline_family": 5,
    "knapsack_repair_note_regression": 6,
    "zdt1_fast_hv_tail": 7,
    "onemax_no_active_target": 8,
}


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


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _markdown_table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        rendered = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                rendered.append(f"{value:.6g}")
            else:
                rendered.append(str(value))
        body.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *body])


def _load_previous_targets(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _study_name(study_dir: Path) -> str:
    parts = study_dir.name.split("_", 1)
    return parts[1] if len(parts) == 2 else study_dir.name


def _load_tail_rows(study_dir: Path) -> list[dict[str, str]]:
    reduction_path = study_dir / "tail_risk_reduction_summary.csv"
    if reduction_path.exists():
        return _read_csv_rows(reduction_path)
    tail_path = study_dir / "tail_risk_summary.csv"
    if tail_path.exists():
        return _read_csv_rows(tail_path)
    return []


def _overall_row(rows: list[dict[str, str]], variant: str) -> dict[str, str] | None:
    for row in rows:
        if str(row.get("scope") or "overall") == "overall" and str(row.get("study_variant") or "") == variant:
            return row
    return None


def _best_tsp_candidate(rows: list[dict[str, str]], baseline_variant: str) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if str(row.get("scope") or "overall") == "overall"
        and str(row.get("study_variant") or "") not in {"quality_first", baseline_variant}
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            _parse_float(row.get("anti_case_p95_loss_pct")) or _parse_float(row.get("anti_case_p90_loss_pct")) or 999.0,
            _parse_float(row.get("anti_case_p90_loss_pct")) or 999.0,
            _parse_float(row.get("anti_case_max_loss_pct")) or _parse_float(row.get("max_loss_pct")) or 999.0,
            _parse_float(row.get("mean_loss_pct")) or 999.0,
        ),
    )


def _best_zdt1_candidate(rows: list[dict[str, str]], baseline_variant: str) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if str(row.get("scope") or "overall") == "overall"
        and str(row.get("study_variant") or "") not in {"quality_first", baseline_variant}
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            _parse_float(row.get("joint_safety_fail_rate")) or 999.0,
            _parse_float(row.get("spread_fail_rate")) or 999.0,
            _parse_float(row.get("p90_loss_pct")) or 999.0,
            _parse_float(row.get("mean_loss_pct")) or 999.0,
        ),
    )


def _tsp_severity(row: dict[str, str] | None) -> str:
    if row is None:
        return "high"
    p95 = _parse_float(row.get("anti_case_p95_loss_pct")) or _parse_float(row.get("anti_case_p90_loss_pct")) or 0.0
    max_loss = _parse_float(row.get("anti_case_max_loss_pct")) or _parse_float(row.get("max_loss_pct")) or 0.0
    if p95 >= 2.5 or max_loss >= 3.0:
        return "high"
    if p95 >= 1.0 or max_loss >= 2.0:
        return "medium"
    return "low"


def _zdt1_severity(row: dict[str, str] | None) -> str:
    if row is None:
        return "high"
    joint_fail = _parse_float(row.get("joint_safety_fail_rate")) or 0.0
    spread_fail = _parse_float(row.get("spread_fail_rate")) or 0.0
    p90 = _parse_float(row.get("p90_loss_pct")) or 0.0
    if joint_fail >= 0.2 or spread_fail >= 0.2 or p90 >= 0.3:
        return "high"
    if joint_fail > 0.0 or spread_fail > 0.0 or p90 >= 0.15:
        return "medium"
    return "low"


def _update_row(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    row = dict(base)
    row.update(changes)
    return row


def _tsp_updates(previous: dict[str, dict[str, Any]], study_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _load_tail_rows(study_dir)
    baseline = _overall_row(rows, "current_fast")
    best = _best_tsp_candidate(rows, "current_fast")
    baseline_p90 = _parse_float(baseline.get("anti_case_p90_loss_pct")) if baseline else None
    baseline_p95 = _parse_float(baseline.get("anti_case_p95_loss_pct")) if baseline else None
    baseline_rescue = _parse_float(baseline.get("rescue_target_mean_loss_pct")) if baseline else None
    best_p90 = _parse_float(best.get("anti_case_p90_loss_pct")) if best else None
    best_p95 = _parse_float(best.get("anti_case_p95_loss_pct")) if best else None
    best_rescue = _parse_float(best.get("rescue_target_mean_loss_pct")) if best else None
    reduced = (
        best is not None
        and best_p90 is not None
        and baseline_p90 is not None
        and best_p90 < baseline_p90 - 0.15
        and (best_p95 is None or baseline_p95 is None or best_p95 <= baseline_p95)
        and (best_rescue is None or baseline_rescue is None or best_rescue <= baseline_rescue + 0.25)
    )

    anti_case_prev = previous["tsp_fast_anti_case_tail"]
    anti_case_notes = (
        "Current baseline "
        f"`{baseline.get('study_variant')}` stayed the default because no candidate lowered anti-case p90/p95 enough "
        "under the tail-first promotion rule."
        if not reduced
        else (
            f"Candidate `{best.get('study_variant')}` reduced anti-case tail on the stress suite, "
            "but the target is not closed yet."
        )
    )
    anti_case_action = (
        "keep_current_fast_and_hold_tsp_anti_case_as_top_regression_target"
        if not reduced
        else f"reduced_but_not_closed; rerun `{best.get('study_variant')}` only if the next pass stays locked to the same corridor-like regression set"
    )
    anti_case_changed = "unchanged_keep_default" if not reduced else "reduced_but_not_closed"
    anti_case_current_row = baseline if not reduced else best
    rescue_prev = previous["tsp_rescue_target_ambiguity"]
    rescue_changed = "unchanged_keep_default"
    rescue_notes = (
        "Anti-case tail remained the dominant TSP target in this pass; rescue-target ambiguity stays secondary."
    )

    return {
        "tsp_fast_anti_case_tail": _update_row(
            anti_case_prev,
            previous_severity=anti_case_prev.get("severity", "high"),
            current_severity=_tsp_severity(anti_case_current_row),
            changed_or_not=anti_case_changed,
            recommended_next_action=anti_case_action,
            notes=anti_case_notes,
        ),
        "tsp_rescue_target_ambiguity": _update_row(
            rescue_prev,
            previous_severity=rescue_prev.get("severity", "medium"),
            current_severity=rescue_prev.get("severity", "medium"),
            changed_or_not=rescue_changed,
            recommended_next_action="keep_secondary_regression_slice; do not outrank corridor-like anti-case tail",
            notes=rescue_notes,
        ),
    }


def _zdt1_updates(previous: dict[str, dict[str, Any]], study_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _load_tail_rows(study_dir)
    baseline = _overall_row(rows, "current_fast")
    best = _best_zdt1_candidate(rows, "current_fast")
    baseline_joint = _parse_float(baseline.get("joint_safety_fail_rate")) if baseline else None
    baseline_spread = _parse_float(baseline.get("spread_fail_rate")) if baseline else None
    baseline_hv = _parse_float(baseline.get("mean_loss_pct")) if baseline else None
    best_joint = _parse_float(best.get("joint_safety_fail_rate")) if best else None
    best_spread = _parse_float(best.get("spread_fail_rate")) if best else None
    best_hv = _parse_float(best.get("mean_loss_pct")) if best else None
    reduced = (
        best is not None
        and (
            (best_joint is not None and baseline_joint is not None and best_joint < baseline_joint - 0.05)
            or (best_spread is not None and baseline_spread is not None and best_spread < baseline_spread - 0.05)
        )
        and (best_hv is None or baseline_hv is None or best_hv <= baseline_hv + 0.05)
    )

    spread_prev = previous["zdt1_fast_spread_safety_fail"]
    joint_prev = previous["zdt1_fast_joint_safety_fail"]
    hv_prev = previous.get("zdt1_fast_hv_tail")
    current_row = baseline if not reduced else best
    change_label = "unchanged_keep_default" if not reduced else "reduced_but_not_closed"
    action = (
        "keep_current_fast_and_leave_final_safety_on_q"
        if not reduced
        else f"reduced_but_not_closed; rerun `{best.get('study_variant')}` only if the next pass stays locked to spread/joint safety failures"
    )
    notes = (
        f"Current baseline `{baseline.get('study_variant')}` stayed the most honest budget-first default once HV preservation was kept in the check."
        if not reduced
        else f"Candidate `{best.get('study_variant')}` reduced spread/joint safety failures on the stress suite, but final safety still belongs to Q."
    )

    updates = {
        "zdt1_fast_spread_safety_fail": _update_row(
            spread_prev,
            previous_severity=spread_prev.get("severity", "high"),
            current_severity=_zdt1_severity(current_row),
            changed_or_not=change_label,
            recommended_next_action=action,
            notes=notes,
        ),
        "zdt1_fast_joint_safety_fail": _update_row(
            joint_prev,
            previous_severity=joint_prev.get("severity", "high"),
            current_severity=_zdt1_severity(current_row),
            changed_or_not=change_label,
            recommended_next_action=action,
            notes=notes,
        ),
    }
    if hv_prev is not None:
        updates["zdt1_fast_hv_tail"] = _update_row(
            hv_prev,
            previous_severity=hv_prev.get("severity", "medium"),
            current_severity=hv_prev.get("severity", "medium"),
            changed_or_not="unchanged_keep_default",
            recommended_next_action="keep_hv_tail_as_secondary_to_safety_failures",
            notes="HV tail stayed secondary to spread/joint safety failures in the stress-target reduction pass.",
        )
    return updates


def _knapsack_updates(previous: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    updates: dict[str, dict[str, Any]] = {}
    for target_id in [
        "knapsack_repair_boundary_subset_sum_tight_capacity",
        "knapsack_repair_borderline_family",
        "knapsack_repair_note_regression",
    ]:
        if target_id not in previous:
            continue
        prev = previous[target_id]
        updates[target_id] = _update_row(
            prev,
            previous_severity=prev.get("severity", "medium"),
            current_severity=prev.get("severity", "medium"),
            changed_or_not="unchanged_keep_default",
            recommended_next_action="keep_knapsack_broadly_parked_and_leave_repair_note_narrow",
            notes="Knapsack stayed on a tiny freeze check only; the narrow repair note remains the only honest exception.",
        )
    return updates


def _onemax_updates(previous: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if "onemax_no_active_target" not in previous:
        return {}
    prev = previous["onemax_no_active_target"]
    return {
        "onemax_no_active_target": _update_row(
            prev,
            previous_severity=prev.get("severity", "low"),
            current_severity="low",
            changed_or_not="unchanged_keep_default",
            recommended_next_action="no_active_target_keep_control",
            notes="Onemax stayed control-only in the stress-target reduction pass.",
        )
    }


def _render_targets_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "target_id",
        "previous_severity",
        "current_severity",
        "changed_or_not",
        "recommended_next_action",
    ]
    return "# Future Optimization Targets\n\n" + _markdown_table(rows, columns) + "\n"


def _render_notes_markdown(rows: list[dict[str, Any]]) -> str:
    row_map = {str(row["target_id"]): row for row in rows}
    lines = [
        "# Stress Reduction Notes",
        "",
        "## TSP",
        "- Current TSP Q/F split stays descriptive. This pass only checked whether the budget-first fast default could trim corridor-like anti-case tail on the fixed stress suite.",
        f"- `tsp_fast_anti_case_tail`: {row_map.get('tsp_fast_anti_case_tail', {}).get('changed_or_not', 'unknown')}.",
        "- Protocol implication: keep anti-case caution wording explicit unless a future pass clearly lowers the p90/p95 corridor tail without reopening rescue-target loss.",
        "",
        "## ZDT1",
        "- Current ZDT1 Q/F split also stays in place. This pass only checked whether the hardened fast default could trim spread/joint safety failures without giving back HV.",
        f"- `zdt1_fast_spread_safety_fail`: {row_map.get('zdt1_fast_spread_safety_fail', {}).get('changed_or_not', 'unknown')}.",
        "- Protocol implication: keep the 'final safety routes to Q' wording unless a future pass materially lowers joint safety failures.",
        "",
        "## Knapsack / Onemax",
        "- Knapsack remains broadly parked, with only the narrow repair-only note preserved.",
        "- Onemax remains control-only with no active optimization target.",
        "",
    ]
    return "\n".join(lines)


def build_stress_target_reduction_registry(
    previous_registry_path: Path,
    output_dir: Path,
    tsp_study_dir: Path | None = None,
    zdt1_study_dir: Path | None = None,
    include_knapsack_freeze: bool = True,
    include_onemax_freeze: bool = True,
) -> dict[str, str]:
    previous_rows = _load_previous_targets(previous_registry_path)
    previous = {str(row["target_id"]): row for row in previous_rows}
    updates: dict[str, dict[str, Any]] = {}
    if tsp_study_dir is not None:
        updates.update(_tsp_updates(previous, tsp_study_dir))
    if zdt1_study_dir is not None:
        updates.update(_zdt1_updates(previous, zdt1_study_dir))
    if include_knapsack_freeze:
        updates.update(_knapsack_updates(previous))
    if include_onemax_freeze:
        updates.update(_onemax_updates(previous))

    refreshed_rows: list[dict[str, Any]] = []
    for row in previous_rows:
        target_id = str(row["target_id"])
        refreshed = dict(row)
        refreshed.update(
            {
                "previous_severity": row.get("severity") or row.get("current_severity") or "",
                "current_severity": row.get("severity") or row.get("current_severity") or "",
                "changed_or_not": "unchanged_keep_default",
                "recommended_next_action": row.get("suggested_next_pass") or "",
            }
        )
        if target_id in updates:
            refreshed.update(updates[target_id])
        refreshed_rows.append(refreshed)

    refreshed_rows.sort(key=lambda row: (PRIORITY_ORDER.get(str(row["target_id"]), 999), str(row["target_id"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "future_optimization_targets.json"
    csv_path = output_dir / "future_optimization_targets.csv"
    md_path = output_dir / "future_optimization_targets.md"
    notes_path = output_dir / "stress_reduction_notes.md"

    json_path.write_text(json.dumps(refreshed_rows, indent=2), encoding="utf-8")
    _write_rows_csv(csv_path, refreshed_rows, TARGET_COLUMNS)
    md_path.write_text(_render_targets_markdown(refreshed_rows), encoding="utf-8")
    notes_path.write_text(_render_notes_markdown(refreshed_rows), encoding="utf-8")

    return {
        "future_optimization_targets_json": str(json_path.resolve()),
        "future_optimization_targets_csv": str(csv_path.resolve()),
        "future_optimization_targets_md": str(md_path.resolve()),
        "stress_reduction_notes_md": str(notes_path.resolve()),
    }
