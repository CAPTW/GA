from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


FAILURE_HYPOTHESIS_COLUMNS = [
    "hypothesis_id",
    "target_id",
    "problem",
    "affected_default",
    "evidence_rows",
    "suspected_mechanism",
    "supporting_trace_signals",
    "confidence",
    "current_evidence_strength",
    "probe_attempted",
    "confirmed_or_weakened",
    "recommended_next_probe",
    "recommended_next_action",
    "keep_as_regression_case",
    "notes",
]

FUTURE_TARGET_COLUMNS = [
    "target_id",
    "problem",
    "current_default_affected",
    "failure_metric",
    "case_group",
    "previous_severity",
    "current_severity",
    "changed_or_not",
    "current_mechanism_hypothesis",
    "current_best_hypothesis",
    "latest_decision",
    "recommended_next_action",
    "next_recommended_probe",
    "keep_as_regression_case",
    "estimated_optimization_value",
    "notes",
]

_TARGET_PRIORITY = {
    "tsp_fast_anti_case_tail": 0,
    "zdt1_fast_spread_safety_fail": 1,
    "zdt1_fast_joint_safety_fail": 2,
    "tsp_rescue_target_ambiguity": 3,
    "knapsack_repair_boundary_subset_sum_tight_capacity": 4,
    "knapsack_repair_borderline_family": 5,
    "knapsack_repair_note_regression": 6,
    "onemax_no_active_target": 7,
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


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _quantile(values: Sequence[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if q <= 0.0:
        return ordered[0]
    if q >= 1.0:
        return ordered[-1]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean_or_none(values: Sequence[float]) -> float | None:
    ordered = list(values)
    return mean(ordered) if ordered else None


def _numeric_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _parse_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _signal_rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    matches = sum(1 for row in rows if (_parse_float(row.get(key)) or 0.0) > 0.0)
    return matches / len(rows)


def _confidence_from_rate(rate: float) -> str:
    if rate >= 0.6:
        return "high"
    if rate >= 0.3:
        return "medium"
    return "low"


def _probe_status_rank(value: str) -> int:
    return {
        "strengthened": 0,
        "partially_supported": 1,
        "mixed": 2,
        "weakened": 3,
        "secondary_only": 4,
        "still_unisolated": 5,
        "not_tested": 6,
    }.get(value, 7)


def _variant_rows(
    rows: Sequence[Mapping[str, Any]],
    variant: str,
    *,
    target_id: str | None = None,
    case_group: str | None = None,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        if str(row.get("study_variant") or "") != variant:
            continue
        if target_id is not None and str(row.get("target_id") or "") != target_id:
            continue
        if case_group is not None and str(row.get("case_group") or "") != case_group:
            continue
        selected.append(row)
    return selected


def _format_metric(value: float | None) -> str:
    return "na" if value is None else f"{value:.4f}"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        columns = list(fieldnames or [])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if columns:
                writer.writeheader()
        return

    ordered = list(fieldnames or [])
    seen = set(ordered)
    for row in rows:
        for key in row:
            if key not in seen:
                ordered.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ordered})


def _markdown_table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for row in rows:
        rendered: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                rendered.append(f"{value:.6g}")
            else:
                rendered.append(str(value))
        body.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *body])


def _pair_key(problem: str, row: Mapping[str, Any]) -> tuple[str, int] | None:
    seed = _parse_int(row.get("seed"))
    if seed is None:
        return None
    if problem == "tsp":
        case_id = str(row.get("case_id") or "").strip()
        return (case_id or "__study__", seed)
    return ("__study__", seed)


def _study_variant_key(row: Mapping[str, Any]) -> str:
    for key in ("study_variant", "variant_label"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _history_rows_from_path(path_value: Any) -> list[dict[str, Any]]:
    if not isinstance(path_value, str) or not path_value.strip():
        return []
    path = Path(path_value.strip())
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in raw.items():
                if value is None:
                    parsed[key] = ""
                    continue
                number = _parse_float(value)
                parsed[key] = number if number is not None else value
            rows.append(parsed)
    return rows


def _closest_row(rows: Sequence[Mapping[str, Any]], generation: int) -> Mapping[str, Any] | None:
    if not rows:
        return None
    best: Mapping[str, Any] | None = None
    best_gap: float | None = None
    for row in rows:
        row_generation = _parse_int(row.get("generation"))
        if row_generation is None:
            continue
        gap = abs(row_generation - generation)
        if best is None or best_gap is None or gap < best_gap:
            best = row
            best_gap = float(gap)
    return best


def _first_generation(rows: Sequence[Mapping[str, Any]], predicate) -> int | None:
    for row in rows:
        generation = _parse_int(row.get("generation"))
        if generation is None:
            continue
        if predicate(row):
            return generation
    return None


def _final_generation(rows: Sequence[Mapping[str, Any]]) -> int | None:
    if not rows:
        return None
    return _parse_int(rows[-1].get("generation"))


def _history_value(rows: Sequence[Mapping[str, Any]], key: str, default: float | None = None) -> float | None:
    if not rows:
        return default
    value = _parse_float(rows[-1].get(key))
    return default if value is None else value


def _first_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return _parse_float(rows[0].get(key))


def _tsp_trace_row(
    row: Mapping[str, Any],
    quality_row: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    quality_history_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    target_id: str,
) -> dict[str, Any] | None:
    quality_metric = _parse_float(quality_row.get("best_route_distance"))
    candidate_metric = _parse_float(row.get("best_route_distance"))
    if quality_metric is None or candidate_metric is None or quality_metric <= 0.0:
        return None

    collapse_threshold = _parse_float(config.get("collapse_diversity_threshold")) or 0.12
    major_improvement_min_delta = _parse_float(config.get("major_improvement_min_delta")) or 1e-9

    initial_best = _first_numeric(history_rows, "best_route_distance")
    initial_mean_fitness = _first_numeric(history_rows, "mean_fitness")
    initial_mean = -initial_mean_fitness if initial_mean_fitness is not None else None
    initial_diversity = _first_numeric(history_rows, "edge_diversity_ratio")
    final_diversity = _history_value(history_rows, "edge_diversity_ratio")
    initial_q_best = _first_numeric(quality_history_rows, "best_route_distance")
    quality_last_improvement = None
    q_final_generation = _final_generation(quality_history_rows)
    q_gens_since_last = _history_value(quality_history_rows, "generations_since_last_improvement")
    if q_final_generation is not None and q_gens_since_last is not None:
        quality_last_improvement = max(0, q_final_generation - int(q_gens_since_last))

    final_generation = _final_generation(history_rows)
    final_gens_since_last = _history_value(history_rows, "generations_since_last_improvement")
    last_improvement_generation = None
    if final_generation is not None and final_gens_since_last is not None:
        last_improvement_generation = max(0, final_generation - int(final_gens_since_last))

    first_major_improvement_generation = _first_generation(
        history_rows,
        lambda current: (_parse_float(current.get("improvement_delta")) or 0.0)
        > major_improvement_min_delta,
    )
    collapse_onset_generation = _first_generation(
        history_rows,
        lambda current: (
            (_parse_float(current.get("edge_diversity_ratio")) or 1.0) <= collapse_threshold
        ),
    )
    if collapse_onset_generation is None:
        collapse_onset_generation = _parse_int(row.get("collapse_onset_generation"))

    init_to_final_gain = _parse_float(row.get("init_to_final_gain"))
    quality_init_to_final_gain = _parse_float(quality_row.get("init_to_final_gain"))
    final_gap_vs_q = candidate_metric - quality_metric
    early_best_headstart_vs_q = (
        initial_best - initial_q_best
        if initial_best is not None and initial_q_best is not None
        else None
    )
    late_refinement_gap_vs_q = (
        quality_init_to_final_gain - init_to_final_gain
        if quality_init_to_final_gain is not None and init_to_final_gain is not None
        else None
    )
    last_improvement_gap_vs_q = (
        last_improvement_generation - quality_last_improvement
        if last_improvement_generation is not None and quality_last_improvement is not None
        else None
    )
    late_refinement_score = None
    if final_generation is not None:
        late_anchor_row = _closest_row(history_rows, max(0, int(final_generation * 0.7)))
        late_anchor_best = (
            _parse_float(late_anchor_row.get("best_route_distance")) if late_anchor_row is not None else None
        )
        total_gain = (
            initial_best - candidate_metric
            if initial_best is not None
            else None
        )
        late_gain = (
            late_anchor_best - candidate_metric
            if late_anchor_best is not None
            else None
        )
        if (
            total_gain is not None
            and total_gain > 1e-9
            and late_gain is not None
        ):
            late_refinement_score = max(0.0, min(1.0, late_gain / total_gain))
    collapse_to_last_improvement_gap = (
        last_improvement_generation - collapse_onset_generation
        if last_improvement_generation is not None and collapse_onset_generation is not None
        else None
    )
    edge_diversity_drop = (
        initial_diversity - final_diversity
        if initial_diversity is not None and final_diversity is not None
        else None
    )
    seed_lockin_signal = (
        early_best_headstart_vs_q is not None
        and early_best_headstart_vs_q <= 0.0
        and final_gap_vs_q > 0.0
        and collapse_onset_generation is not None
        and final_generation is not None
        and collapse_onset_generation <= max(1, int(final_generation * 0.55))
    )
    late_refinement_deficit_signal = (
        final_gap_vs_q > 0.0
        and late_refinement_gap_vs_q is not None
        and late_refinement_gap_vs_q > 0.0
        and last_improvement_gap_vs_q is not None
        and last_improvement_gap_vs_q < 0
    )
    if seed_lockin_signal and late_refinement_deficit_signal:
        mechanism_hint = "seed_lockin_plus_late_refinement_deficit"
    elif seed_lockin_signal:
        mechanism_hint = "seed_lockin_and_diversity_collapse"
    elif late_refinement_deficit_signal:
        mechanism_hint = "late_refinement_deficit"
    else:
        mechanism_hint = "mixed_or_seed_variance"

    return {
        "target_id": target_id,
        "problem": "tsp",
        "study_variant": _study_variant_key(row),
        "quality_variant": _study_variant_key(quality_row),
        "case_id": row.get("case_id") or quality_row.get("case_id") or "",
        "case_group": row.get("case_group") or quality_row.get("case_group") or "overall",
        "case_note": row.get("case_note") or quality_row.get("case_note") or "",
        "seed": _parse_int(row.get("seed")),
        "configured_budget": _parse_float(row.get("configured_budget")),
        "actual_evaluations_used": _parse_float(row.get("actual_evaluations_used")),
        "runtime_seconds": _parse_float(row.get("runtime_seconds")),
        "best_route_distance": candidate_metric,
        "quality_best_route_distance": quality_metric,
        "route_distance_loss_pct_vs_quality": (candidate_metric - quality_metric) / quality_metric * 100.0,
        "route_distance_delta_vs_quality": final_gap_vs_q,
        "initial_best_route_distance": initial_best,
        "initial_mean_route_distance": initial_mean,
        "initial_edge_diversity_ratio": initial_diversity,
        "final_edge_diversity_ratio": final_diversity,
        "edge_diversity_drop": edge_diversity_drop,
        "init_to_final_gain": init_to_final_gain,
        "quality_init_to_final_gain": quality_init_to_final_gain,
        "first_major_improvement_generation": first_major_improvement_generation,
        "last_improvement_generation": last_improvement_generation,
        "quality_last_improvement_generation": quality_last_improvement,
        "collapse_onset_generation": collapse_onset_generation,
        "generations_since_last_improvement": _history_value(history_rows, "generations_since_last_improvement"),
        "recent_window_improvement": _history_value(history_rows, "recent_window_improvement"),
        "recent_window_slope": _history_value(history_rows, "recent_window_slope"),
        "initial_best_headstart_vs_q": early_best_headstart_vs_q,
        "late_refinement_gap_vs_q": late_refinement_gap_vs_q,
        "late_refinement_score": late_refinement_score,
        "last_improvement_gap_vs_q": last_improvement_gap_vs_q,
        "collapse_to_last_improvement_gap": collapse_to_last_improvement_gap,
        "seed_lockin_signal": 1.0 if seed_lockin_signal else 0.0,
        "late_refinement_deficit_signal": 1.0 if late_refinement_deficit_signal else 0.0,
        "mechanism_hint": mechanism_hint,
        "history_path": row.get("history_path"),
        "quality_history_path": quality_row.get("history_path"),
    }


def _zdt1_trace_row(
    row: Mapping[str, Any],
    quality_row: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    quality_history_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    target_id: str,
    secondary_target_id: str | None,
) -> dict[str, Any] | None:
    quality_hv = _parse_float(quality_row.get("hypervolume"))
    candidate_hv = _parse_float(row.get("hypervolume"))
    if quality_hv is None or candidate_hv is None or quality_hv <= 0.0:
        return None

    pareto_drop_threshold = _parse_float(config.get("pareto_ratio_drop_threshold")) or 0.01
    spread_threshold = _parse_float(config.get("spread_degradation_threshold")) or 0.05
    hv_plateau_threshold = _parse_float(config.get("hv_plateau_slope_threshold")) or 0.005

    final_generation = _final_generation(history_rows)
    final_hv_slope = _history_value(history_rows, "recent_window_slope")
    final_spread = _history_value(history_rows, "spread")
    final_pareto_ratio = _history_value(history_rows, "pareto_ratio")
    final_front_size = _history_value(history_rows, "pareto_front_size")
    if final_front_size is None:
        final_front_size = _history_value(history_rows, "front_size")

    quality_final_spread = _parse_float(quality_row.get("spread"))
    quality_final_pareto_ratio = _parse_float(quality_row.get("pareto_ratio"))
    spread_delta_vs_quality = (
        final_spread - quality_final_spread
        if final_spread is not None and quality_final_spread is not None
        else None
    )
    pareto_delta_vs_quality = (
        final_pareto_ratio - quality_final_pareto_ratio
        if final_pareto_ratio is not None and quality_final_pareto_ratio is not None
        else None
    )
    spread_fail = spread_delta_vs_quality is not None and spread_delta_vs_quality > spread_threshold
    pareto_fail = pareto_delta_vs_quality is not None and pareto_delta_vs_quality < -pareto_drop_threshold

    hv_plateau_generation = _first_generation(
        history_rows,
        lambda current: abs(_parse_float(current.get("recent_window_slope")) or 1.0) <= hv_plateau_threshold,
    )
    trigger_fire_count = _parse_float(row.get("trigger_fire_count"))
    first_trigger_generation = _parse_float(row.get("first_trigger_generation"))
    if trigger_fire_count is None:
        trigger_fire_count = _history_value(history_rows, "adaptive_event_count", 0.0)
    if first_trigger_generation is None:
        first_trigger_generation = _first_generation(
            history_rows,
            lambda current: str(current.get("adaptive_event") or "").strip().lower() not in {"", "none"},
        )

    early_front_plateau_signal = (
        final_generation is not None
        and hv_plateau_generation is not None
        and hv_plateau_generation <= max(2, int(final_generation * 0.4))
    )

    safety_fail_onset_generation = None
    for current in history_rows:
        generation = _parse_int(current.get("generation"))
        if generation is None:
            continue
        quality_current = _closest_row(quality_history_rows, generation)
        if quality_current is None:
            continue
        candidate_spread = _parse_float(current.get("spread"))
        quality_spread = _parse_float(quality_current.get("spread"))
        candidate_pareto = _parse_float(current.get("pareto_ratio"))
        quality_pareto = _parse_float(quality_current.get("pareto_ratio"))
        current_spread_delta = (
            candidate_spread - quality_spread
            if candidate_spread is not None and quality_spread is not None
            else None
        )
        current_pareto_delta = (
            candidate_pareto - quality_pareto
            if candidate_pareto is not None and quality_pareto is not None
            else None
        )
        if (
            current_spread_delta is not None
            and current_spread_delta > spread_threshold
        ) or (
            current_pareto_delta is not None
            and current_pareto_delta < -pareto_drop_threshold
        ):
            safety_fail_onset_generation = generation
            break

    late_spread_collapse_signal = (
        safety_fail_onset_generation is not None
        and final_generation is not None
        and safety_fail_onset_generation >= int(final_generation * 0.5)
        and spread_fail
    )
    refresh_timing_mismatch_signal = (
        (spread_fail or pareto_fail)
        and (
            first_trigger_generation is None
            or safety_fail_onset_generation is None
            or first_trigger_generation > safety_fail_onset_generation
            or (trigger_fire_count or 0.0) <= 1.0
        )
    )

    if refresh_timing_mismatch_signal:
        mechanism_hint = "refresh_timing_mismatch"
    elif late_spread_collapse_signal:
        mechanism_hint = "late_spread_collapse"
    elif early_front_plateau_signal:
        mechanism_hint = "early_front_plateau"
    else:
        mechanism_hint = "mixed_or_seed_variance"

    target_for_row = secondary_target_id if pareto_fail or (spread_fail and pareto_fail) else target_id
    mid_generation = max(0, int(final_generation * 0.5)) if final_generation is not None else None
    mid_row = _closest_row(history_rows, mid_generation) if mid_generation is not None else None
    initial_front_size = _first_numeric(history_rows, "pareto_front_size")
    if initial_front_size is None:
        initial_front_size = _first_numeric(history_rows, "front_size")
    mid_front_size = _parse_float(mid_row.get("pareto_front_size")) if mid_row else None
    if mid_front_size is None and mid_row is not None:
        mid_front_size = _parse_float(mid_row.get("front_size"))
    early_front_growth = (
        mid_front_size - initial_front_size
        if mid_front_size is not None and initial_front_size is not None
        else None
    )
    late_front_growth = (
        final_front_size - mid_front_size
        if final_front_size is not None and mid_front_size is not None
        else None
    )

    return {
        "target_id": target_for_row,
        "problem": "zdt1",
        "study_variant": _study_variant_key(row),
        "quality_variant": _study_variant_key(quality_row),
        "seed": _parse_int(row.get("seed")),
        "configured_budget": _parse_float(row.get("configured_budget")),
        "actual_evaluations_used": _parse_float(row.get("actual_evaluations_used")),
        "runtime_seconds": _parse_float(row.get("runtime_seconds")),
        "hypervolume": candidate_hv,
        "quality_hypervolume": quality_hv,
        "hv_loss_pct_vs_quality": (quality_hv - candidate_hv) / quality_hv * 100.0,
        "pareto_ratio_delta_vs_quality": pareto_delta_vs_quality,
        "spread_delta_vs_quality": spread_delta_vs_quality,
        "pareto_ratio_safety_fail": 1.0 if pareto_fail else 0.0,
        "spread_safety_fail": 1.0 if spread_fail else 0.0,
        "joint_safety_fail": 1.0 if pareto_fail or spread_fail else 0.0,
        "hv_plateau_generation": hv_plateau_generation,
        "safety_fail_onset_generation": safety_fail_onset_generation,
        "final_front_size": final_front_size,
        "front_size_growth_early": early_front_growth,
        "front_size_growth_late": late_front_growth,
        "recent_window_slope": final_hv_slope,
        "trigger_fire_count": trigger_fire_count,
        "first_trigger_generation": first_trigger_generation,
        "realized_refresh_volume": _parse_float(row.get("realized_refresh_volume")),
        "total_refresh_fraction_realized": _parse_float(row.get("total_refresh_fraction_realized")),
        "early_front_plateau_signal": 1.0 if early_front_plateau_signal else 0.0,
        "late_spread_collapse_signal": 1.0 if late_spread_collapse_signal else 0.0,
        "refresh_timing_mismatch_signal": 1.0 if refresh_timing_mismatch_signal else 0.0,
        "mechanism_hint": mechanism_hint,
        "history_path": row.get("history_path"),
        "quality_history_path": quality_row.get("history_path"),
    }


def build_failure_trace_rows(
    problem: str,
    config: Mapping[str, Any] | None,
    raw_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(config, Mapping):
        return []

    quality_variant = str(config.get("quality_variant") or "").strip()
    target_id = str(config.get("target_id") or "").strip()
    if not quality_variant or not target_id:
        return []

    fast_variant = str(config.get("fast_variant") or "").strip()
    reference_variants = [
        str(value).strip()
        for value in config.get("reference_variants", [])
        if isinstance(value, str) and str(value).strip()
    ]
    candidate_variants = {value for value in [fast_variant, *reference_variants] if value}
    secondary_target_id = str(config.get("secondary_target_id") or "").strip() or None

    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in raw_rows:
        variant = _study_variant_key(row)
        if not variant:
            continue
        pair_key = _pair_key(problem, row)
        if pair_key is None:
            continue
        grouped.setdefault(pair_key, {})[variant] = row

    trace_rows: list[dict[str, Any]] = []
    for variants in grouped.values():
        quality_row = variants.get(quality_variant)
        if quality_row is None:
            continue
        quality_history = _history_rows_from_path(quality_row.get("history_path"))
        for variant in candidate_variants:
            row = variants.get(variant)
            if row is None:
                continue
            history = _history_rows_from_path(row.get("history_path"))
            trace_row = None
            if problem == "tsp":
                trace_row = _tsp_trace_row(row, quality_row, history, quality_history, config, target_id)
            elif problem == "zdt1":
                trace_row = _zdt1_trace_row(
                    row,
                    quality_row,
                    history,
                    quality_history,
                    config,
                    target_id,
                    secondary_target_id,
                )
            if trace_row is not None:
                trace_rows.append(trace_row)

    trace_rows.sort(
        key=lambda item: (
            str(item.get("target_id") or ""),
            str(item.get("case_group") or ""),
            -abs(
                _parse_float(item.get("route_distance_loss_pct_vs_quality"))
                or _parse_float(item.get("hv_loss_pct_vs_quality"))
                or 0.0
            ),
            str(item.get("study_variant") or ""),
            _parse_int(item.get("seed")) or 0,
        )
    )
    return trace_rows


def _hypothesis_row(
    *,
    hypothesis_id: str,
    target_id: str,
    problem: str,
    affected_default: str,
    evidence_rows: int,
    suspected_mechanism: str,
    supporting_trace_signals: str,
    confidence: str,
    current_evidence_strength: str | None = None,
    probe_attempted: str,
    confirmed_or_weakened: str,
    recommended_next_probe: str,
    keep_as_regression_case: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "target_id": target_id,
        "problem": problem,
        "affected_default": affected_default,
        "evidence_rows": evidence_rows,
        "suspected_mechanism": suspected_mechanism,
        "supporting_trace_signals": supporting_trace_signals,
        "confidence": confidence,
        "current_evidence_strength": current_evidence_strength or confidence,
        "probe_attempted": probe_attempted,
        "confirmed_or_weakened": confirmed_or_weakened,
        "recommended_next_probe": recommended_next_probe,
        "recommended_next_action": recommended_next_probe,
        "keep_as_regression_case": keep_as_regression_case,
        "notes": notes,
    }


def _tsp_probe_hypothesis_rows(
    config: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    probe_config = config.get("hypothesis_probe")
    if not isinstance(probe_config, Mapping):
        return []

    target_id = str(config.get("target_id") or "").strip()
    baseline_variant = str(probe_config.get("baseline_variant") or config.get("fast_variant") or "").strip()
    late_probe_variant = str(probe_config.get("late_refinement_probe_variant") or "").strip()
    population_probe_variant = str(probe_config.get("population_probe_variant") or "").strip()
    seed_probe_variant = str(probe_config.get("seed_fraction_probe_variant") or "").strip()
    contour_probe_variants = [
        str(value).strip()
        for value in probe_config.get("contour_probe_variants", [])
        if isinstance(value, str) and str(value).strip()
    ]
    tail_focus = str(probe_config.get("tail_focus") or "p90_p95").strip().lower()
    extreme_tail_focus = tail_focus in {"extreme_tail", "p95_max", "p95max"}
    focus_case_group = str(probe_config.get("focus_case_group") or "anti_case").strip()
    preserve_case_group = str(probe_config.get("preserve_case_group") or "rescue_target").strip()
    hypothesis_ids = [
        str(value).strip()
        for value in config.get("hypothesis_ids", [])
        if isinstance(value, str) and str(value).strip()
    ]
    late_hypothesis_id = (
        hypothesis_ids[0] if len(hypothesis_ids) > 0 else "tsp_anticase_late_refinement_deficit"
    )
    seed_hypothesis_id = (
        hypothesis_ids[1] if len(hypothesis_ids) > 1 else "tsp_anticase_seed_lockin_secondary"
    )
    for variant in (late_probe_variant, population_probe_variant):
        if variant and variant not in contour_probe_variants:
            contour_probe_variants.append(variant)

    baseline_rows = _variant_rows(trace_rows, baseline_variant, target_id=target_id)
    baseline_focus_rows = _variant_rows(
        trace_rows,
        baseline_variant,
        target_id=target_id,
        case_group=focus_case_group,
    )
    if not baseline_rows or not baseline_focus_rows:
        return []
    baseline_rescue_rows = _variant_rows(
        trace_rows,
        baseline_variant,
        target_id=target_id,
        case_group=preserve_case_group,
    )
    late_probe_focus_rows = []
    late_probe_rescue_rows = []
    if late_probe_variant:
        late_probe_focus_rows = _variant_rows(
            trace_rows,
            late_probe_variant,
            target_id=target_id,
            case_group=focus_case_group,
        )
        late_probe_rescue_rows = _variant_rows(
            trace_rows,
            late_probe_variant,
            target_id=target_id,
            case_group=preserve_case_group,
        )
    population_probe_focus_rows = []
    if population_probe_variant:
        population_probe_focus_rows = _variant_rows(
            trace_rows,
            population_probe_variant,
            target_id=target_id,
            case_group=focus_case_group,
        )
    seed_probe_focus_rows = (
        _variant_rows(trace_rows, seed_probe_variant, target_id=target_id, case_group=focus_case_group)
        if seed_probe_variant
        else []
    )
    seed_probe_rescue_rows = (
        _variant_rows(trace_rows, seed_probe_variant, target_id=target_id, case_group=preserve_case_group)
        if seed_probe_variant
        else []
    )

    baseline_focus_losses = _numeric_values(baseline_focus_rows, "route_distance_loss_pct_vs_quality")
    late_probe_losses = _numeric_values(late_probe_focus_rows, "route_distance_loss_pct_vs_quality")
    population_probe_losses = _numeric_values(
        population_probe_focus_rows, "route_distance_loss_pct_vs_quality"
    )
    seed_probe_losses = _numeric_values(seed_probe_focus_rows, "route_distance_loss_pct_vs_quality")

    baseline_focus_p90 = _quantile(baseline_focus_losses, 0.9)
    baseline_focus_p95 = _quantile(baseline_focus_losses, 0.95)
    baseline_focus_max = max(baseline_focus_losses) if baseline_focus_losses else None
    late_probe_p90 = _quantile(late_probe_losses, 0.9)
    late_probe_p95 = _quantile(late_probe_losses, 0.95)
    late_probe_max = max(late_probe_losses) if late_probe_losses else None
    population_probe_p90 = _quantile(population_probe_losses, 0.9)
    population_probe_p95 = _quantile(population_probe_losses, 0.95)
    seed_probe_p90 = _quantile(seed_probe_losses, 0.9)
    seed_probe_p95 = _quantile(seed_probe_losses, 0.95)

    baseline_rescue_mean = _mean_or_none(
        _numeric_values(baseline_rescue_rows, "route_distance_loss_pct_vs_quality")
    )
    late_probe_rescue_mean = _mean_or_none(
        _numeric_values(late_probe_rescue_rows, "route_distance_loss_pct_vs_quality")
    )
    late_probe_late_score = _mean_or_none(
        _numeric_values(late_probe_focus_rows, "late_refinement_score")
    )
    late_probe_collapse_gap = _mean_or_none(
        _numeric_values(late_probe_focus_rows, "collapse_to_last_improvement_gap")
    )
    seed_probe_rescue_mean = _mean_or_none(
        _numeric_values(seed_probe_rescue_rows, "route_distance_loss_pct_vs_quality")
    )

    contour_metrics: dict[str, dict[str, float | None]] = {}
    for variant in contour_probe_variants:
        focus_rows = _variant_rows(
            trace_rows,
            variant,
            target_id=target_id,
            case_group=focus_case_group,
        )
        if not focus_rows:
            continue
        rescue_rows = _variant_rows(
            trace_rows,
            variant,
            target_id=target_id,
            case_group=preserve_case_group,
        )
        focus_losses = _numeric_values(focus_rows, "route_distance_loss_pct_vs_quality")
        rescue_losses = _numeric_values(rescue_rows, "route_distance_loss_pct_vs_quality")
        contour_metrics[variant] = {
            "anti_mean": _mean_or_none(focus_losses),
            "anti_p90": _quantile(focus_losses, 0.9),
            "anti_p95": _quantile(focus_losses, 0.95),
            "anti_max": max(focus_losses) if focus_losses else None,
            "rescue_mean": _mean_or_none(rescue_losses),
            "late_score": _mean_or_none(_numeric_values(focus_rows, "late_refinement_score")),
            "collapse_gap": _mean_or_none(
                _numeric_values(focus_rows, "collapse_to_last_improvement_gap")
            ),
        }

    best_contour_variant = ""
    best_contour_metrics: dict[str, float | None] | None = None
    def _metric_or_default(metrics: Mapping[str, float | None], key: str) -> float:
        value = metrics.get(key)
        return float(value) if value is not None else 99.0
    if contour_metrics:
        best_contour_variant, best_contour_metrics = sorted(
            contour_metrics.items(),
            key=lambda item: (
                _metric_or_default(item[1], "anti_p95" if extreme_tail_focus else "anti_p90"),
                _metric_or_default(item[1], "anti_max" if extreme_tail_focus else "anti_p95"),
                _metric_or_default(item[1], "anti_p90"),
                _metric_or_default(item[1], "anti_max"),
                _metric_or_default(item[1], "rescue_mean"),
            ),
        )[0]
    best_contour_p90 = best_contour_metrics.get("anti_p90") if best_contour_metrics else late_probe_p90
    best_contour_p95 = best_contour_metrics.get("anti_p95") if best_contour_metrics else late_probe_p95
    best_contour_max = best_contour_metrics.get("anti_max") if best_contour_metrics else late_probe_max
    best_contour_rescue_mean = (
        best_contour_metrics.get("rescue_mean") if best_contour_metrics else late_probe_rescue_mean
    )
    best_contour_late_score = (
        best_contour_metrics.get("late_score") if best_contour_metrics else late_probe_late_score
    )
    best_contour_collapse_gap = (
        best_contour_metrics.get("collapse_gap") if best_contour_metrics else late_probe_collapse_gap
    )

    late_signal_rate = _signal_rate(baseline_focus_rows, "late_refinement_deficit_signal")
    seed_signal_rate = _signal_rate(baseline_focus_rows, "seed_lockin_signal")
    late_confidence = _confidence_from_rate(late_signal_rate)
    seed_confidence = _confidence_from_rate(seed_signal_rate)

    baseline_late_score = _mean_or_none(_numeric_values(baseline_focus_rows, "late_refinement_score"))
    baseline_collapse_gap = _mean_or_none(
        _numeric_values(baseline_focus_rows, "collapse_to_last_improvement_gap")
    )

    rescue_delta_late = (
        best_contour_rescue_mean - baseline_rescue_mean
        if best_contour_rescue_mean is not None and baseline_rescue_mean is not None
        else None
    )
    late_score_delta = (
        best_contour_late_score - baseline_late_score
        if best_contour_late_score is not None and baseline_late_score is not None
        else None
    )
    collapse_gap_delta = (
        best_contour_collapse_gap - baseline_collapse_gap
        if best_contour_collapse_gap is not None and baseline_collapse_gap is not None
        else None
    )

    late_p90_improved = (
        best_contour_p90 is not None
        and baseline_focus_p90 is not None
        and best_contour_p90 < baseline_focus_p90 - 0.05
    )
    late_p95_improved = (
        best_contour_p95 is not None
        and baseline_focus_p95 is not None
        and best_contour_p95 < baseline_focus_p95 - 0.05
    )
    late_max_improved = (
        best_contour_max is not None
        and baseline_focus_max is not None
        and best_contour_max < baseline_focus_max - 0.05
    )
    late_tail_improved = late_p90_improved or late_p95_improved or late_max_improved
    rescue_preserved = rescue_delta_late is None or rescue_delta_late <= 0.25
    best_contour_anchor = best_contour_p95 if extreme_tail_focus else best_contour_p90
    population_probe_anchor = population_probe_p95 if extreme_tail_focus else population_probe_p90
    population_direction_supports = (
        population_probe_anchor is None
        or best_contour_anchor is None
        or best_contour_anchor <= population_probe_anchor + 0.01
    )
    late_core_improved = (
        late_p95_improved and late_max_improved if extreme_tail_focus else late_p90_improved and late_p95_improved
    )
    late_partial_improved = (
        late_p95_improved or late_max_improved if extreme_tail_focus else late_p90_improved
    )
    if (
        late_core_improved
        and rescue_preserved
        and population_direction_supports
    ):
        late_status = "strengthened"
    elif (late_partial_improved and rescue_preserved) or (
        late_score_delta is not None and late_score_delta > 0.0 and rescue_preserved
    ):
        late_status = "partially_supported"
    else:
        late_status = "weakened"

    seed_rescue_delta = (
        seed_probe_rescue_mean - baseline_rescue_mean
        if seed_probe_rescue_mean is not None and baseline_rescue_mean is not None
        else None
    )
    seed_probe_anchor = seed_probe_p95 if extreme_tail_focus else seed_probe_p90
    baseline_seed_anchor = baseline_focus_p95 if extreme_tail_focus else baseline_focus_p90
    best_contour_seed_anchor = best_contour_p95 if extreme_tail_focus else best_contour_p90
    seed_tail_improved = (
        seed_probe_anchor is not None
        and baseline_seed_anchor is not None
        and seed_probe_anchor < baseline_seed_anchor - 0.05
    )
    if (
        seed_tail_improved
        and seed_probe_anchor is not None
        and best_contour_seed_anchor is not None
        and seed_probe_anchor < best_contour_seed_anchor - 0.05
        and (seed_rescue_delta is None or seed_rescue_delta <= 0.25)
    ):
        seed_status = "strengthened"
    elif seed_tail_improved and (seed_rescue_delta is None or seed_rescue_delta <= 0.25):
        seed_status = "partially_supported"
    else:
        seed_status = "secondary_only"

    late_signals = "; ".join(
        [
            f"baseline_late_signal_rate={late_signal_rate:.2f}",
            f"baseline_anti_p90={_format_metric(baseline_focus_p90)}",
            f"baseline_anti_p95={_format_metric(baseline_focus_p95)}",
            f"baseline_anti_max={_format_metric(baseline_focus_max)}",
            f"best_contour_probe={best_contour_variant or late_probe_variant or baseline_variant}",
            f"best_contour_anti_p90={_format_metric(best_contour_p90)}",
            f"best_contour_anti_p95={_format_metric(best_contour_p95)}",
            f"best_contour_anti_max={_format_metric(best_contour_max)}",
            f"population_probe_anti_p90={_format_metric(population_probe_p90)}",
            f"population_probe_anti_p95={_format_metric(population_probe_p95)}",
            f"rescue_mean_delta={_format_metric(rescue_delta_late)}",
            f"late_refinement_score_delta={_format_metric(late_score_delta)}",
            f"collapse_gap_delta={_format_metric(collapse_gap_delta)}",
        ]
    )
    seed_signals = "; ".join(
        [
            f"baseline_seed_lockin_rate={seed_signal_rate:.2f}",
            f"baseline_anti_p90={_format_metric(baseline_focus_p90)}",
            f"baseline_anti_p95={_format_metric(baseline_focus_p95)}",
            f"seed_probe_anti_p90={_format_metric(seed_probe_p90)}",
            f"seed_probe_anti_p95={_format_metric(seed_probe_p95)}",
            f"best_contour_anti_p90={_format_metric(best_contour_p90)}",
            f"best_contour_anti_p95={_format_metric(best_contour_p95)}",
            f"seed_probe_rescue_delta={_format_metric(seed_rescue_delta)}",
        ]
    )

    late_notes = (
        "Same-budget contour probe reduced anti-case tail without reopening rescue rows."
        if late_status == "strengthened"
        else "Contour probe changed the anti-case tail, but not enough to justify a default rewrite."
        if late_status == "partially_supported"
        else "Same-budget contour probe did not close the anti-case tail cleanly on the current stress slice."
    )
    seed_notes = (
        "Seed-fraction-only probe beat the baseline, but it still needs to prove it is not just a secondary effect."
        if seed_status == "strengthened"
        else "Seed-fraction-only probe moved some tail rows, but it did not clearly outrank the late-refinement probe."
        if seed_status == "partially_supported"
        else "Seed lock-in still looks secondary: the seed-fraction-only probe did not beat the late-refinement probe cleanly."
    )

    return [
        _hypothesis_row(
            hypothesis_id=late_hypothesis_id,
            target_id=target_id,
            problem="tsp",
            affected_default="configs/local_profiles/tsp_seeded_swap_local_fast.json",
            evidence_rows=sum(
                1
                for row in baseline_focus_rows
                if (_parse_float(row.get("late_refinement_deficit_signal")) or 0.0) > 0.0
            ),
            suspected_mechanism="late_refinement_deficit",
            supporting_trace_signals=late_signals,
            confidence=late_confidence,
            probe_attempted=",".join(
                value
                for value in contour_probe_variants
                if value
            ),
            confirmed_or_weakened=late_status,
            recommended_next_probe="investigate_population_generation_tradeoff",
            keep_as_regression_case=True,
            notes=late_notes,
        ),
        _hypothesis_row(
            hypothesis_id=seed_hypothesis_id,
            target_id=target_id,
            problem="tsp",
            affected_default="configs/local_profiles/tsp_seeded_swap_local_fast.json",
            evidence_rows=sum(
                1
                for row in baseline_focus_rows
                if (_parse_float(row.get("seed_lockin_signal")) or 0.0) > 0.0
            ),
            suspected_mechanism="seed_lockin_and_diversity_collapse",
            supporting_trace_signals=seed_signals,
            confidence=seed_confidence,
            probe_attempted=seed_probe_variant or "none",
            confirmed_or_weakened=seed_status,
            recommended_next_probe="investigate_seed_fraction_secondary_only",
            keep_as_regression_case=True,
            notes=seed_notes,
        ),
    ]


def _zdt1_probe_hypothesis_rows(
    config: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    probe_config = config.get("hypothesis_probe")
    if not isinstance(probe_config, Mapping):
        return []

    spread_target_id = str(config.get("target_id") or "").strip()
    joint_target_id = str(config.get("secondary_target_id") or spread_target_id).strip()
    baseline_variant = str(probe_config.get("baseline_variant") or config.get("fast_variant") or "").strip()
    probe_mode = str(probe_config.get("probe_mode") or "combined").strip().lower()
    refresh_probe_variant = str(probe_config.get("refresh_probe_variant") or "").strip()
    cooldown_probe_variant = str(probe_config.get("cooldown_probe_variant") or "").strip()
    population_probe_variant = str(probe_config.get("population_probe_variant") or "").strip()
    population_generation_probe_variants = [
        str(value).strip()
        for value in probe_config.get("population_generation_probe_variants", [])
        if isinstance(value, str) and str(value).strip()
    ]
    if population_probe_variant and population_probe_variant not in population_generation_probe_variants:
        population_generation_probe_variants.append(population_probe_variant)
    hypothesis_ids = [
        str(value).strip()
        for value in config.get("hypothesis_ids", [])
        if isinstance(value, str) and str(value).strip()
    ]
    timing_hypothesis_id = (
        hypothesis_ids[0]
        if len(hypothesis_ids) > 0
        else "zdt1_fast_refresh_timing_mismatch_joint"
    )
    spread_candidate_hypothesis_id = (
        hypothesis_ids[1]
        if len(hypothesis_ids) > 1
        else "zdt1_fast_spread_mechanism_population_generation_candidate"
    )
    spread_unisolated_hypothesis_id = (
        hypothesis_ids[2]
        if len(hypothesis_ids) > 2
        else "zdt1_fast_spread_mechanism_still_unisolated"
    )
    joint_unisolated_hypothesis_id = str(
        probe_config.get("joint_unisolated_hypothesis_id")
        or "zdt1_fast_joint_mechanism_still_unisolated"
    ).strip()

    def _zdt1_metrics(variant: str) -> dict[str, float | None]:
        variant_rows = _variant_rows(trace_rows, variant)
        hv_losses = _numeric_values(variant_rows, "hv_loss_pct_vs_quality")
        return {
            "count": float(len(variant_rows)),
            "hv_mean": _mean_or_none(hv_losses),
            "hv_p75": _quantile(hv_losses, 0.75),
            "hv_p90": _quantile(hv_losses, 0.9),
            "hv_max": max(hv_losses) if hv_losses else None,
            "spread_fail_rate": _mean_or_none(_numeric_values(variant_rows, "spread_safety_fail")),
            "joint_fail_rate": _mean_or_none(_numeric_values(variant_rows, "joint_safety_fail")),
            "timing_signal_rate": _signal_rate(variant_rows, "refresh_timing_mismatch_signal"),
        }

    baseline_metrics = _zdt1_metrics(baseline_variant)
    refresh_metrics = _zdt1_metrics(refresh_probe_variant) if refresh_probe_variant else {}
    cooldown_metrics = _zdt1_metrics(cooldown_probe_variant) if cooldown_probe_variant else {}
    population_metrics_map = {
        variant: _zdt1_metrics(variant) for variant in population_generation_probe_variants
    }

    timing_candidate_rows = [
        (refresh_probe_variant, refresh_metrics),
        (cooldown_probe_variant, cooldown_metrics),
    ]
    timing_candidate_rows = [
        (variant, metrics)
        for variant, metrics in timing_candidate_rows
        if variant and metrics.get("count")
    ]
    population_candidate_rows = [
        (variant, metrics)
        for variant, metrics in population_metrics_map.items()
        if variant and metrics.get("count")
    ]

    best_timing_variant = baseline_variant
    best_timing_metrics = baseline_metrics
    if timing_candidate_rows:
        best_timing_variant, best_timing_metrics = sorted(
            timing_candidate_rows,
            key=lambda item: (
                item[1].get("joint_fail_rate") if item[1].get("joint_fail_rate") is not None else 99.0,
                item[1].get("spread_fail_rate") if item[1].get("spread_fail_rate") is not None else 99.0,
                item[1].get("hv_p90") if item[1].get("hv_p90") is not None else 99.0,
            ),
        )[0]

    best_spread_variant = baseline_variant
    best_spread_metrics = baseline_metrics
    if population_candidate_rows:
        best_spread_variant, best_spread_metrics = sorted(
            population_candidate_rows,
            key=lambda item: (
                item[1].get("spread_fail_rate") if item[1].get("spread_fail_rate") is not None else 99.0,
                item[1].get("joint_fail_rate") if item[1].get("joint_fail_rate") is not None else 99.0,
                item[1].get("hv_p90") if item[1].get("hv_p90") is not None else 99.0,
            ),
        )[0]

    baseline_joint_rate = baseline_metrics.get("joint_fail_rate")
    baseline_spread_rate = baseline_metrics.get("spread_fail_rate")
    best_joint_rate = best_timing_metrics.get("joint_fail_rate")
    best_spread_rate = best_timing_metrics.get("spread_fail_rate")
    best_pg_spread_rate = best_spread_metrics.get("spread_fail_rate")
    best_pg_joint_rate = best_spread_metrics.get("joint_fail_rate")
    baseline_hv_p90 = baseline_metrics.get("hv_p90")
    best_hv_p90 = best_timing_metrics.get("hv_p90")
    best_pg_hv_p90 = best_spread_metrics.get("hv_p90")
    joint_reduction = (
        baseline_joint_rate - best_joint_rate
        if baseline_joint_rate is not None and best_joint_rate is not None
        else None
    )
    timing_spread_reduction = (
        baseline_spread_rate - best_spread_rate
        if baseline_spread_rate is not None and best_spread_rate is not None
        else None
    )
    pg_spread_reduction = (
        baseline_spread_rate - best_pg_spread_rate
        if baseline_spread_rate is not None and best_pg_spread_rate is not None
        else None
    )
    hv_preservation_delta = (
        best_hv_p90 - baseline_hv_p90
        if best_hv_p90 is not None and baseline_hv_p90 is not None
        else None
    )
    pg_hv_preservation_delta = (
        best_pg_hv_p90 - baseline_hv_p90
        if best_pg_hv_p90 is not None and baseline_hv_p90 is not None
        else None
    )
    timing_signal_rate = baseline_metrics.get("timing_signal_rate") or 0.0
    timing_confidence = _confidence_from_rate(float(timing_signal_rate))
    spread_confidence = _confidence_from_rate(float(baseline_spread_rate or 0.0))

    if (
        joint_reduction is not None
        and joint_reduction >= 0.1
        and (hv_preservation_delta is None or hv_preservation_delta <= 0.1)
    ):
        timing_status = "strengthened"
    elif (
        joint_reduction is not None
        and joint_reduction > 0.0
        and (hv_preservation_delta is None or hv_preservation_delta <= 0.15)
    ):
        timing_status = "partially_supported"
    else:
        timing_status = "weakened"

    spread_candidate_supported = (
        pg_spread_reduction is not None
        and pg_spread_reduction > 0.0
        and (
            timing_spread_reduction is None
            or pg_spread_reduction >= timing_spread_reduction + 0.02
            or best_spread_variant != baseline_variant
        )
        and (pg_hv_preservation_delta is None or pg_hv_preservation_delta <= 0.15)
    )
    if (
        spread_candidate_supported
        and pg_spread_reduction is not None
        and pg_spread_reduction >= 0.1
        and (pg_hv_preservation_delta is None or pg_hv_preservation_delta <= 0.1)
    ):
        spread_status = "strengthened"
    elif spread_candidate_supported:
        spread_status = "partially_supported"
    else:
        spread_status = "still_unisolated"

    timing_signals = "; ".join(
        [
            f"baseline_timing_signal_rate={float(timing_signal_rate):.2f}",
            f"baseline_joint_fail_rate={_format_metric(baseline_joint_rate)}",
            f"best_probe={best_timing_variant or baseline_variant}",
            f"best_probe_joint_fail_rate={_format_metric(best_joint_rate)}",
            f"best_probe_spread_fail_rate={_format_metric(best_spread_rate)}",
            f"hv_p90_delta={_format_metric(hv_preservation_delta)}",
        ]
    )
    spread_signals = "; ".join(
        [
            f"baseline_spread_fail_rate={_format_metric(baseline_spread_rate)}",
            f"refresh_probe_spread_fail_rate={_format_metric(refresh_metrics.get('spread_fail_rate'))}",
            f"cooldown_probe_spread_fail_rate={_format_metric(cooldown_metrics.get('spread_fail_rate'))}",
            f"best_pg_probe={best_spread_variant or baseline_variant}",
            f"best_pg_spread_fail_rate={_format_metric(best_pg_spread_rate)}",
            f"best_pg_joint_fail_rate={_format_metric(best_pg_joint_rate)}",
            f"timing_spread_reduction={_format_metric(timing_spread_reduction)}",
            f"pg_spread_reduction={_format_metric(pg_spread_reduction)}",
            f"pg_hv_p90_delta={_format_metric(pg_hv_preservation_delta)}",
        ]
    )
    timing_notes = (
        "Timing-only probe lowered joint safety failures without giving back the HV tail."
        if timing_status == "strengthened"
        else "Timing probe moved the joint safety slice, but not enough to promote a fast-default rewrite."
        if timing_status == "partially_supported"
        else "Timing-only probes did not close the joint safety failures cleanly under the current fast budget."
    )
    spread_notes = (
        "Population/generation tradeoff now explains more spread-fail rows than timing-only probes under the same fast budget."
        if spread_status in {"strengthened", "partially_supported"}
        else "Spread failures still do not collapse to one dominant timing-only story; keep the mechanism split open."
    )

    baseline_variant_rows = _variant_rows(trace_rows, baseline_variant)
    timing_hypothesis_output_id = (
        timing_hypothesis_id if timing_status != "weakened" else joint_unisolated_hypothesis_id
    )
    timing_mechanism = (
        "refresh_timing_mismatch_joint"
        if timing_status != "weakened"
        else "joint_mechanism_still_unisolated"
    )
    spread_hypothesis_id = (
        spread_candidate_hypothesis_id
        if spread_status in {"strengthened", "partially_supported"}
        else spread_unisolated_hypothesis_id
    )
    spread_mechanism = (
        "spread_mechanism_population_generation_candidate"
        if spread_status in {"strengthened", "partially_supported"}
        else "spread_mechanism_still_unisolated"
    )
    rows: list[dict[str, Any]] = []
    if probe_mode in {"combined", "joint_only"}:
        rows.append(
            _hypothesis_row(
                hypothesis_id=timing_hypothesis_output_id,
                target_id=joint_target_id,
                problem="zdt1",
                affected_default="configs/local_profiles/zdt1_diversity_injection_fast.json",
                evidence_rows=sum(
                    1
                    for row in baseline_variant_rows
                    if (_parse_float(row.get("refresh_timing_mismatch_signal")) or 0.0) > 0.0
                ),
                suspected_mechanism=timing_mechanism,
                supporting_trace_signals=timing_signals,
                confidence=timing_confidence,
                probe_attempted=",".join(
                    value
                    for value in [
                        refresh_probe_variant,
                        cooldown_probe_variant,
                        *population_generation_probe_variants,
                    ]
                    if value
                ),
                confirmed_or_weakened=timing_status,
                recommended_next_probe=(
                    "investigate_refresh_vs_cooldown_timing_joint_only"
                    if timing_status != "weakened"
                    else "split_spread_vs_joint_fail_mechanisms"
                ),
                keep_as_regression_case=True,
                notes=timing_notes,
            )
        )
    if probe_mode in {"combined", "spread_only"}:
        rows.append(
            _hypothesis_row(
                hypothesis_id=spread_hypothesis_id,
                target_id=spread_target_id,
                problem="zdt1",
                affected_default="configs/local_profiles/zdt1_diversity_injection_fast.json",
                evidence_rows=sum(
                    1
                    for row in baseline_variant_rows
                    if (_parse_float(row.get("spread_safety_fail")) or 0.0) > 0.0
                ),
                suspected_mechanism=spread_mechanism,
                supporting_trace_signals=spread_signals,
                confidence=spread_confidence,
                probe_attempted=",".join(
                    value
                    for value in [
                        refresh_probe_variant,
                        cooldown_probe_variant,
                        *population_generation_probe_variants,
                    ]
                    if value
                ),
                confirmed_or_weakened=spread_status,
                recommended_next_probe=(
                    "investigate_population_generation_tradeoff"
                    if spread_status in {"strengthened", "partially_supported"}
                    else "split_spread_vs_joint_fail_mechanisms"
                ),
                keep_as_regression_case=True,
                notes=spread_notes,
            )
        )
    return rows


def build_failure_hypothesis_rows(
    problem: str,
    config: Mapping[str, Any] | None,
    trace_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(config, Mapping) or not trace_rows:
        return []

    target_id = str(config.get("target_id") or "").strip()
    fast_variant = str(config.get("fast_variant") or "").strip()
    hypothesis_ids = [
        str(value).strip()
        for value in config.get("hypothesis_ids", [])
        if isinstance(value, str) and str(value).strip()
    ]
    focus_rows = [row for row in trace_rows if str(row.get("study_variant") or "") == fast_variant] or list(trace_rows)

    rows: list[dict[str, Any]] = []
    probe_config = config.get("hypothesis_probe")
    if isinstance(probe_config, Mapping):
        if problem == "tsp":
            rows = _tsp_probe_hypothesis_rows(config, trace_rows)
        elif problem == "zdt1":
            rows = _zdt1_probe_hypothesis_rows(config, trace_rows)

    if problem == "tsp":
        if not rows:
            candidate_specs = [
                (
                    hypothesis_ids[0] if len(hypothesis_ids) > 0 else "tsp_anticase_seed_lockin",
                    "seed_lockin_and_diversity_collapse",
                    "seed_lockin_signal",
                    "investigate_seed_fraction_only",
                ),
                (
                    hypothesis_ids[1] if len(hypothesis_ids) > 1 else "tsp_anticase_late_refinement_deficit",
                    "late_refinement_deficit",
                    "late_refinement_deficit_signal",
                    "investigate_population_generation_tradeoff",
                ),
            ]
            for hypothesis_id, mechanism, signal_key, next_probe in candidate_specs:
                evidence = [row for row in focus_rows if float(row.get(signal_key, 0.0) or 0.0) > 0.0]
                rate = len(evidence) / len(focus_rows) if focus_rows else 0.0
                confidence = _confidence_from_rate(rate)
                rows.append(
                    _hypothesis_row(
                        hypothesis_id=hypothesis_id,
                        target_id=target_id,
                        problem="tsp",
                        affected_default="configs/local_profiles/tsp_seeded_swap_local_fast.json",
                        evidence_rows=len(evidence),
                        suspected_mechanism=mechanism,
                        supporting_trace_signals=(
                            f"{signal_key}_rate={rate:.2f}; mean_final_gap="
                            f"{mean([_parse_float(row.get('route_distance_delta_vs_quality')) or 0.0 for row in evidence]):.4f}"
                            if evidence
                            else f"{signal_key}_rate=0.00"
                        ),
                        confidence=confidence,
                        current_evidence_strength=confidence,
                        probe_attempted="none",
                        confirmed_or_weakened="not_tested",
                        recommended_next_probe=next_probe,
                        keep_as_regression_case=True,
                        notes=(
                            "Anti-case rows dominate the evidence set."
                            if evidence
                            else "Signal did not dominate the current fast failure rows."
                        ),
                    )
                )
    elif problem == "zdt1":
        if not rows:
            target_ids = {
                "spread": target_id,
                "joint": str(config.get("secondary_target_id") or target_id).strip(),
            }
            candidate_specs = [
                (
                    hypothesis_ids[0] if len(hypothesis_ids) > 0 else "zdt1_fast_early_front_plateau",
                    target_ids["spread"],
                    "early_front_plateau",
                    "early_front_plateau_signal",
                    "investigate_population_generation_tradeoff",
                ),
                (
                    hypothesis_ids[1] if len(hypothesis_ids) > 1 else "zdt1_fast_late_spread_collapse",
                    target_ids["spread"],
                    "late_spread_collapse",
                    "late_spread_collapse_signal",
                    "investigate_refresh_vs_cooldown_timing",
                ),
                (
                    hypothesis_ids[2] if len(hypothesis_ids) > 2 else "zdt1_fast_refresh_timing_mismatch",
                    target_ids["joint"],
                    "refresh_timing_mismatch",
                    "refresh_timing_mismatch_signal",
                    "investigate_refresh_vs_cooldown_timing",
                ),
            ]
            for hypothesis_id, row_target, mechanism, signal_key, next_probe in candidate_specs:
                evidence = [row for row in focus_rows if float(row.get(signal_key, 0.0) or 0.0) > 0.0]
                rate = len(evidence) / len(focus_rows) if focus_rows else 0.0
                confidence = _confidence_from_rate(rate)
                rows.append(
                    _hypothesis_row(
                        hypothesis_id=hypothesis_id,
                        target_id=row_target,
                        problem="zdt1",
                        affected_default="configs/local_profiles/zdt1_diversity_injection_fast.json",
                        evidence_rows=len(evidence),
                        suspected_mechanism=mechanism,
                        supporting_trace_signals=(
                            f"{signal_key}_rate={rate:.2f}; mean_hv_loss="
                            f"{mean([_parse_float(row.get('hv_loss_pct_vs_quality')) or 0.0 for row in evidence]):.4f}"
                            if evidence
                            else f"{signal_key}_rate=0.00"
                        ),
                        confidence=confidence,
                        current_evidence_strength=confidence,
                        probe_attempted="none",
                        confirmed_or_weakened="not_tested",
                        recommended_next_probe=next_probe,
                        keep_as_regression_case=True,
                        notes=(
                            "Safety-fail rows dominate the evidence set."
                            if evidence
                            else "Signal did not dominate the hardened fast failure rows."
                        ),
                    )
                )
    rows.sort(
        key=lambda item: (
            _TARGET_PRIORITY.get(str(item.get("target_id") or ""), 99),
            _probe_status_rank(str(item.get("confirmed_or_weakened") or "")),
            _confidence_rank(
                str(item.get("current_evidence_strength") or item.get("confidence") or "")
            ),
            -(_parse_int(item.get("evidence_rows")) or 0),
            str(item.get("hypothesis_id") or ""),
        )
    )
    return rows


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _confidence_rank(value: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(value, 3)


def _pick_top_hypothesis(rows: Sequence[Mapping[str, Any]], target_id: str) -> Mapping[str, Any] | None:
    candidates = [row for row in rows if str(row.get("target_id") or "") == target_id]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            _probe_status_rank(str(item.get("confirmed_or_weakened") or "")),
            _confidence_rank(
                str(item.get("current_evidence_strength") or item.get("confidence") or "")
            ),
            -(_parse_int(item.get("evidence_rows")) or 0),
            str(item.get("hypothesis_id") or ""),
        ),
    )[0]


def _pick_supported_hypothesis(rows: Sequence[Mapping[str, Any]], target_id: str) -> Mapping[str, Any] | None:
    hypothesis = _pick_top_hypothesis(rows, target_id)
    if hypothesis is None:
        return None
    if (_parse_int(hypothesis.get("evidence_rows")) or 0) <= 0:
        return None
    return hypothesis


def _severity_from_trace(problem: str, trace_rows: Sequence[Mapping[str, Any]], target_id: str) -> str:
    candidates = [row for row in trace_rows if str(row.get("target_id") or "") == target_id]
    if not candidates:
        return "low"
    if problem == "tsp":
        anti_losses = [
            _parse_float(row.get("route_distance_loss_pct_vs_quality")) or 0.0
            for row in candidates
            if str(row.get("case_group") or "") == "anti_case"
        ]
        p95 = _quantile(anti_losses, 0.95) or 0.0
        if p95 >= 2.5:
            return "high"
        if p95 >= 1.0:
            return "medium"
        return "low"
    hv_losses = [_parse_float(row.get("hv_loss_pct_vs_quality")) or 0.0 for row in candidates]
    joint_rate = mean([float(row.get("joint_safety_fail", 0.0) or 0.0) for row in candidates])
    if joint_rate >= 0.2 or (_quantile(hv_losses, 0.9) or 0.0) >= 0.3:
        return "high"
    if joint_rate > 0.0 or (_quantile(hv_losses, 0.9) or 0.0) >= 0.15:
        return "medium"
    return "low"


def _update_target_row(
    previous: Mapping[str, Any],
    *,
    current_severity: str,
    mechanism_hypothesis: str,
    latest_decision: str,
    recommended_next_action: str,
    notes: str,
) -> dict[str, Any]:
    previous_severity = str(
        previous.get("current_severity")
        or previous.get("severity")
        or previous.get("previous_severity")
        or "medium"
    )
    changed = (
        "reduced_but_not_closed"
        if current_severity != previous_severity and previous_severity == "high" and current_severity == "medium"
        else "unchanged_keep_default"
    )
    row = dict(previous)
    row["previous_severity"] = previous_severity
    row["current_severity"] = current_severity
    row["changed_or_not"] = changed
    row["current_mechanism_hypothesis"] = mechanism_hypothesis
    row["current_best_hypothesis"] = mechanism_hypothesis
    row["latest_decision"] = latest_decision
    row["recommended_next_action"] = recommended_next_action
    row["next_recommended_probe"] = recommended_next_action
    row["notes"] = notes
    return row


def _tsp_freeze_decision(study_dir: Path) -> str:
    summary_paths = [
        study_dir / "tsp_protocol_limitation_freeze_summary.md",
        study_dir / "tsp_irreducible_tail_freeze_summary.md",
    ]
    return (
        "freeze_as_protocol_limitation"
        if any(path.exists() for path in summary_paths)
        else "monitor_only"
    )


def _zdt1_spread_validation_decision(study_dir: Path) -> tuple[str, str]:
    summary_path = study_dir / "zdt1_spread_candidate_boundary_table.csv"
    if not summary_path.exists():
        summary_path = study_dir / "zdt1_spread_candidate_validation_summary.csv"
    if not summary_path.exists():
        return ("monitor_only", "keep_default_and_monitor")
    rows = _read_csv_rows(summary_path)
    baseline = next(
        (
            row
            for row in rows
            if str(row.get("scope") or "") == "overall"
            and str(row.get("study_variant") or "") == "current_fast"
        ),
        None,
    )
    candidate = next(
        (
            row
            for row in rows
            if str(row.get("scope") or "") == "overall"
            and str(row.get("study_variant") or "") == "spread_pg_pop41_gen88"
        ),
        None,
    )
    if baseline is None or candidate is None:
        return ("monitor_only", "keep_default_and_monitor")

    baseline_spread_fail = _parse_float(baseline.get("spread_fail_rate")) or 0.0
    candidate_spread_fail = _parse_float(candidate.get("spread_fail_rate")) or 0.0
    baseline_joint_fail = _parse_float(baseline.get("joint_safety_fail_rate")) or 0.0
    candidate_joint_fail = _parse_float(candidate.get("joint_safety_fail_rate")) or 0.0
    baseline_p90 = _parse_float(baseline.get("p90_loss_pct"))
    candidate_p90 = _parse_float(candidate.get("p90_loss_pct"))
    baseline_p95_spread = _parse_float(baseline.get("spread_delta_p95"))
    candidate_p95_spread = _parse_float(candidate.get("spread_delta_p95"))
    baseline_pareto = _parse_float(baseline.get("pareto_ratio_delta_mean"))
    candidate_pareto = _parse_float(candidate.get("pareto_ratio_delta_mean"))
    spread_stress_baseline = next(
        (
            row
            for row in rows
            if str(row.get("scope") or "") == "slice"
            and str(row.get("slice") or "") == "spread_stress"
            and str(row.get("study_variant") or "") == "current_fast"
        ),
        None,
    )
    spread_stress_candidate = next(
        (
            row
            for row in rows
            if str(row.get("scope") or "") == "slice"
            and str(row.get("slice") or "") == "spread_stress"
            and str(row.get("study_variant") or "") == "spread_pg_pop41_gen88"
        ),
        None,
    )
    stable_baseline = next(
        (
            row
            for row in rows
            if str(row.get("scope") or "") == "slice"
            and str(row.get("slice") or "") == "stable_contrast"
            and str(row.get("study_variant") or "") == "current_fast"
        ),
        None,
    )
    stable_candidate = next(
        (
            row
            for row in rows
            if str(row.get("scope") or "") == "slice"
            and str(row.get("slice") or "") == "stable_contrast"
            and str(row.get("study_variant") or "") == "spread_pg_pop41_gen88"
        ),
        None,
    )
    normal_baseline = next(
        (
            row
            for row in rows
            if str(row.get("scope") or "") == "slice"
            and str(row.get("slice") or "") == "normal_holdout"
            and str(row.get("study_variant") or "") == "current_fast"
        ),
        None,
    )
    normal_candidate = next(
        (
            row
            for row in rows
            if str(row.get("scope") or "") == "slice"
            and str(row.get("slice") or "") == "normal_holdout"
            and str(row.get("study_variant") or "") == "spread_pg_pop41_gen88"
        ),
        None,
    )

    if (
        candidate_spread_fail <= baseline_spread_fail - 0.05
        and (baseline_p90 is None or candidate_p90 is None or candidate_p90 <= baseline_p90 + 0.02)
        and (baseline_p95_spread is None or candidate_p95_spread is None or candidate_p95_spread <= baseline_p95_spread - 0.005)
        and candidate_joint_fail <= baseline_joint_fail
        and (baseline_pareto is None or candidate_pareto is None or candidate_pareto >= baseline_pareto - 0.002)
        and (
            stable_baseline is None
            or stable_candidate is None
            or (
                (_parse_float(stable_candidate.get("spread_fail_rate")) or 0.0)
                <= (_parse_float(stable_baseline.get("spread_fail_rate")) or 0.0)
            )
        )
        and (
            normal_baseline is None
            or normal_candidate is None
            or (
                (_parse_float(normal_candidate.get("spread_fail_rate")) or 0.0)
                <= (_parse_float(normal_baseline.get("spread_fail_rate")) or 0.0)
            )
        )
    ):
        return ("replace_fast_default", "keep_default_and_monitor")
    spread_stress_baseline_fail = _parse_float((spread_stress_baseline or {}).get("spread_fail_rate")) or 0.0
    spread_stress_candidate_fail = _parse_float((spread_stress_candidate or {}).get("spread_fail_rate")) or 0.0
    spread_stress_baseline_p95 = _parse_float((spread_stress_baseline or {}).get("spread_delta_p95"))
    spread_stress_candidate_p95 = _parse_float((spread_stress_candidate or {}).get("spread_delta_p95"))
    stable_regressed = (
        stable_baseline is not None
        and stable_candidate is not None
        and (_parse_float(stable_candidate.get("spread_fail_rate")) or 0.0)
        > (_parse_float(stable_baseline.get("spread_fail_rate")) or 0.0)
    )
    normal_regressed = (
        normal_baseline is not None
        and normal_candidate is not None
        and (_parse_float(normal_candidate.get("spread_fail_rate")) or 0.0)
        > (_parse_float(normal_baseline.get("spread_fail_rate")) or 0.0)
    )
    if (
        (
            candidate_spread_fail < baseline_spread_fail
            or (
                spread_stress_candidate_fail < spread_stress_baseline_fail
                and (
                    spread_stress_baseline_p95 is None
                    or spread_stress_candidate_p95 is None
                    or spread_stress_candidate_p95 < spread_stress_baseline_p95
                )
            )
        )
        and candidate_joint_fail <= baseline_joint_fail
        and (baseline_p90 is None or candidate_p90 is None or candidate_p90 <= baseline_p90 + 0.05)
        and (stable_regressed or normal_regressed)
    ):
        return ("note_only_stress_slice", "keep_as_slice_conditioned_note")
    if (
        stable_regressed
        and normal_regressed
        and candidate_spread_fail >= baseline_spread_fail
        and candidate_joint_fail > baseline_joint_fail
    ):
        return ("retire_candidate", "keep_default_and_monitor")
    return ("monitor_only", "keep_default_and_monitor")


def _zdt1_joint_note_decision(study_dir: Path) -> str:
    hypothesis_rows = _read_csv_rows(study_dir / "failure_hypotheses.csv") if (study_dir / "failure_hypotheses.csv").exists() else []
    joint_row = next(
        (
            row
            for row in hypothesis_rows
            if str(row.get("target_id") or "") == "zdt1_fast_joint_safety_fail"
        ),
        None,
    )
    status = str((joint_row or {}).get("confirmed_or_weakened") or "")
    if status in {"weakened", "still_unisolated"}:
        return "monitor_only"
    return "note_only"


def _render_failure_trace_notes(
    hypotheses: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["# Failure Trace Notes", ""]
    strongest = sorted(
        hypotheses,
        key=lambda item: (
            _TARGET_PRIORITY.get(str(item.get("target_id") or ""), 99),
            _probe_status_rank(str(item.get("confirmed_or_weakened") or "")),
            _confidence_rank(
                str(item.get("current_evidence_strength") or item.get("confidence") or "")
            ),
            -(_parse_int(item.get("evidence_rows")) or 0),
        ),
    )
    if strongest:
        lines.append("## Strongest Mechanism Hypotheses")
        lines.append("")
        for row in strongest:
            lines.append(
                f"- `{row.get('target_id')}` / `{row.get('hypothesis_id')}`: "
                f"{row.get('suspected_mechanism')} "
                f"(status={row.get('confirmed_or_weakened')}, "
                f"confidence={row.get('current_evidence_strength') or row.get('confidence')}, "
                f"next={row.get('recommended_next_action') or row.get('recommended_next_probe')})"
            )
        lines.append("")

    target_map = {str(row.get("target_id") or ""): row for row in targets}
    tsp_row = target_map.get("tsp_fast_anti_case_tail")
    if tsp_row is not None:
        lines.append("## TSP")
        lines.append("")
        mechanism = str(tsp_row.get("current_mechanism_hypothesis") or "")
        if mechanism == "tsp_anticase_late_refinement_deficit":
            lines.append(
                "- Anti-case caution stays in place because the current fast default can keep an early seeded head-start "
                "but still under-refine the late corridor cleanup window."
            )
        elif mechanism == "tsp_anticase_seed_lockin_secondary":
            lines.append(
                "- Anti-case caution stays in place because seeded head-start plus early diversity collapse still shows up "
                "in the hardest corridor-like rows."
            )
        else:
            lines.append(
                "- Anti-case caution stays in place because the current fast default still fails mostly through "
                f"`{mechanism}`."
            )
        lines.append("- The protocol should keep Q-first final runs for corridor-like or anti-case-suspected instances.")
        lines.append("")

    zdt1_row = target_map.get("zdt1_fast_spread_safety_fail")
    zdt1_joint_row = target_map.get("zdt1_fast_joint_safety_fail")
    if zdt1_row is not None:
        lines.append("## ZDT1")
        lines.append("")
        mechanism = str(zdt1_row.get("current_mechanism_hypothesis") or "")
        joint_mechanism = str((zdt1_joint_row or {}).get("current_mechanism_hypothesis") or "")
        if mechanism == "zdt1_fast_spread_mechanism_population_generation_candidate":
            lines.append(
                "- Final safety remains on Q, but the spread-fail slice now looks more like a population/generation "
                "tradeoff problem than a timing-only problem."
            )
        elif mechanism == "zdt1_fast_spread_mechanism_still_unisolated":
            lines.append(
                "- Final safety remains on Q because the hardened fast default still leaves spread-fail rows that are not "
                "yet explained by one clean mechanism, even when HV looks acceptable."
            )
        elif mechanism == "zdt1_fast_refresh_timing_mismatch_joint":
            lines.append(
                "- Final safety remains on Q because the hardened fast default still shows a refresh/cooldown timing "
                "mismatch in the joint safety-fail slice."
            )
        elif mechanism == "zdt1_fast_safety_mechanism_not_yet_isolated":
            lines.append(
                "- Final safety remains on Q because the hardened fast default still has spread/joint safety failures, "
                "but the refreshed trace did not isolate one dominant mechanism yet."
            )
        else:
            lines.append(
                "- Final safety remains on Q because the hardened fast default still shows "
                f"`{zdt1_row.get('current_mechanism_hypothesis')}` in the safety-fail slice."
            )
        if joint_mechanism == "zdt1_fast_refresh_timing_mismatch_joint":
            lines.append(
                "- Joint failures now read more like a timing mismatch than the spread failures do, so the two slices "
                "should stay separated in the next reduction pass."
            )
        lines.append("- F remains acceptable for exploratory HV checks, but not as the final safety authority.")
        lines.append("")

    lines.append("## Knapsack / Onemax")
    lines.append("")
    lines.append("- Knapsack stays on the narrow repair-only note with no broader rule expansion.")
    lines.append("- Onemax remains a control-only baseline with no active optimization target.")
    lines.append("")
    return "\n".join(lines)


def build_failure_hypothesis_registry(
    *,
    previous_registry_path: Path,
    output_dir: Path,
    tsp_study_dir: Path,
    zdt1_study_dir: Path,
    additional_zdt1_study_dirs: Sequence[Path] | None = None,
    include_knapsack_freeze: bool = True,
    include_onemax_freeze: bool = True,
) -> dict[str, str]:
    previous_rows = _load_json_rows(previous_registry_path)
    previous_map = {
        str(row.get("target_id") or ""): row
        for row in previous_rows
        if isinstance(row, dict) and str(row.get("target_id") or "").strip()
    }

    tsp_trace_rows = _read_csv_rows(tsp_study_dir / "failure_trace_table.csv")
    zdt1_dirs = [zdt1_study_dir, *(additional_zdt1_study_dirs or [])]
    zdt1_trace_rows: list[dict[str, Any]] = []
    zdt1_hypothesis_rows: list[dict[str, Any]] = []
    for study_dir in zdt1_dirs:
        zdt1_trace_rows.extend(_read_csv_rows(study_dir / "failure_trace_table.csv"))
        zdt1_hypothesis_rows.extend(_read_csv_rows(study_dir / "failure_hypotheses.csv"))
    tsp_hypothesis_rows = _read_csv_rows(tsp_study_dir / "failure_hypotheses.csv")
    hypothesis_rows: list[dict[str, Any]] = [*tsp_hypothesis_rows, *zdt1_hypothesis_rows]

    refreshed_targets: list[dict[str, Any]] = []
    for target_id, previous in previous_map.items():
        if target_id == "tsp_fast_anti_case_tail":
            hypothesis = _pick_top_hypothesis(hypothesis_rows, target_id)
            latest_decision = _tsp_freeze_decision(tsp_study_dir)
            refreshed_targets.append(
                _update_target_row(
                    previous,
                    current_severity=_severity_from_trace("tsp", tsp_trace_rows, target_id),
                    mechanism_hypothesis=str((hypothesis or {}).get("hypothesis_id") or "tsp_anticase_mechanism_not_yet_isolated"),
                    latest_decision=latest_decision,
                    recommended_next_action=str(
                        "open_new_mechanism_only"
                        if latest_decision == "freeze_as_protocol_limitation"
                        else (hypothesis or {}).get("recommended_next_action")
                        or (hypothesis or {}).get("recommended_next_probe")
                        or "investigate_population_generation_tradeoff"
                    ),
                    notes=(
                        "Current TSP fast still needs anti-case caution; same-budget contour tuning no longer looks like the honest way to close the anti-case p95/max tail."
                        if latest_decision == "freeze_as_protocol_limitation"
                        else "Current TSP fast still needs anti-case caution; the failure trace now narrows the dominant mechanism."
                    ),
                )
            )
        elif target_id == "tsp_rescue_target_ambiguity":
            hypothesis = _pick_top_hypothesis(hypothesis_rows, "tsp_fast_anti_case_tail")
            refreshed_targets.append(
                _update_target_row(
                    previous,
                    current_severity=str(previous.get("severity") or previous.get("current_severity") or "medium"),
                    mechanism_hypothesis=str((hypothesis or {}).get("hypothesis_id") or "tsp_rescue_target_secondary_follow_up"),
                    latest_decision="monitor_only",
                    recommended_next_action="keep_secondary_regression_slice; do not outrank anti-case tail",
                    notes="Rescue-target ambiguity remains secondary until the anti-case tail is materially reduced.",
                )
            )
        elif target_id == "zdt1_fast_spread_safety_fail":
            hypothesis = _pick_supported_hypothesis(hypothesis_rows, target_id)
            latest_decision, next_action = _zdt1_spread_validation_decision(zdt1_study_dir)
            refreshed_targets.append(
                _update_target_row(
                    previous,
                    current_severity=_severity_from_trace("zdt1", zdt1_trace_rows, target_id),
                    mechanism_hypothesis=str((hypothesis or {}).get("hypothesis_id") or "zdt1_fast_safety_mechanism_not_yet_isolated"),
                    latest_decision=latest_decision,
                    recommended_next_action=str(
                        "keep_default_and_monitor"
                        if latest_decision == "replace_fast_default"
                        else next_action
                        or (hypothesis or {}).get("recommended_next_action")
                        or (hypothesis or {}).get("recommended_next_probe")
                        or "keep_default_and_monitor"
                    ),
                    notes=(
                        "The spread candidate now clears same-budget validation well enough to justify a same-name fast-default replacement."
                        if latest_decision == "replace_fast_default"
                        else "The spread candidate improved the spread-stress slice without surviving stable/normal non-regression, so keep it only as a slice-conditioned note."
                        if latest_decision == "note_only_stress_slice"
                        else "The spread candidate regressed too much outside the spread-stress slice, so retire it from default-replacement consideration."
                        if latest_decision == "retire_candidate"
                        else "Current ZDT1 fast still keeps final safety on Q; the refreshed trace did not isolate a promotion-worthy spread replacement."
                        if hypothesis is None
                        else "Current ZDT1 fast still keeps final safety on Q; the failure trace now narrows the safety-fail mechanism."
                    ),
                )
            )
        elif target_id == "zdt1_fast_joint_safety_fail":
            hypothesis = _pick_supported_hypothesis(hypothesis_rows, target_id)
            latest_decision = _zdt1_joint_note_decision(zdt1_dirs[-1])
            refreshed_targets.append(
                _update_target_row(
                    previous,
                    current_severity=_severity_from_trace("zdt1", zdt1_trace_rows, target_id),
                    mechanism_hypothesis=str((hypothesis or {}).get("hypothesis_id") or "zdt1_fast_joint_mechanism_still_unisolated"),
                    latest_decision=latest_decision,
                    recommended_next_action=str(
                        "monitor_with_Q_final_safety"
                        if latest_decision == "monitor_only"
                        else (hypothesis or {}).get("recommended_next_action")
                        or (hypothesis or {}).get("recommended_next_probe")
                        or "split_spread_vs_joint_fail_mechanisms"
                    ),
                    notes=(
                        "Timing-only joint probes still do not justify a fast-default change; keep final safety on Q and monitor the joint slice separately."
                    ),
                )
            )
        elif target_id.startswith("knapsack_") and include_knapsack_freeze:
            row = dict(previous)
            row["previous_severity"] = str(previous.get("severity") or previous.get("current_severity") or "medium")
            row["current_severity"] = str(previous.get("severity") or previous.get("current_severity") or "medium")
            row["changed_or_not"] = "unchanged_keep_default"
            row["current_mechanism_hypothesis"] = "keep_narrow_repair_note_only"
            row["current_best_hypothesis"] = "keep_narrow_repair_note_only"
            row["latest_decision"] = "monitor_only"
            row["recommended_next_action"] = "keep_default_and_monitor"
            row["next_recommended_probe"] = "keep_default_and_monitor"
            refreshed_targets.append(row)
        elif target_id == "onemax_no_active_target" and include_onemax_freeze:
            row = dict(previous)
            row["previous_severity"] = str(previous.get("severity") or previous.get("current_severity") or "low")
            row["current_severity"] = "low"
            row["changed_or_not"] = "unchanged_keep_default"
            row["current_mechanism_hypothesis"] = "no_active_target"
            row["current_best_hypothesis"] = "no_active_target"
            row["latest_decision"] = "no_active_target"
            row["recommended_next_action"] = "no_active_target"
            row["next_recommended_probe"] = "no_active_target"
            refreshed_targets.append(row)

    refreshed_targets.sort(
        key=lambda item: (
            _TARGET_PRIORITY.get(str(item.get("target_id") or ""), 99),
            str(item.get("target_id") or ""),
        )
    )

    failure_hypotheses_json = output_dir / "failure_hypotheses.json"
    failure_hypotheses_csv = output_dir / "failure_hypotheses.csv"
    failure_hypotheses_md = output_dir / "failure_hypotheses.md"
    refreshed_targets_json = output_dir / "future_optimization_targets.json"
    refreshed_targets_csv = output_dir / "future_optimization_targets.csv"
    refreshed_targets_md = output_dir / "future_optimization_targets.md"
    failure_trace_notes_md = output_dir / "failure_trace_notes.md"

    output_dir.mkdir(parents=True, exist_ok=True)
    failure_hypotheses_json.write_text(json.dumps(hypothesis_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_rows_csv(failure_hypotheses_csv, list(hypothesis_rows), FAILURE_HYPOTHESIS_COLUMNS)
    failure_hypotheses_md.write_text(
        "# Failure Hypotheses\n\n" + _markdown_table(list(hypothesis_rows), FAILURE_HYPOTHESIS_COLUMNS),
        encoding="utf-8",
    )

    refreshed_targets_json.write_text(json.dumps(refreshed_targets, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_rows_csv(refreshed_targets_csv, refreshed_targets, FUTURE_TARGET_COLUMNS)
    refreshed_targets_md.write_text(
        "# Future Optimization Targets\n\n" + _markdown_table(refreshed_targets, FUTURE_TARGET_COLUMNS),
        encoding="utf-8",
    )
    failure_trace_notes_md.write_text(_render_failure_trace_notes(hypothesis_rows, refreshed_targets), encoding="utf-8")

    return {
        "failure_hypotheses_json": str(failure_hypotheses_json.resolve()),
        "failure_hypotheses_csv": str(failure_hypotheses_csv.resolve()),
        "failure_hypotheses_md": str(failure_hypotheses_md.resolve()),
        "future_optimization_targets_json": str(refreshed_targets_json.resolve()),
        "future_optimization_targets_csv": str(refreshed_targets_csv.resolve()),
        "future_optimization_targets_md": str(refreshed_targets_md.resolve()),
        "failure_trace_notes_md": str(failure_trace_notes_md.resolve()),
    }
