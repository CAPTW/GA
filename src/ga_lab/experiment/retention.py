from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


def _label(row: Mapping[str, Any]) -> str:
    return str(row.get("baseline_label") or row.get("run_name") or "unlabeled")


def _parse_timestamp(row: Mapping[str, Any]) -> datetime:
    output_dir = row.get("output_dir")
    if not isinstance(output_dir, str):
        return datetime.min
    stem = Path(output_dir).name
    timestamp = stem.split("_", 1)[0]
    try:
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return datetime.min


def _ranking_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    maximize = bool(row.get("maximize", True))
    best_fitness = row.get("best_fitness")
    mean_fitness = row.get("mean_fitness")
    final_generation = row.get("final_generation")

    best_value = float(best_fitness) if isinstance(best_fitness, int | float) else float("-inf")
    mean_value = float(mean_fitness) if isinstance(mean_fitness, int | float) else float("-inf")
    final_generation_value = (
        float(final_generation) if isinstance(final_generation, int | float) else float("inf")
    )

    if maximize:
        best_rank = -best_value
        mean_rank = -mean_value
    else:
        best_rank = best_value
        mean_rank = mean_value

    return (
        best_rank,
        mean_rank,
        final_generation_value,
        -_parse_timestamp(row).timestamp() if _parse_timestamp(row) != datetime.min else 0.0,
        str(row.get("run_name") or ""),
    )


def build_retention_plan(
    results_name: str,
    rows: list[Mapping[str, Any]],
    *,
    mode: str = "history-only",
    keep_best_runs: int = 2,
    keep_latest_runs: int = 1,
) -> dict[str, Any]:
    if mode not in {"history-only", "run-dirs"}:
        raise ValueError("mode must be either 'history-only' or 'run-dirs'")
    if keep_best_runs < 0 or keep_latest_runs < 0:
        raise ValueError("keep_best_runs and keep_latest_runs must be non-negative")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_label(row), []).append(row)

    actions = []
    keep_full_count = 0
    prune_history_count = 0
    drop_run_dir_count = 0
    for label in sorted(grouped):
        group_rows = grouped[label]
        keep_keys: dict[tuple[str, str], list[str]] = {}
        common_stop_reason = Counter(
            str(row.get("stop_reason")) for row in group_rows if row.get("stop_reason") is not None
        ).most_common(1)
        dominant_stop_reason = common_stop_reason[0][0] if common_stop_reason else None

        for row in sorted(group_rows, key=_ranking_key)[:keep_best_runs]:
            key = (str(row.get("run_name")), str(row.get("output_dir")))
            keep_keys.setdefault(key, []).append("top_best")

        latest_rows = sorted(group_rows, key=_parse_timestamp, reverse=True)[:keep_latest_runs]
        for row in latest_rows:
            key = (str(row.get("run_name")), str(row.get("output_dir")))
            keep_keys.setdefault(key, []).append("latest")

        if dominant_stop_reason is not None:
            for row in group_rows:
                if str(row.get("stop_reason")) != dominant_stop_reason:
                    key = (str(row.get("run_name")), str(row.get("output_dir")))
                    keep_keys.setdefault(key, []).append("diagnostic_stop_reason")

        for row in sorted(
            group_rows,
            key=lambda item: (_parse_timestamp(item), str(item.get("run_name"))),
        ):
            key = (str(row.get("run_name")), str(row.get("output_dir")))
            reasons = keep_keys.get(key, [])
            if reasons:
                action = "keep_full"
                keep_full_count += 1
            elif mode == "history-only":
                action = "prune_history"
                prune_history_count += 1
            else:
                action = "drop_run_dir"
                drop_run_dir_count += 1

            output_dir = row.get("output_dir")
            history_path = (
                str(Path(output_dir) / "history.csv") if isinstance(output_dir, str) else None
            )
            actions.append(
                {
                    "baseline_label": label,
                    "run_name": row.get("run_name"),
                    "seed": row.get("seed"),
                    "action": action,
                    "reasons": reasons,
                    "output_dir": output_dir,
                    "history_path": history_path,
                }
            )

    return {
        "results_name": results_name,
        "policy": {
            "mode": mode,
            "keep_best_runs": keep_best_runs,
            "keep_latest_runs": keep_latest_runs,
            "keep_suite_artifacts": True,
            "suite_artifacts": [
                "manifest.json",
                "suite_summary.json",
                "results_summary.json",
                "SUMMARY.md",
                "RUNS.csv",
                "RUNS.jsonl",
                "*_grid_summary.json",
            ],
        },
        "totals": {
            "run_count": len(rows),
            "keep_full_count": keep_full_count,
            "prune_history_count": prune_history_count,
            "drop_run_dir_count": drop_run_dir_count,
        },
        "run_actions": actions,
    }


def render_retention_markdown(plan: Mapping[str, Any]) -> str:
    lines = [f"# Retention Plan: {plan['results_name']}", ""]
    policy = plan.get("policy", {})
    totals = plan.get("totals", {})
    lines.append(f"- Mode: `{policy.get('mode', '-')}`")
    lines.append(f"- Keep best runs per label: {policy.get('keep_best_runs', '-')}")
    lines.append(f"- Keep latest runs per label: {policy.get('keep_latest_runs', '-')}")
    lines.append(f"- Total runs: {totals.get('run_count', 0)}")
    lines.append(f"- Keep full: {totals.get('keep_full_count', 0)}")
    lines.append(f"- Prune history: {totals.get('prune_history_count', 0)}")
    lines.append(f"- Drop run dirs: {totals.get('drop_run_dir_count', 0)}")
    lines.append("")
    lines.append("| Label | Run | Seed | Action | Reasons |")
    lines.append("| --- | --- | ---: | --- | --- |")
    for action in plan.get("run_actions", []):
        lines.append(
            "| {label} | {run_name} | {seed} | {action_name} | {reasons} |".format(
                label=action.get("baseline_label", "-"),
                run_name=action.get("run_name", "-"),
                seed=action.get("seed", "-"),
                action_name=action.get("action", "-"),
                reasons=", ".join(action.get("reasons", [])) or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_retention_bundle(
    results_dir: str | Path,
    plan: Mapping[str, Any],
    *,
    plan_filename: str = "retention_plan.json",
    markdown_filename: str = "RETENTION.md",
) -> dict[str, Path]:
    root = Path(results_dir)
    plan_path = root / plan_filename
    markdown_path = root / markdown_filename
    plan_path.write_text(json.dumps(dict(plan), indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_retention_markdown(plan), encoding="utf-8")
    return {
        "retention_plan": plan_path,
        "retention_markdown": markdown_path,
    }


def apply_retention_plan(plan: Mapping[str, Any], *, apply: bool = False) -> dict[str, int]:
    pruned_history = 0
    deleted_run_dirs = 0
    for action in plan.get("run_actions", []):
        action_name = action.get("action")
        if action_name == "prune_history":
            history_path = action.get("history_path")
            if not isinstance(history_path, str):
                continue
            path = Path(history_path)
            if apply and path.exists():
                path.unlink()
            if path.exists() or apply:
                pruned_history += 1
        elif action_name == "drop_run_dir":
            output_dir = action.get("output_dir")
            if not isinstance(output_dir, str):
                continue
            path = Path(output_dir)
            if apply and path.exists():
                shutil.rmtree(path)
            if path.exists() or apply:
                deleted_run_dirs += 1
    return {
        "pruned_history_count": pruned_history,
        "deleted_run_dir_count": deleted_run_dirs,
    }
