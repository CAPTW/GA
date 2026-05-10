from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ga_lab.experiment.comparison import (
    write_comparison_csv,
    write_comparison_jsonl,
)


def _format_number(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_percent(value: Any) -> str:
    if not isinstance(value, int | float):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _join(values: Any) -> str:
    if not isinstance(values, list):
        return "-"
    if not values:
        return "-"
    return ", ".join(str(value) for value in values)


def render_collection_markdown(summary: Mapping[str, Any]) -> str:
    lines = [f"# GA Comparison Report: {summary['suite_name']}", ""]
    lines.append(f"- Baselines: {summary.get('baseline_count', 0)}")
    lines.append(f"- Runs: {summary.get('run_count', 0)}")
    manifest_path = summary.get("manifest_path")
    if manifest_path:
        lines.append(f"- Manifest: `{manifest_path}`")
    lines.append("")

    baselines = summary.get("baselines")
    if isinstance(baselines, list) and baselines:
        lines.append("## Baseline Summary")
        lines.append("")
        lines.append(
            "| Label | Runs | Seeds | Selection | Crossover | Mutation | "
            "Mean best | Median best | Std best | Mean gen | Std gen | Target hit |"
        )
        lines.append(
            "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for baseline in baselines:
            lines.append(
                (
                    "| {label} | {runs} | {seeds} | {selection} | {crossover} | "
                    "{mutation} | {mean_best} | {median_best} | {stdev_best} | "
                    "{mean_gen} | {stdev_gen} | {target_hit} |"
                ).format(
                    label=baseline["baseline_label"],
                    runs=baseline["run_count"],
                    seeds=_join(baseline.get("seed_values")),
                    selection=_join(baseline.get("selections")),
                    crossover=_join(baseline.get("crossovers")),
                    mutation=_join(baseline.get("mutations")),
                    mean_best=_format_number(baseline.get("mean_best_fitness")),
                    median_best=_format_number(baseline.get("median_best_fitness")),
                    stdev_best=_format_number(baseline.get("stdev_best_fitness")),
                    mean_gen=_format_number(baseline.get("mean_final_generation")),
                    stdev_gen=_format_number(baseline.get("stdev_final_generation")),
                    target_hit=_format_percent(baseline.get("target_hit_rate")),
                )
            )
        lines.append("")

    comparison_groups = summary.get("comparison_groups")
    if isinstance(comparison_groups, list) and comparison_groups:
        lines.append("## Comparison Groups")
        lines.append("")
        for group in comparison_groups:
            baseline_labels = group.get("baseline_labels", [])
            if not isinstance(baseline_labels, list) or len(baseline_labels) < 2:
                continue
            signature = group.get("signature", {})
            if isinstance(signature, Mapping):
                signature_text = ", ".join(
                    f"{key}={value}" for key, value in signature.items() if value is not None
                )
            else:
                signature_text = "-"
            lines.append(f"### {signature.get('problem', 'comparison')} ({signature_text})")
            lines.append("")
            lines.append(
                f"- Best by mean best_fitness: `{group.get('best_by_mean_best_fitness') or '-'}`"
            )
            lines.append(
                f"- Best by mean mean_fitness: `{group.get('best_by_mean_mean_fitness') or '-'}`"
            )
            lines.append(
                "- Best by mean final_generation: "
                f"`{group.get('best_by_mean_final_generation') or '-'}`"
            )
            lines.append("")
            lines.append(
                "| Left | Right | Paired seeds | Winner(best) | "
                "Mean best delta (L-R) | Winner(gen) | "
                "Mean gen delta (L-R) | Winner(runtime) |"
            )
            lines.append("| --- | --- | ---: | --- | ---: | --- | ---: | --- |")
            for pair in group.get("paired_comparisons", []):
                best = pair["best_fitness"]
                generations = pair["final_generation"]
                runtime = pair["runtime_seconds"]
                lines.append(
                    (
                        "| {left} | {right} | {paired} | {best_winner} | "
                        "{best_delta} | {gen_winner} | {gen_delta} | "
                        "{runtime_winner} |"
                    ).format(
                        left=pair["left_label"],
                        right=pair["right_label"],
                        paired=best.get("paired_seed_count", 0),
                        best_winner=best.get("winner_by_mean", "-"),
                        best_delta=_format_number(best.get("left_minus_right")),
                        gen_winner=generations.get("winner_by_mean", "-"),
                        gen_delta=_format_number(generations.get("left_minus_right")),
                        runtime_winner=runtime.get("winner_by_mean", "-"),
                    )
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_collection_bundle(
    results_dir: str | Path,
    rows: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    summary_filename: str,
) -> dict[str, Path]:
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs_csv_path = root / "RUNS.csv"
    runs_jsonl_path = root / "RUNS.jsonl"
    summary_json_path = root / summary_filename
    summary_md_path = root / "SUMMARY.md"

    write_comparison_csv(runs_csv_path, rows)
    write_comparison_jsonl(runs_jsonl_path, rows)
    summary_json_path.write_text(
        json.dumps(dict(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_md_path.write_text(render_collection_markdown(summary), encoding="utf-8")

    return {
        "runs_csv": runs_csv_path,
        "runs_jsonl": runs_jsonl_path,
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
    }
