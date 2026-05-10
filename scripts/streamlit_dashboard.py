from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


def _load_summary_files(results_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(results_dir.glob("**/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["_summary_path"] = str(summary_path)
        payload["_history_path"] = str(summary_path.parent / "history.csv")
        summaries.append(payload)
    return summaries


def _load_grid_summary_files(results_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(results_dir.glob("**/*_grid_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["_grid_summary_path"] = str(summary_path)
        summaries.append(payload)
    return summaries


def _load_history(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _coerce_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    coerced_rows: list[dict[str, float]] = []
    for row in rows:
        coerced_rows.append({key: float(value) for key, value in row.items()})
    return coerced_rows


def _format_reference_point(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value)
    if value is None:
        return "-"
    return str(value)


def _render_reference_point_comparison(payload: dict[str, Any], heading: str) -> None:
    reference_keys = (
        "hypervolume_reference_point_preset",
        "hypervolume_reference_point_override",
        "hypervolume_reference_point",
        "hypervolume_reference_point_source",
    )
    if not any(key in payload for key in reference_keys):
        return

    st.subheader(heading)
    comparison = pd.DataFrame(
        [
            {
                "kind": "problem_preset",
                "value": _format_reference_point(payload.get("hypervolume_reference_point_preset")),
            },
            {
                "kind": "config_override",
                "value": _format_reference_point(
                    payload.get("hypervolume_reference_point_override")
                ),
            },
            {
                "kind": "resolved",
                "value": _format_reference_point(payload.get("hypervolume_reference_point")),
            },
            {
                "kind": "source",
                "value": _format_reference_point(payload.get("hypervolume_reference_point_source")),
            },
        ]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)


def _build_dual_axis_chart(
    dataframe: pd.DataFrame,
    left_columns: list[str],
    right_columns: list[str],
    left_title: str,
    right_title: str,
) -> alt.Chart | None:
    working = dataframe.reset_index()
    left_present = [column for column in left_columns if column in working.columns]
    right_present = [column for column in right_columns if column in working.columns]
    if not left_present and not right_present:
        return None

    color_scale = alt.Scale(
        domain=[
            "mean_pareto_ratio",
            "mean_normalized_hypervolume",
            "mean_hypervolume",
            "mean_spread",
            "mean_convergence_speed",
            "mean_pareto_front_size",
            "runs",
        ],
        range=[
            "#1b9e77",
            "#66a61e",
            "#d95f02",
            "#7570b3",
            "#e7298a",
            "#e6ab02",
            "#666666",
        ],
    )
    base = alt.Chart(working).encode(x=alt.X("generation:Q", title="Generation"))
    layers: list[alt.Chart] = []

    if left_present:
        left_chart = (
            base.transform_fold(left_present, as_=["metric", "value"])
            .mark_line(point=False)
            .encode(
                y=alt.Y(
                    "value:Q",
                    axis=alt.Axis(title=left_title, orient="left"),
                ),
                color=alt.Color(
                    "metric:N",
                    scale=color_scale,
                    legend=alt.Legend(title="Left axis"),
                ),
                tooltip=[
                    alt.Tooltip("generation:Q", title="Generation"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Value", format=".6f"),
                ],
            )
        )
        layers.append(left_chart)

    if right_present:
        right_chart = (
            base.transform_fold(right_present, as_=["metric", "value"])
            .mark_line(point=False, strokeDash=[6, 4])
            .encode(
                y=alt.Y(
                    "value:Q",
                    axis=alt.Axis(title=right_title, orient="right"),
                ),
                color=alt.Color(
                    "metric:N",
                    scale=color_scale,
                    legend=alt.Legend(title="Right axis"),
                ),
                tooltip=[
                    alt.Tooltip("generation:Q", title="Generation"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Value", format=".6f"),
                ],
            )
        )
        layers.append(right_chart)

    return alt.layer(*layers).resolve_scale(y="independent").interactive()


def _render_run_detail(summary: dict[str, Any]) -> None:
    history_path = Path(summary["_history_path"])
    history_rows = _coerce_history_rows(_load_history(history_path))
    history_df = pd.DataFrame(history_rows).set_index("generation")

    st.subheader("Run summary")
    st.json(summary)
    _render_reference_point_comparison(summary, "Reference Point Comparison")

    st.subheader("Convergence")
    st.line_chart(history_df[["best_fitness", "mean_fitness", "worst_fitness"]])

    if "pareto_front_size" in history_df.columns:
        st.subheader("Pareto diagnostics")
        pareto_columns = [
            column
            for column in (
                "pareto_front_size",
                "pareto_ratio",
                "hypervolume",
                "spread",
                "normalized_hypervolume",
                "convergence_speed",
            )
            if column in history_df.columns
        ]
        st.line_chart(history_df[pareto_columns])

    st.subheader("Objective diagnostics")
    objective_columns = [col for col in history_df.columns if col.startswith("best_objective_")]
    if objective_columns:
        st.line_chart(history_df[objective_columns])

    if st.toggle("Show history table", value=False):
        st.dataframe(history_rows)


def _render_grid_detail(grid_summary: dict[str, Any]) -> None:
    st.subheader("Grid summary")
    overview = {
        key: value
        for key, value in grid_summary.items()
        if key not in {"runs", "multiobjective_progress"}
    }
    st.json(overview)
    _render_reference_point_comparison(grid_summary, "Reference Point Comparison")

    progress_rows = grid_summary.get("multiobjective_progress")
    if isinstance(progress_rows, list) and progress_rows:
        progress_df = pd.DataFrame(progress_rows).set_index("generation")

        st.subheader("Multiobjective Progress")
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Mean Pareto Ratio",
            f"{float(grid_summary.get('mean_pareto_ratio', 0.0)):.4f}",
        )
        metric_columns[1].metric(
            "Mean Hypervolume",
            f"{float(grid_summary.get('mean_hypervolume', 0.0)):.4f}",
        )
        metric_columns[2].metric(
            "Mean Spread",
            f"{float(grid_summary.get('mean_spread', 0.0)):.4f}",
        )
        metric_columns[3].metric(
            "Mean Conv. Speed",
            f"{float(grid_summary.get('mean_convergence_speed', 0.0)):.6f}",
        )

        st.caption("Left axis uses solid lines. Right axis uses dashed lines.")

        attainment_chart = _build_dual_axis_chart(
            progress_df,
            left_columns=["mean_pareto_ratio", "mean_normalized_hypervolume"],
            right_columns=["mean_hypervolume"],
            left_title="Ratio / normalized attainment",
            right_title="Hypervolume",
        )
        if attainment_chart is not None:
            st.caption("Attainment vs hypervolume")
            st.altair_chart(attainment_chart, use_container_width=True)

        diversity_chart = _build_dual_axis_chart(
            progress_df,
            left_columns=["mean_spread"],
            right_columns=["mean_convergence_speed"],
            left_title="Spread",
            right_title="Convergence speed",
        )
        if diversity_chart is not None:
            st.caption("Diversity vs convergence speed")
            st.altair_chart(diversity_chart, use_container_width=True)

        front_size_chart = _build_dual_axis_chart(
            progress_df,
            left_columns=["mean_pareto_front_size"],
            right_columns=["runs"],
            left_title="Pareto front size",
            right_title="Runs",
        )
        if front_size_chart is not None:
            st.caption("Front size coverage")
            st.altair_chart(front_size_chart, use_container_width=True)

        if st.toggle("Show multiobjective progress table", value=False):
            st.dataframe(progress_rows)
    else:
        st.info("This grid summary does not include multiobjective progress data.")

    runs = grid_summary.get("runs")
    if isinstance(runs, list) and runs:
        runs_df = pd.DataFrame(runs)
        preferred_columns = [
            column
            for column in (
                "run_name",
                "seed",
                "best_fitness",
                "pareto_ratio",
                "hypervolume",
                "spread",
                "mean_convergence_speed",
            )
            if column in runs_df.columns
        ]
        if preferred_columns:
            st.subheader("Per-seed summary")
            st.dataframe(runs_df[preferred_columns])
        reference_columns = [
            column
            for column in (
                "run_name",
                "seed",
                "hypervolume_reference_point_preset",
                "hypervolume_reference_point_override",
                "hypervolume_reference_point",
                "hypervolume_reference_point_source",
            )
            if column in runs_df.columns
        ]
        if len(reference_columns) > 2:
            st.subheader("Per-seed reference points")
            reference_df = runs_df[reference_columns].copy()
            for column in reference_columns:
                if column not in {"run_name", "seed"}:
                    reference_df[column] = reference_df[column].map(_format_reference_point)
            st.dataframe(reference_df, use_container_width=True)
    with st.expander("Show raw grid summary", expanded=False):
        st.json(grid_summary)


def main() -> None:
    st.set_page_config(page_title="GA Codex Lab Dashboard", layout="wide")
    st.title("GA Codex Lab Dashboard")

    results_dir = Path(st.sidebar.text_input("Results directory", "outputs"))
    if not results_dir.exists():
        st.error("Results directory not found.")
        return

    summaries = _load_summary_files(results_dir)
    if not summaries:
        st.info("No summary files found.")
        return

    grid_summaries = _load_grid_summary_files(results_dir)
    run_tab, grid_tab = st.tabs(["Run detail", "Grid detail"])

    with run_tab:
        selected_run = st.sidebar.selectbox("Run", [entry["run_name"] for entry in summaries])
        selected = next(item for item in summaries if item["run_name"] == selected_run)
        _render_run_detail(selected)

    with grid_tab:
        if not grid_summaries:
            st.info("No grid summary files found.")
        else:
            selected_grid = st.sidebar.selectbox(
                "Grid summary",
                [entry["run_name"] for entry in grid_summaries],
            )
            selected_grid_summary = next(
                item for item in grid_summaries if item["run_name"] == selected_grid
            )
            _render_grid_detail(selected_grid_summary)

    st.caption("Run with: `streamlit run scripts/streamlit_dashboard.py`")


if __name__ == "__main__":
    main()
