# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

import streamlit as st

from ga_ops.db import OpsDatabase
from ga_ops.settings import OpsSettings


def _load_dashboard(db_path: Path) -> dict[str, object]:
    with OpsDatabase(db_path) as database:
        database.initialize()
        return database.dashboard_summary()


def main() -> None:
    settings = OpsSettings.from_env(project_root=PROJECT_ROOT)
    st.set_page_config(page_title="GA Lab Ops Dashboard", layout="wide")
    st.title("GA Lab Ops Dashboard")

    db_path = Path(st.sidebar.text_input("Ops DB path", str(settings.db_path)))
    if not db_path.exists():
        st.warning("Ops DB not found. Run `python scripts/ops_sync_results.py` first.")
        return

    summary = _load_dashboard(db_path)
    totals = summary["totals"]
    metrics = st.columns(3)
    metrics[0].metric("Runs", totals["runs"])
    metrics[1].metric("Artifacts", totals["artifacts"])
    metrics[2].metric("Scheduled jobs", totals["scheduled_jobs"])

    run_tab, regression_tab, config_tab, scheduler_tab, audit_tab = st.tabs(
        ["Runs", "Regressions", "Best Configs", "Scheduler", "Audit"]
    )

    with run_tab:
        st.subheader("Recent runs")
        st.dataframe(summary["recent_runs"], use_container_width=True, hide_index=True)
        run_options = summary["recent_runs"]
        if run_options:
            selected = st.selectbox(
                "Inspect run artifacts",
                run_options,
                format_func=lambda item: (
                    f"{item['run_name']} ({item['problem']}, seed={item['seed']})"
                ),
            )
            with OpsDatabase(db_path) as database:
                database.initialize()
                st.dataframe(
                    database.list_run_artifacts(int(selected["id"])),
                    use_container_width=True,
                    hide_index=True,
                )

    with regression_tab:
        st.subheader("Recent regressions")
        regressions = summary["recent_regressions"]
        if regressions:
            st.dataframe(regressions, use_container_width=True, hide_index=True)
        else:
            st.info("No recent regressions detected.")

    with config_tab:
        st.subheader("Best configs")
        st.dataframe(summary["best_configs"], use_container_width=True, hide_index=True)

    with scheduler_tab:
        st.subheader("Scheduled jobs")
        st.dataframe(summary["scheduled_jobs"], use_container_width=True, hide_index=True)
        st.subheader("Recent job runs")
        st.dataframe(summary["recent_job_runs"], use_container_width=True, hide_index=True)

    with audit_tab:
        st.subheader("Recent audit events")
        st.dataframe(summary["recent_audit_logs"], use_container_width=True, hide_index=True)

    st.caption("Run with: `streamlit run scripts/ops_dashboard.py`")


if __name__ == "__main__":
    main()
