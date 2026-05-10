from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .db import OpsDatabase
from .settings import OpsSettings


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _in_lookback(record: dict[str, Any], cutoff: datetime) -> bool:
    for key in ("run_created_at", "ingested_at", "started_at", "created_at"):
        parsed = _parse_timestamp(record.get(key))
        if parsed is not None:
            return parsed >= cutoff
    return False


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Weekly Report: {payload['report_name']}",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Lookback days: `{payload['lookback_days']}`",
        f"- Runs in window: `{payload['totals']['runs']}`",
        f"- Job runs in window: `{payload['totals']['job_runs']}`",
        f"- Audit events in window: `{payload['totals']['audit_events']}`",
        "",
        "## Runs By Problem",
        "",
    ]
    if payload["runs_by_problem"]:
        for problem, count in payload["runs_by_problem"].items():
            lines.append(f"- `{problem}`: {count}")
    else:
        lines.append("- No runs recorded in the lookback window.")

    lines.extend(["", "## Recent Regressions", ""])
    if payload["recent_regressions"]:
        for regression in payload["recent_regressions"]:
            lines.append(
                (
                    "- `{label}` on `{problem}`: reasons=`{reasons}` "
                    "latest_best=`{latest}` previous_best=`{previous}`"
                ).format(
                    label=regression["label"],
                    problem=regression["problem"],
                    reasons=", ".join(regression["reasons"]),
                    latest=regression.get("latest_best_fitness"),
                    previous=regression.get("previous_best_fitness"),
                )
            )
    else:
        lines.append("- No regressions detected by the current heuristic.")

    lines.extend(["", "## Best Configs", ""])
    for record in payload["best_configs"]:
        lines.append(
            (
                "- `{problem}` `{selection}/{crossover}/{mutation}`: "
                "mean_best=`{mean_best}` runs=`{runs}` source=`{source}`"
            ).format(
                problem=record.get("problem"),
                selection=record.get("selection"),
                crossover=record.get("crossover"),
                mutation=record.get("mutation"),
                mean_best=record.get("mean_best_fitness"),
                runs=record.get("runs"),
                source=record.get("source_config_path"),
            )
        )
    if not payload["best_configs"]:
        lines.append("- No ranked configs available.")

    lines.extend(["", "## Scheduler", ""])
    if payload["job_status"]:
        for record in payload["job_status"]:
            lines.append(
                (
                    "- `{name}`: enabled=`{enabled}` cron=`{cron}` "
                    "last_run=`{last_run}` next_run=`{next_run}`"
                ).format(
                    name=record.get("name"),
                    enabled=record.get("enabled"),
                    cron=record.get("cron"),
                    last_run=record.get("last_run_at") or "-",
                    next_run=record.get("next_run_at") or "-",
                )
            )
    else:
        lines.append("- No scheduled jobs registered.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_weekly_report(
    *,
    settings: OpsSettings,
    output_dir: str | Path | None = None,
    lookback_days: int = 7,
    actor: str = "weekly_report",
) -> dict[str, Any]:
    settings.ensure_directories()
    report_root = Path(output_dir) if output_dir is not None else settings.reports_root
    report_root.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_name = f"{timestamp}_weekly_report"
    report_json_path = report_root / f"{report_name}.json"
    report_markdown_path = report_root / f"{report_name}.md"

    with OpsDatabase(settings.db_path) as database:
        database.initialize()
        all_runs = database.list_runs(limit=5000)
        recent_runs = [run for run in all_runs if _in_lookback(run, cutoff)]
        recent_jobs = [
            job for job in database.list_job_runs(limit=500) if _in_lookback(job, cutoff)
        ]
        recent_audit = [
            log for log in database.list_audit_logs(limit=500) if _in_lookback(log, cutoff)
        ]
        runs_by_problem = Counter(str(run.get("problem") or "unknown") for run in recent_runs)
        payload = {
            "report_name": report_name,
            "generated_at": datetime.now(UTC).isoformat(),
            "lookback_days": lookback_days,
            "totals": {
                "runs": len(recent_runs),
                "job_runs": len(recent_jobs),
                "audit_events": len(recent_audit),
            },
            "runs_by_problem": dict(sorted(runs_by_problem.items())),
            "recent_regressions": database.recent_regressions(limit=10),
            "best_configs": database.best_configs(limit=10),
            "job_status": database.list_scheduled_jobs(),
            "recent_job_runs": recent_jobs[:10],
            "recent_audit_logs": recent_audit[:20],
            "recent_runs": recent_runs[:20],
        }
        report_json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_markdown(report_markdown_path, payload)
        database.log_audit(
            actor=actor,
            action="generate_weekly_report",
            resource_type="report",
            resource_id=str(report_json_path),
            status="success",
            details={
                "report_json": str(report_json_path),
                "report_markdown": str(report_markdown_path),
                "lookback_days": lookback_days,
            },
        )
        return {
            "payload": payload,
            "report_json": str(report_json_path),
            "report_markdown": str(report_markdown_path),
        }
