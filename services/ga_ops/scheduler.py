from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .db import OpsDatabase
from .settings import OpsSettings


@dataclass(slots=True)
class ScheduledJobDefinition:
    name: str
    job_type: str
    cron: str
    enabled: bool
    config: dict[str, Any]


def load_job_definitions(path: str | Path) -> list[ScheduledJobDefinition]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise ValueError("Scheduler config must be an object with a jobs array")
    jobs = []
    for item in payload["jobs"]:
        if not isinstance(item, dict):
            raise ValueError("Each scheduler job must be an object")
        jobs.append(
            ScheduledJobDefinition(
                name=str(item["name"]),
                job_type=str(item["job_type"]),
                cron=str(item["cron"]),
                enabled=bool(item.get("enabled", True)),
                config=dict(item.get("config", {})),
            )
        )
    return jobs


def _as_path(value: Any, *, fallback: Path) -> Path:
    if isinstance(value, str) and value.strip():
        return Path(value)
    return fallback


def build_job_command(job: ScheduledJobDefinition, *, settings: OpsSettings) -> list[str]:
    python = sys.executable
    if job.job_type == "nightly_benchmark":
        output_root = _as_path(
            job.config.get("output_root"),
            fallback=settings.project_root / "outputs" / "nightly",
        )
        return [
            python,
            "scripts/run_nightly.py",
            "--output-root",
            str(output_root),
            "--sync-ops",
            "--ops-actor",
            job.name,
        ]
    if job.job_type == "weekly_report":
        output_dir = _as_path(job.config.get("output_dir"), fallback=settings.reports_root)
        lookback_days = int(job.config.get("lookback_days", 7))
        return [
            python,
            "scripts/ops_generate_weekly_report.py",
            "--db-path",
            str(settings.db_path),
            "--output-dir",
            str(output_dir),
            "--lookback-days",
            str(lookback_days),
            "--actor",
            job.name,
        ]
    if job.job_type == "release_candidate_regression":
        manifest = _as_path(
            job.config.get("manifest"),
            fallback=(
                settings.project_root / "configs" / "release_candidate" / "regression_manifest.json"
            ),
        )
        output_root = _as_path(
            job.config.get("output_root"),
            fallback=settings.project_root / "outputs" / "release_candidate",
        )
        return [
            python,
            "scripts/run_release_candidate_regression.py",
            "--manifest",
            str(manifest),
            "--output-root",
            str(output_root),
            "--db-path",
            str(settings.db_path),
            "--actor",
            job.name,
        ]
    raise ValueError(f"Unsupported scheduler job_type: {job.job_type}")


def register_job_definitions(
    *,
    settings: OpsSettings,
    jobs: list[ScheduledJobDefinition],
) -> None:
    with OpsDatabase(settings.db_path) as database:
        database.initialize()
        for job in jobs:
            database.upsert_scheduled_job(
                name=job.name,
                job_type=job.job_type,
                cron=job.cron,
                enabled=job.enabled,
                config=job.config,
            )


def run_job(
    job: ScheduledJobDefinition,
    *,
    settings: OpsSettings,
    triggered_by: str = "scheduler",
) -> dict[str, Any]:
    settings.ensure_directories()
    logs_root = settings.logs_root / "scheduler"
    logs_root.mkdir(parents=True, exist_ok=True)
    command = build_job_command(job, settings=settings)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_root / f"{timestamp}_{job.name}.log"

    with OpsDatabase(settings.db_path) as database:
        database.initialize()
        job_run_id = database.record_job_start(
            job_name=job.name,
            job_type=job.job_type,
            command=subprocess.list2cmdline(command),
            triggered_by=triggered_by,
        )

    completed = subprocess.run(
        command,
        cwd=settings.project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(
        "\n".join(
            [
                f"Command: {subprocess.list2cmdline(command)}",
                f"Return code: {completed.returncode}",
                "",
                "STDOUT",
                completed.stdout.rstrip(),
                "",
                "STDERR",
                completed.stderr.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    with OpsDatabase(settings.db_path) as database:
        database.initialize()
        database.record_job_finish(
            job_run_id,
            status="passed" if completed.returncode == 0 else "failed",
            log_path=str(log_path),
            error=completed.stderr.strip() or None,
        )
        database.update_schedule_state(
            name=job.name,
            last_run_at=datetime.utcnow().isoformat() + "Z",
        )
        database.log_audit(
            actor=triggered_by,
            action="run_scheduled_job",
            resource_type="scheduled_job",
            resource_id=job.name,
            status="success" if completed.returncode == 0 else "failed",
            details={
                "job_type": job.job_type,
                "log_path": str(log_path),
                "returncode": completed.returncode,
            },
        )

    return {
        "job_name": job.name,
        "job_type": job.job_type,
        "returncode": completed.returncode,
        "log_path": str(log_path),
    }
