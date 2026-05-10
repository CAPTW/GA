# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from ga_ops.db import OpsDatabase
from ga_ops.scheduler import (
    ScheduledJobDefinition,
    load_job_definitions,
    register_job_definitions,
    run_job,
)
from ga_ops.settings import OpsSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or host the GA Lab ops scheduler.")
    parser.add_argument(
        "--jobs-config",
        default=None,
        help="Scheduler job config JSON path. Defaults to GA_LAB_SCHEDULER_JOBS_PATH.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute enabled jobs once and exit instead of running a persistent scheduler",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Optional single job name to execute when --once is used",
    )
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="Register jobs in the metadata DB and exit",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional override for the ops SQLite database path",
    )
    return parser.parse_args()


def _load_timezone(name: str):
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _register_jobs(settings: OpsSettings, jobs: list[ScheduledJobDefinition]) -> None:
    timezone = _load_timezone(settings.scheduler_timezone)
    register_job_definitions(settings=settings, jobs=jobs)
    try:
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        return

    now = datetime.now(timezone)
    with OpsDatabase(settings.db_path) as database:
        database.initialize()
        for job in jobs:
            trigger = CronTrigger.from_crontab(job.cron, timezone=timezone)
            next_run = trigger.get_next_fire_time(None, now)
            database.upsert_scheduled_job(
                name=job.name,
                job_type=job.job_type,
                cron=job.cron,
                enabled=job.enabled,
                config=job.config,
                next_run_at=next_run.isoformat() if next_run is not None else None,
            )


def main() -> None:
    args = parse_args()
    settings = OpsSettings.from_env(project_root=PROJECT_ROOT)
    if args.db_path is not None:
        settings.db_path = Path(args.db_path)
    jobs_path = (
        Path(args.jobs_config) if args.jobs_config is not None else settings.scheduler_jobs_path
    )
    jobs = load_job_definitions(jobs_path)
    _register_jobs(settings, jobs)
    if args.register_only:
        print(json.dumps({"registered_jobs": [job.name for job in jobs]}, indent=2))
        return

    enabled_jobs = [job for job in jobs if job.enabled]
    if args.job_name is not None:
        enabled_jobs = [job for job in enabled_jobs if job.name == args.job_name]
    if args.once:
        results = [
            run_job(job, settings=settings, triggered_by="scheduler") for job in enabled_jobs
        ]
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError as exc:
        raise SystemExit(
            "Persistent scheduling requires the optional 'apscheduler' dependency. "
            "Use --once otherwise."
        ) from exc

    timezone = _load_timezone(settings.scheduler_timezone)
    scheduler = BlockingScheduler(timezone=timezone)
    for job in enabled_jobs:
        scheduler.add_job(
            run_job,
            trigger=CronTrigger.from_crontab(job.cron, timezone=timezone),
            args=[job],
            kwargs={"settings": settings, "triggered_by": "scheduler"},
            id=job.name,
            replace_existing=True,
        )
    scheduler.start()


if __name__ == "__main__":
    main()
