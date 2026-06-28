from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.runner import run_experiment
from ga_ops.db import OpsDatabase
from ga_ops.ingestion import sync_results_dir
from ga_ops.reporting import generate_weekly_report
from ga_ops.scheduler import (
    ScheduledJobDefinition,
    load_job_definitions,
    register_job_definitions,
)
from ga_ops.settings import OpsSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _settings(tmp_path: Path) -> OpsSettings:
    return OpsSettings(
        project_root=PROJECT_ROOT,
        db_path=tmp_path / "var" / "ga_ops.db",
        object_store_provider="local",
        object_store_root=tmp_path / "var" / "object_store",
        object_store_bucket="ga-lab-artifacts",
        s3_endpoint_url=None,
        s3_region=None,
        s3_access_key_id=None,
        s3_secret_access_key=None,
        scheduler_jobs_path=PROJECT_ROOT / "configs" / "schedules" / "default_jobs.json",
        scheduler_timezone="Asia/Seoul",
        logs_root=tmp_path / "var" / "logs",
        reports_root=tmp_path / "reports",
        dashboard_results_dir=tmp_path / "outputs",
    )


def _config(**overrides: object) -> GAConfig:
    payload = {
        "run_name": "ops_onemax",
        "problem": "onemax",
        "population_size": 24,
        "genome_length": 16,
        "generations": 20,
        "crossover_rate": 0.9,
        "mutation_rate": 0.02,
        "elitism": 1,
        "tournament_size": 3,
        "seed": 7,
        "maximize": True,
        "target_fitness": 16,
        "log_every": 1,
    }
    payload.update(overrides)
    return GAConfig(**payload)


def test_sync_results_dir_populates_metadata_and_artifacts(tmp_path) -> None:
    settings = _settings(tmp_path)
    outputs_dir = tmp_path / "outputs"
    result = run_experiment(_config(), output_root=outputs_dir)

    sync_summary = sync_results_dir(outputs_dir, settings=settings, actor="pytest")

    assert sync_summary["runs"] == 1
    with OpsDatabase(settings.db_path) as database:
        database.initialize()
        runs = database.list_runs(limit=10)
        assert len(runs) == 1
        assert runs[0]["run_name"] == result.summary["run_name"]
        artifacts = database.list_run_artifacts(int(runs[0]["id"]))
        assert {artifact["artifact_type"] for artifact in artifacts} == {
            "canonical_config",
            "config",
            "history",
            "summary",
        }
    assert (settings.object_store_root / "runs" / result.output_dir.name / "summary.json").exists()


def test_generate_weekly_report_uses_metadata_db(tmp_path) -> None:
    settings = _settings(tmp_path)
    outputs_dir = tmp_path / "outputs"
    run_experiment(_config(), output_root=outputs_dir)
    sync_results_dir(outputs_dir, settings=settings, actor="pytest")

    report = generate_weekly_report(
        settings=settings,
        output_dir=tmp_path / "reports",
        lookback_days=30,
        actor="pytest",
    )

    assert Path(report["report_json"]).exists()
    assert Path(report["report_markdown"]).exists()
    assert report["payload"]["totals"]["runs"] == 1


def test_ops_database_auth_scheduler_and_regression_detection(tmp_path) -> None:
    settings = _settings(tmp_path)
    with OpsDatabase(settings.db_path) as database:
        database.initialize()
        token = database.create_token("reader", scopes=["ops.read"])
        actor = database.authenticate_token(token)
        assert actor is not None
        assert actor.name == "reader"
        assert actor.scopes == {"ops.read"}

        register_job_definitions(
            settings=settings,
            jobs=[
                ScheduledJobDefinition(
                    name="nightly_benchmark",
                    job_type="nightly_benchmark",
                    cron="0 2 * * *",
                    enabled=True,
                    config={"output_root": "outputs/nightly"},
                )
            ],
        )
        scheduled_jobs = database.list_scheduled_jobs()
        assert scheduled_jobs[0]["name"] == "nightly_benchmark"

        common = {
            "suite_name": "nightly",
            "baseline_label": "onemax_smoke",
            "source_collection": "nightly",
            "problem": "onemax",
            "algorithm": "ga",
            "representation": "bit",
            "selection": "tournament",
            "crossover": "one_point",
            "mutation": "bit_flip",
            "population_size": 80,
            "genome_length": 64,
            "generations": 80,
            "crossover_rate": 0.9,
            "mutation_rate": 0.02,
            "elitism": 2,
            "tournament_size": 3,
            "maximize": True,
            "target_fitness": 64,
            "config_path": "config.json",
            "config_canonical_path": "config.canonical.json",
            "source_config_path": "generated.json",
            "history_path": "history.csv",
            "config_hash": "hash",
            "config_payload": json.dumps({"run_name": "a"}),
            "summary_payload": json.dumps({"best_fitness": 64}),
            "ingested_at": "2026-03-08T00:00:00+00:00",
        }
        database.upsert_run(
            {
                **common,
                "run_key": "previous",
                "run_name": "onemax_prev",
                "seed": 7,
                "best_fitness": 64.0,
                "mean_fitness": 63.0,
                "worst_fitness": 60.0,
                "stop_reason": "target_fitness_reached",
                "convergence_generation": 29,
                "final_generation": 29,
                "runtime_seconds": 1.0,
                "objective_count": 1,
                "pareto_front_size": None,
                "pareto_ratio": None,
                "hypervolume": None,
                "spread": None,
                "output_dir": "prev",
                "summary_path": "prev/summary.json",
                "run_created_at": "2026-03-07T00:00:00+00:00",
            }
        )
        database.upsert_run(
            {
                **common,
                "run_key": "latest",
                "run_name": "onemax_latest",
                "seed": 8,
                "best_fitness": 60.0,
                "mean_fitness": 59.0,
                "worst_fitness": 55.0,
                "stop_reason": "max_generations",
                "convergence_generation": None,
                "final_generation": 80,
                "runtime_seconds": 1.4,
                "objective_count": 1,
                "pareto_front_size": None,
                "pareto_ratio": None,
                "hypervolume": None,
                "spread": None,
                "output_dir": "latest",
                "summary_path": "latest/summary.json",
                "run_created_at": "2026-03-08T00:00:00+00:00",
            }
        )
        regressions = database.recent_regressions(limit=5)
        assert regressions
        assert set(regressions[0]["reasons"]) == {
            "best_fitness",
            "runtime_seconds",
            "stop_reason",
        }


def test_run_baselines_script_can_sync_ops(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "ops_sync_suite",
                "entries": [
                    {
                        "label": "onemax_sync",
                        "config": "configs/onemax_baseline.json",
                        "seeds": 1,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "ops.db"
    object_store_root = tmp_path / "object_store"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_baselines.py"),
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "suite_outputs"),
            "--sync-ops",
            "--ops-db-path",
            str(db_path),
            "--object-store-root",
            str(object_store_root),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with OpsDatabase(db_path) as database:
        database.initialize()
        runs = database.list_runs(limit=10)
        assert len(runs) == 1
        assert runs[0]["suite_name"] == "ops_sync_suite"
        artifacts = database.list_run_artifacts(int(runs[0]["id"]))
        assert len(artifacts) == 4


def test_load_job_definitions_accepts_utf8_bom(tmp_path) -> None:
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "weekly_report",
                        "job_type": "weekly_report",
                        "cron": "* * * * *",
                        "enabled": True,
                        "config": {"output_dir": "outputs/reports"},
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8-sig",
    )

    jobs = load_job_definitions(jobs_path)

    assert len(jobs) == 1
    assert jobs[0].name == "weekly_report"


def test_settings_loads_ops_env_file_relative_paths(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env.ops"
    env_path.write_text(
        "\n".join(
            [
                "GA_LAB_OPS_DB_PATH=var/ops/custom.db",
                "GA_LAB_LOGS_ROOT=var/ops/custom-logs",
                "GA_LAB_SCHEDULER_JOBS_PATH=configs/schedules/default_jobs.json",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "schedules").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "schedules" / "default_jobs.json").write_text(
        '{"jobs": []}',
        encoding="utf-8",
    )
    for name in (
        "GA_LAB_OPS_DB_PATH",
        "GA_LAB_LOGS_ROOT",
        "GA_LAB_SCHEDULER_JOBS_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = OpsSettings.from_env(project_root=tmp_path)

    assert settings.db_path == tmp_path / "var" / "ops" / "custom.db"
    assert settings.logs_root == tmp_path / "var" / "ops" / "custom-logs"
    assert settings.scheduler_jobs_path == tmp_path / "configs" / "schedules" / "default_jobs.json"
