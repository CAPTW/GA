from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_json_loads(value: str | None) -> Any:
    if not value:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(slots=True)
class AuthenticatedActor:
    name: str
    scopes: set[str]


class OpsDatabase:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> OpsDatabase:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def initialize(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_key TEXT NOT NULL UNIQUE,
              suite_name TEXT,
              baseline_label TEXT,
              source_collection TEXT,
              run_name TEXT NOT NULL,
              problem TEXT,
              seed INTEGER,
              algorithm TEXT,
              representation TEXT,
              selection TEXT,
              crossover TEXT,
              mutation TEXT,
              population_size INTEGER,
              genome_length INTEGER,
              generations INTEGER,
              crossover_rate REAL,
              mutation_rate REAL,
              elitism INTEGER,
              tournament_size INTEGER,
              maximize INTEGER,
              target_fitness REAL,
              best_fitness REAL,
              mean_fitness REAL,
              worst_fitness REAL,
              stop_reason TEXT,
              convergence_generation INTEGER,
              final_generation INTEGER,
              runtime_seconds REAL,
              objective_count INTEGER,
              pareto_front_size INTEGER,
              pareto_ratio REAL,
              hypervolume REAL,
              spread REAL,
              config_path TEXT,
              config_canonical_path TEXT,
              source_config_path TEXT,
              output_dir TEXT,
              summary_path TEXT,
              history_path TEXT,
              config_hash TEXT,
              config_payload TEXT,
              summary_payload TEXT,
              run_created_at TEXT,
              ingested_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS artifacts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id INTEGER NOT NULL,
              artifact_type TEXT NOT NULL,
              object_key TEXT NOT NULL,
              provider TEXT NOT NULL,
              uri TEXT NOT NULL,
              source_path TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              checksum_sha256 TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(run_id, artifact_type, object_key),
              FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              actor TEXT NOT NULL,
              action TEXT NOT NULL,
              resource_type TEXT NOT NULL,
              resource_id TEXT,
              status TEXT NOT NULL,
              details_json TEXT,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_tokens (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL UNIQUE,
              token_hash TEXT NOT NULL UNIQUE,
              scopes TEXT NOT NULL,
              active INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scheduled_jobs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL UNIQUE,
              job_type TEXT NOT NULL,
              cron TEXT NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1,
              config_json TEXT,
              last_run_at TEXT,
              next_run_at TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS job_runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_name TEXT NOT NULL,
              job_type TEXT NOT NULL,
              status TEXT NOT NULL,
              command TEXT NOT NULL,
              triggered_by TEXT NOT NULL,
              output_dir TEXT,
              report_path TEXT,
              log_path TEXT,
              error TEXT,
              started_at TEXT NOT NULL,
              finished_at TEXT
            );
            """
        )
        self._conn.commit()

    def _execute(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        cursor = self._conn.execute(query, params)
        self._conn.commit()
        return cursor

    def _fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(query, params))

    def _fetchone(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self._conn.execute(query, params).fetchone()

    def log_audit(
        self,
        *,
        actor: str,
        action: str,
        resource_type: str,
        status: str,
        resource_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO audit_logs (
              actor,
              action,
              resource_type,
              resource_id,
              status,
              details_json,
              created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actor,
                action,
                resource_type,
                resource_id,
                status,
                _json_dumps(details) if details is not None else None,
                utc_now(),
            ),
        )

    def list_audit_logs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM audit_logs ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(row) for row in rows]

    def create_token(self, name: str, *, scopes: list[str], raw_token: str | None = None) -> str:
        token = raw_token or secrets.token_urlsafe(24)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        self._execute(
            """
            INSERT INTO api_tokens (name, token_hash, scopes, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              token_hash=excluded.token_hash,
              scopes=excluded.scopes,
              active=1,
              created_at=excluded.created_at
            """,
            (name, token_hash, ",".join(sorted(set(scopes))), utc_now()),
        )
        return token

    def authenticate_token(self, token: str | None) -> AuthenticatedActor | None:
        admin_token = os.getenv("GA_LAB_OPS_ADMIN_TOKEN")
        if admin_token and token == admin_token:
            return AuthenticatedActor(
                name="admin",
                scopes={"ops.read", "ops.write", "codex.invoke"},
            )
        if not token:
            return None
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        row = self._fetchone(
            "SELECT name, scopes FROM api_tokens WHERE token_hash = ? AND active = 1",
            (token_hash,),
        )
        if row is None:
            return None
        scopes = {scope.strip() for scope in str(row["scopes"]).split(",") if scope.strip()}
        return AuthenticatedActor(name=str(row["name"]), scopes=scopes)

    def upsert_run(self, record: dict[str, Any]) -> int:
        now = utc_now()
        payload = {
            "run_key": record["run_key"],
            "suite_name": record.get("suite_name"),
            "baseline_label": record.get("baseline_label"),
            "source_collection": record.get("source_collection"),
            "run_name": record["run_name"],
            "problem": record.get("problem"),
            "seed": record.get("seed"),
            "algorithm": record.get("algorithm"),
            "representation": record.get("representation"),
            "selection": record.get("selection"),
            "crossover": record.get("crossover"),
            "mutation": record.get("mutation"),
            "population_size": record.get("population_size"),
            "genome_length": record.get("genome_length"),
            "generations": record.get("generations"),
            "crossover_rate": record.get("crossover_rate"),
            "mutation_rate": record.get("mutation_rate"),
            "elitism": record.get("elitism"),
            "tournament_size": record.get("tournament_size"),
            "maximize": int(bool(record.get("maximize", True))),
            "target_fitness": record.get("target_fitness"),
            "best_fitness": record.get("best_fitness"),
            "mean_fitness": record.get("mean_fitness"),
            "worst_fitness": record.get("worst_fitness"),
            "stop_reason": record.get("stop_reason"),
            "convergence_generation": record.get("convergence_generation"),
            "final_generation": record.get("final_generation"),
            "runtime_seconds": record.get("runtime_seconds"),
            "objective_count": record.get("objective_count"),
            "pareto_front_size": record.get("pareto_front_size"),
            "pareto_ratio": record.get("pareto_ratio"),
            "hypervolume": record.get("hypervolume"),
            "spread": record.get("spread"),
            "config_path": record.get("config_path"),
            "config_canonical_path": record.get("config_canonical_path"),
            "source_config_path": record.get("source_config_path"),
            "output_dir": record.get("output_dir"),
            "summary_path": record.get("summary_path"),
            "history_path": record.get("history_path"),
            "config_hash": record.get("config_hash"),
            "config_payload": record.get("config_payload"),
            "summary_payload": record.get("summary_payload"),
            "run_created_at": record.get("run_created_at"),
            "ingested_at": record.get("ingested_at", now),
            "updated_at": now,
        }
        columns = ", ".join(payload.keys())
        placeholders = ", ".join("?" for _ in payload)
        updates = ", ".join(
            f"{column}=excluded.{column}"
            for column in payload
            if column not in {"run_key", "ingested_at"}
        )
        self._execute(
            f"""
            INSERT INTO runs ({columns})
            VALUES ({placeholders})
            ON CONFLICT(run_key) DO UPDATE SET
              {updates},
              updated_at=excluded.updated_at
            """,
            tuple(payload.values()),
        )
        row = self._fetchone("SELECT id FROM runs WHERE run_key = ?", (record["run_key"],))
        if row is None:
            raise RuntimeError("Failed to upsert run record")
        return int(row["id"])

    def record_artifact(self, run_id: int, artifact: dict[str, Any]) -> None:
        self._execute(
            """
            INSERT INTO artifacts (
              run_id,
              artifact_type,
              object_key,
              provider,
              uri,
              source_path,
              size_bytes,
              checksum_sha256,
              created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, artifact_type, object_key) DO UPDATE SET
              provider=excluded.provider,
              uri=excluded.uri,
              source_path=excluded.source_path,
              size_bytes=excluded.size_bytes,
              checksum_sha256=excluded.checksum_sha256,
              created_at=excluded.created_at
            """,
            (
                run_id,
                artifact["artifact_type"],
                artifact["object_key"],
                artifact["provider"],
                artifact["uri"],
                artifact["source_path"],
                artifact["size_bytes"],
                artifact["checksum_sha256"],
                utc_now(),
            ),
        )

    def list_runs(
        self,
        *,
        limit: int = 100,
        suite_name: str | None = None,
        problem: str | None = None,
        baseline_label: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if suite_name:
            clauses.append("suite_name = ?")
            params.append(suite_name)
        if problem:
            clauses.append("problem = ?")
            params.append(problem)
        if baseline_label:
            clauses.append("baseline_label = ?")
            params.append(baseline_label)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._fetchall(
            f"""
            SELECT * FROM runs
            {where}
            ORDER BY COALESCE(run_created_at, ingested_at) DESC, id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return [self._row_to_dict(row) for row in rows]

    def list_run_artifacts(self, run_id: int) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY artifact_type, object_key",
            (run_id,),
        )
        return [self._row_to_dict(row) for row in rows]

    def upsert_scheduled_job(
        self,
        *,
        name: str,
        job_type: str,
        cron: str,
        enabled: bool,
        config: dict[str, Any],
        next_run_at: str | None = None,
    ) -> None:
        now = utc_now()
        self._execute(
            """
            INSERT INTO scheduled_jobs (
              name, job_type, cron, enabled, config_json, next_run_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              job_type=excluded.job_type,
              cron=excluded.cron,
              enabled=excluded.enabled,
              config_json=excluded.config_json,
              next_run_at=excluded.next_run_at,
              updated_at=excluded.updated_at
            """,
            (
                name,
                job_type,
                cron,
                int(enabled),
                _json_dumps(config),
                next_run_at,
                now,
                now,
            ),
        )

    def update_schedule_state(
        self,
        *,
        name: str,
        last_run_at: str | None = None,
        next_run_at: str | None = None,
    ) -> None:
        self._execute(
            """
            UPDATE scheduled_jobs
            SET last_run_at = COALESCE(?, last_run_at),
                next_run_at = COALESCE(?, next_run_at),
                updated_at = ?
            WHERE name = ?
            """,
            (last_run_at, next_run_at, utc_now(), name),
        )

    def list_scheduled_jobs(self) -> list[dict[str, Any]]:
        rows = self._fetchall("SELECT * FROM scheduled_jobs ORDER BY enabled DESC, name ASC")
        return [self._row_to_dict(row) for row in rows]

    def record_job_start(
        self,
        *,
        job_name: str,
        job_type: str,
        command: str,
        triggered_by: str,
        status: str = "running",
    ) -> int:
        cursor = self._execute(
            """
            INSERT INTO job_runs (job_name, job_type, status, command, triggered_by, started_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_name, job_type, status, command, triggered_by, utc_now()),
        )
        return int(cursor.lastrowid)

    def record_job_finish(
        self,
        job_run_id: int,
        *,
        status: str,
        output_dir: str | None = None,
        report_path: str | None = None,
        log_path: str | None = None,
        error: str | None = None,
    ) -> None:
        self._execute(
            """
            UPDATE job_runs
            SET status = ?,
                output_dir = ?,
                report_path = ?,
                log_path = ?,
                error = ?,
                finished_at = ?
            WHERE id = ?
            """,
            (status, output_dir, report_path, log_path, error, utc_now(), job_run_id),
        )

    def list_job_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM job_runs ORDER BY started_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(row) for row in rows]

    def recent_regressions(self, *, limit: int = 10) -> list[dict[str, Any]]:
        runs = self.list_runs(limit=1000)
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            label = str(run.get("baseline_label") or run.get("run_name") or "unlabeled")
            signature = (
                label,
                str(run.get("problem") or "-"),
                str(run.get("algorithm") or "-"),
                str(run.get("representation") or "-"),
            )
            grouped[signature].append(run)

        regressions: list[dict[str, Any]] = []
        for signature, group_runs in grouped.items():
            sorted_runs = sorted(
                group_runs,
                key=lambda row: (
                    _parse_iso_timestamp(row.get("run_created_at"))
                    or _parse_iso_timestamp(row.get("ingested_at"))
                    or datetime.min.replace(tzinfo=UTC),
                    int(row.get("id", 0)),
                ),
                reverse=True,
            )
            if len(sorted_runs) < 2:
                continue
            latest, previous = sorted_runs[0], sorted_runs[1]
            maximize = bool(latest.get("maximize", True))
            reasons: list[str] = []
            best_latest = latest.get("best_fitness")
            best_previous = previous.get("best_fitness")
            if isinstance(best_latest, (int, float)) and isinstance(best_previous, (int, float)):
                worsened = best_latest < best_previous if maximize else best_latest > best_previous
                if worsened:
                    reasons.append("best_fitness")
            latest_stop = latest.get("stop_reason")
            previous_stop = previous.get("stop_reason")
            if previous_stop == "target_fitness_reached" and latest_stop != previous_stop:
                reasons.append("stop_reason")
            latest_runtime = latest.get("runtime_seconds")
            previous_runtime = previous.get("runtime_seconds")
            if (
                isinstance(latest_runtime, (int, float))
                and isinstance(previous_runtime, (int, float))
                and previous_runtime > 0
                and latest_runtime > previous_runtime * 1.10
            ):
                reasons.append("runtime_seconds")
            if reasons:
                regressions.append(
                    {
                        "label": signature[0],
                        "problem": signature[1],
                        "latest_run_name": latest.get("run_name"),
                        "previous_run_name": previous.get("run_name"),
                        "latest_best_fitness": best_latest,
                        "previous_best_fitness": best_previous,
                        "latest_runtime_seconds": latest_runtime,
                        "previous_runtime_seconds": previous_runtime,
                        "latest_stop_reason": latest_stop,
                        "previous_stop_reason": previous_stop,
                        "reasons": reasons,
                    }
                )
        return regressions[:limit]

    def best_configs(self, *, limit: int = 10) -> list[dict[str, Any]]:
        runs = self.list_runs(limit=5000)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            key = str(
                run.get("config_hash") or run.get("source_config_path") or run.get("config_path")
            )
            if key and key != "None":
                grouped[key].append(run)

        records: list[dict[str, Any]] = []
        for key, group_runs in grouped.items():
            best_values = [
                float(run["best_fitness"])
                for run in group_runs
                if isinstance(run.get("best_fitness"), (int, float))
            ]
            if not best_values:
                continue
            runtime_values = [
                float(run["runtime_seconds"])
                for run in group_runs
                if isinstance(run.get("runtime_seconds"), (int, float))
            ]
            sample = group_runs[0]
            maximize = bool(sample.get("maximize", True))
            payload = _safe_json_loads(sample.get("config_payload"))
            mean_best = mean(best_values)
            ranking_score = mean_best if maximize else -mean_best
            records.append(
                {
                    "config_hash": key,
                    "problem": sample.get("problem"),
                    "suite_name": sample.get("suite_name"),
                    "baseline_label": sample.get("baseline_label"),
                    "runs": len(group_runs),
                    "mean_best_fitness": mean_best,
                    "mean_runtime_seconds": mean(runtime_values) if runtime_values else None,
                    "selection": sample.get("selection"),
                    "crossover": sample.get("crossover"),
                    "mutation": sample.get("mutation"),
                    "source_config_path": sample.get("source_config_path"),
                    "config_payload": payload,
                    "ranking_score": ranking_score,
                }
            )
        return sorted(
            records,
            key=lambda record: (-record["ranking_score"], str(record["config_hash"])),
        )[:limit]

    def dashboard_summary(self) -> dict[str, Any]:
        total_runs = self._fetchone("SELECT COUNT(*) AS count FROM runs")
        total_artifacts = self._fetchone("SELECT COUNT(*) AS count FROM artifacts")
        total_jobs = self._fetchone("SELECT COUNT(*) AS count FROM scheduled_jobs")
        return {
            "generated_at": utc_now(),
            "totals": {
                "runs": int(total_runs["count"]) if total_runs else 0,
                "artifacts": int(total_artifacts["count"]) if total_artifacts else 0,
                "scheduled_jobs": int(total_jobs["count"]) if total_jobs else 0,
            },
            "recent_runs": self.list_runs(limit=20),
            "recent_regressions": self.recent_regressions(limit=10),
            "best_configs": self.best_configs(limit=10),
            "scheduled_jobs": self.list_scheduled_jobs(),
            "recent_job_runs": self.list_job_runs(limit=10),
            "recent_audit_logs": self.list_audit_logs(limit=20),
        }

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        for key in ("maximize", "active", "enabled"):
            if key in payload and payload[key] is not None:
                payload[key] = bool(payload[key])
        for key in ("config_payload", "summary_payload", "details_json", "config_json"):
            if key in payload:
                payload[key] = _safe_json_loads(payload[key])
        return payload
