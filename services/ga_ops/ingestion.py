from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ga_lab.experiment.comparison import collect_comparison_rows

from .db import OpsDatabase, utc_now
from .settings import OpsSettings
from .storage import build_object_store


def _parse_run_timestamp(output_dir: Path) -> str | None:
    timestamp = output_dir.name.split("_", 1)[0]
    try:
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return parsed.isoformat()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return None


def _config_hash(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sync_results_dir(
    results_dir: str | Path,
    *,
    settings: OpsSettings,
    actor: str = "system",
    upload_artifacts: bool = True,
    source_collection: str | None = None,
) -> dict[str, Any]:
    root = Path(results_dir)
    settings.ensure_directories()
    object_store = build_object_store(settings)
    object_store.ensure_ready()

    with OpsDatabase(settings.db_path) as database:
        database.initialize()
        rows = collect_comparison_rows(root)
        synced_runs = 0
        synced_artifacts = 0

        for row in rows:
            output_dir = Path(str(row["output_dir"]))
            summary_path = output_dir / "summary.json"
            history_path = output_dir / "history.csv"
            config_path = output_dir / "config.json"
            config_canonical_path = output_dir / "config.canonical.json"
            summary_payload = _load_json(summary_path) or {}
            config_payload = _load_json(config_path) or {}
            canonical_payload = _load_json(config_canonical_path) or config_payload
            run_record = {
                **row,
                "run_key": str(summary_path.resolve()),
                "source_collection": source_collection or row.get("suite_name") or root.name,
                "source_config_path": row.get("config_path"),
                "config_path": str(config_path),
                "config_canonical_path": str(config_canonical_path),
                "summary_path": str(summary_path),
                "history_path": str(history_path),
                "output_dir": str(output_dir),
                "config_hash": _config_hash(canonical_payload),
                "config_payload": json.dumps(canonical_payload, ensure_ascii=False, sort_keys=True),
                "summary_payload": json.dumps(summary_payload, ensure_ascii=False, sort_keys=True),
                "run_created_at": _parse_run_timestamp(output_dir)
                or datetime.fromtimestamp(summary_path.stat().st_mtime, UTC).isoformat(),
                "ingested_at": utc_now(),
            }
            run_id = database.upsert_run(run_record)
            synced_runs += 1

            if not upload_artifacts:
                continue
            for artifact_type, artifact_path in (
                ("config", config_path),
                ("canonical_config", config_canonical_path),
                ("summary", summary_path),
                ("history", history_path),
            ):
                if not artifact_path.exists():
                    continue
                object_key = f"runs/{output_dir.name}/{artifact_path.name}"
                stored = object_store.upload_file(artifact_path, object_key)
                database.record_artifact(
                    run_id,
                    {
                        "artifact_type": artifact_type,
                        "object_key": stored.object_key,
                        "provider": stored.provider,
                        "uri": stored.uri,
                        "source_path": str(artifact_path),
                        "size_bytes": stored.size_bytes,
                        "checksum_sha256": stored.checksum_sha256,
                    },
                )
                synced_artifacts += 1

        database.log_audit(
            actor=actor,
            action="sync_results",
            resource_type="results_dir",
            resource_id=str(root),
            status="success",
            details={
                "results_dir": str(root),
                "source_collection": source_collection or root.name,
                "runs": synced_runs,
                "artifacts": synced_artifacts,
                "upload_artifacts": upload_artifacts,
            },
        )
        return {
            "results_dir": str(root),
            "source_collection": source_collection or root.name,
            "runs": synced_runs,
            "artifacts": synced_artifacts,
        }
