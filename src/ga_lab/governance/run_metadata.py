from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

RUN_METADATA_SCHEMA_VERSION = 1
TRACKED_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITHUB_SHA",
    "GITHUB_REF",
    "GITHUB_REF_NAME",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_WORKFLOW",
    "RUNNER_OS",
)


def _sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return _sha256_bytes(path.read_bytes())


def _json_peek(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    snapshot: dict[str, Any] = {}
    for key in ("suite_name", "summary_version", "summary_schema_version", "manifest_version"):
        if key in payload:
            snapshot[key] = payload[key]
    if "entries" in payload and isinstance(payload["entries"], list):
        snapshot["entry_count"] = len(payload["entries"])
    return snapshot


def manifest_snapshots(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        snapshot: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
        }
        if path.exists() and path.is_file():
            snapshot.update(
                {
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
            snapshot.update(_json_peek(path))
        snapshots.append(snapshot)
    return snapshots


def _run_git(project_root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_snapshot(project_root: Path) -> dict[str, Any]:
    sha = _run_git(project_root, "rev-parse", "HEAD")
    if sha is None:
        return {
            "available": False,
            "sha": None,
            "branch": None,
            "status_short": None,
        }
    return {
        "available": True,
        "sha": sha,
        "branch": _run_git(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": _run_git(project_root, "status", "--short"),
    }


def _package_snapshot() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        packages[name] = distribution.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def benchmark_metadata_snapshot(project_root: Path) -> dict[str, Any]:
    metadata_path = project_root / "benchmarks" / "metadata.json"
    snapshot: dict[str, Any] = {
        "path": str(metadata_path),
        "exists": metadata_path.exists(),
    }
    if metadata_path.exists() and metadata_path.is_file():
        snapshot.update(
            {
                "sha256": file_sha256(metadata_path),
                "size_bytes": metadata_path.stat().st_size,
            }
        )
        snapshot.update(_json_peek(metadata_path))
    return snapshot


def build_run_metadata(
    *,
    project_root: str | Path | None = None,
    summary_stem: str | None = None,
    output_root: str | Path | None = None,
    manifest_paths: Iterable[str | Path] = (),
    include_packages: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    output_dir = Path(output_root).resolve() if output_root is not None else None
    metadata: dict[str, Any] = {
        "run_metadata_schema_version": RUN_METADATA_SCHEMA_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(root),
        "summary_stem": summary_stem,
        "output_root": str(output_dir) if output_dir is not None else None,
        "python": {
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
            },
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "environment": {
            key: os.environ.get(key) for key in TRACKED_ENV_VARS if os.environ.get(key)
        },
        "git": git_snapshot(root),
        "manifest_snapshots": manifest_snapshots(manifest_paths),
        "benchmark_metadata": benchmark_metadata_snapshot(root),
        "packages": _package_snapshot() if include_packages else {},
    }
    if extra:
        metadata["extra"] = extra
    return metadata


def write_run_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
