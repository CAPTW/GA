from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_project_env_files(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for name in (".env", ".env.ops"):
        path = root / name
        if path.exists():
            load_dotenv(path, override=False)


def _path_from_env(name: str, *, root: Path, default: Path) -> Path:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return root / candidate


@dataclass(slots=True)
class OpsSettings:
    project_root: Path
    db_path: Path
    object_store_provider: str
    object_store_root: Path
    object_store_bucket: str
    s3_endpoint_url: str | None
    s3_region: str | None
    s3_access_key_id: str | None
    s3_secret_access_key: str | None
    scheduler_jobs_path: Path
    scheduler_timezone: str
    logs_root: Path
    reports_root: Path
    dashboard_results_dir: Path

    @classmethod
    def from_env(cls, *, project_root: Path | None = None) -> OpsSettings:
        root = (project_root or PROJECT_ROOT).resolve()
        _load_project_env_files(root)
        data_root = root / "var" / "ops"
        db_path = _path_from_env(
            "GA_LAB_OPS_DB_PATH",
            root=root,
            default=data_root / "ga_ops.db",
        )
        object_store_root = _path_from_env(
            "GA_LAB_OBJECT_STORE_ROOT",
            root=root,
            default=data_root / "object_store",
        )
        scheduler_jobs_path = _path_from_env(
            "GA_LAB_SCHEDULER_JOBS_PATH",
            root=root,
            default=root / "configs" / "schedules" / "default_jobs.json",
        )
        return cls(
            project_root=root,
            db_path=db_path,
            object_store_provider=os.getenv("GA_LAB_OBJECT_STORE_PROVIDER", "local").strip(),
            object_store_root=object_store_root,
            object_store_bucket=os.getenv("GA_LAB_OBJECT_STORE_BUCKET", "ga-lab-artifacts"),
            s3_endpoint_url=os.getenv("GA_LAB_S3_ENDPOINT_URL"),
            s3_region=os.getenv("GA_LAB_S3_REGION"),
            s3_access_key_id=os.getenv("GA_LAB_S3_ACCESS_KEY_ID"),
            s3_secret_access_key=os.getenv("GA_LAB_S3_SECRET_ACCESS_KEY"),
            scheduler_jobs_path=scheduler_jobs_path,
            scheduler_timezone=os.getenv("GA_LAB_SCHEDULER_TIMEZONE", "Asia/Seoul"),
            logs_root=_path_from_env(
                "GA_LAB_LOGS_ROOT",
                root=root,
                default=data_root / "logs",
            ),
            reports_root=_path_from_env(
                "GA_LAB_REPORTS_ROOT",
                root=root,
                default=root / "outputs" / "reports",
            ),
            dashboard_results_dir=_path_from_env(
                "GA_LAB_DASHBOARD_RESULTS_DIR",
                root=root,
                default=root / "outputs",
            ),
        )

    def ensure_directories(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)
        if self.object_store_provider == "local":
            self.object_store_root.mkdir(parents=True, exist_ok=True)
