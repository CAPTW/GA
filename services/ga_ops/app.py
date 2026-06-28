from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise RuntimeError("The ops API requires the optional 'fastapi' dependency") from exc

from .db import AuthenticatedActor, OpsDatabase
from .ingestion import sync_results_dir
from .scheduler import load_job_definitions, register_job_definitions, run_job
from .settings import OpsSettings

app = FastAPI(title="Evolutionary Solver Benchmark Lab Ops API", version="0.1.0")
security = HTTPBearer(auto_error=False)


class SyncRequest(BaseModel):
    results_dir: str = Field(default="outputs")
    source_collection: str | None = Field(default=None)
    upload_artifacts: bool = Field(default=True)


@lru_cache
def get_settings() -> OpsSettings:
    settings = OpsSettings.from_env()
    settings.ensure_directories()
    return settings


def _open_database(settings: OpsSettings) -> OpsDatabase:
    database = OpsDatabase(settings.db_path)
    database.initialize()
    return database


def _current_actor(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    settings: Annotated[OpsSettings, Depends(get_settings)],
) -> AuthenticatedActor:
    token = credentials.credentials if credentials is not None else None
    with _open_database(settings) as database:
        actor = database.authenticate_token(token)
    if actor is None:
        raise HTTPException(status_code=401, detail="A valid Bearer token is required")
    return actor


def require_scope(scope: str):
    def dependency(
        actor: Annotated[AuthenticatedActor, Depends(_current_actor)],
    ) -> AuthenticatedActor:
        if scope not in actor.scopes:
            raise HTTPException(status_code=403, detail=f"Missing required scope: {scope}")
        return actor

    return dependency


@app.on_event("startup")
def startup() -> None:
    settings = get_settings()
    with _open_database(settings) as database:
        database.initialize()
    if settings.scheduler_jobs_path.exists():
        jobs = load_job_definitions(settings.scheduler_jobs_path)
        register_job_definitions(settings=settings, jobs=jobs)


@app.middleware("http")
async def audit_requests(request: Request, call_next):
    settings = get_settings()
    actor_name = "anonymous"
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        with _open_database(settings) as database:
            actor = database.authenticate_token(token)
            if actor is not None:
                actor_name = actor.name

    status = "success"
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            status = "error"
        return response
    except Exception:
        status = "error"
        raise
    finally:
        with _open_database(settings) as database:
            database.log_audit(
                actor=actor_name,
                action=f"{request.method} {request.url.path}",
                resource_type="api_request",
                resource_id=request.url.path,
                status=status,
                details={"query": str(request.url.query)},
            )


@app.get("/health")
def health(settings: Annotated[OpsSettings, Depends(get_settings)]) -> dict[str, Any]:
    return {
        "status": "ok",
        "db_path": str(settings.db_path),
        "object_store_provider": settings.object_store_provider,
        "scheduler_jobs_path": str(settings.scheduler_jobs_path),
    }


@app.get("/dashboard/summary")
def dashboard_summary(
    actor: Annotated[AuthenticatedActor, Depends(require_scope("ops.read"))],
    settings: Annotated[OpsSettings, Depends(get_settings)],
) -> dict[str, Any]:
    with _open_database(settings) as database:
        summary = database.dashboard_summary()
        database.log_audit(
            actor=actor.name,
            action="dashboard_summary",
            resource_type="dashboard",
            resource_id="summary",
            status="success",
        )
        return summary


@app.get("/runs")
def list_runs(
    actor: Annotated[AuthenticatedActor, Depends(require_scope("ops.read"))],
    settings: Annotated[OpsSettings, Depends(get_settings)],
    limit: int = 100,
    suite_name: str | None = None,
    problem: str | None = None,
    baseline_label: str | None = None,
) -> list[dict[str, Any]]:
    with _open_database(settings) as database:
        runs = database.list_runs(
            limit=limit,
            suite_name=suite_name,
            problem=problem,
            baseline_label=baseline_label,
        )
        database.log_audit(
            actor=actor.name,
            action="list_runs",
            resource_type="run",
            status="success",
            details={
                "limit": limit,
                "suite_name": suite_name,
                "problem": problem,
                "baseline_label": baseline_label,
            },
        )
        return runs


@app.get("/runs/{run_id}/artifacts")
def run_artifacts(
    run_id: int,
    actor: Annotated[AuthenticatedActor, Depends(require_scope("ops.read"))],
    settings: Annotated[OpsSettings, Depends(get_settings)],
) -> list[dict[str, Any]]:
    with _open_database(settings) as database:
        artifacts = database.list_run_artifacts(run_id)
        database.log_audit(
            actor=actor.name,
            action="list_run_artifacts",
            resource_type="run",
            resource_id=str(run_id),
            status="success",
        )
        return artifacts


@app.get("/jobs")
def jobs(
    actor: Annotated[AuthenticatedActor, Depends(require_scope("ops.read"))],
    settings: Annotated[OpsSettings, Depends(get_settings)],
) -> dict[str, Any]:
    with _open_database(settings) as database:
        payload = {
            "scheduled_jobs": database.list_scheduled_jobs(),
            "recent_job_runs": database.list_job_runs(limit=20),
        }
        database.log_audit(
            actor=actor.name,
            action="list_jobs",
            resource_type="scheduled_job",
            status="success",
        )
        return payload


@app.post("/ingestions/sync")
def sync_ingestion(
    request: SyncRequest,
    actor: Annotated[AuthenticatedActor, Depends(require_scope("ops.write"))],
    settings: Annotated[OpsSettings, Depends(get_settings)],
) -> dict[str, Any]:
    summary = sync_results_dir(
        request.results_dir,
        settings=settings,
        actor=actor.name,
        upload_artifacts=request.upload_artifacts,
        source_collection=request.source_collection,
    )
    return summary


@app.post("/jobs/{job_name}/run")
def run_named_job(
    job_name: str,
    actor: Annotated[AuthenticatedActor, Depends(require_scope("ops.write"))],
    settings: Annotated[OpsSettings, Depends(get_settings)],
) -> dict[str, Any]:
    jobs = load_job_definitions(settings.scheduler_jobs_path)
    selected = next((job for job in jobs if job.name == job_name), None)
    if selected is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_name}")
    return run_job(selected, settings=settings, triggered_by=actor.name)
