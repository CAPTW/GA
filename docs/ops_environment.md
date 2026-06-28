# Ops Environment

This repository includes an isolated ops layer under `services/ga_ops/` so the
GA engine stays in `src/ga_lab/`.

## What It Adds

- SQLite metadata DB for experiment, scheduler, artifact, and audit records.
- Object storage abstraction with local filesystem mode and S3-compatible mode.
- Audit logs and bearer-token access control for the internal ops API.
- DB-backed Streamlit dashboard for recent runs, regressions, best configs,
  jobs, and audit events.
- Scheduler jobs for nightly benchmarks, weekly reports, and release-candidate
  regression runs.

## Local Setup

Install the ops extra:

```bash
pip install -e .[ops]
```

Create an admin token or a named token:

```bash
python scripts/ops_manage_tokens.py --name local-admin --scopes ops.read ops.write
```

Sync existing outputs into the metadata DB and object store:

```bash
python scripts/ops_sync_results.py --results-dir outputs
```

Start the API:

```bash
python -m uvicorn ga_ops.app:app --host 0.0.0.0 --port 8000 --app-dir services
```

Start the dashboard:

```bash
streamlit run scripts/ops_dashboard.py
```

Run the scheduler once:

```bash
python scripts/run_ops_scheduler.py --once
```

Run the persistent scheduler:

```bash
python scripts/run_ops_scheduler.py
```

Generate a weekly report:

```bash
python scripts/ops_generate_weekly_report.py
```

Run the release-candidate regression bundle:

```bash
python scripts/run_release_candidate_regression.py
```

## Docker Environment

The repository ships with a containerized local ops stack:

```bash
docker compose up --build
```

Services:

- `ops-api`: FastAPI service on `http://localhost:8000`
- `ops-dashboard`: Streamlit dashboard on `http://localhost:8502`
- `ops-scheduler`: APScheduler host for recurring jobs
- `minio`: S3-compatible object storage on `http://localhost:9000` with console
  on `http://localhost:9001`

The compose file uses:

- SQLite DB at `/app/var/ops/ga_ops.db`
- MinIO bucket `ga-lab-artifacts`
- Scheduler definitions from `configs/schedules/default_jobs.json`

Set `GA_LAB_OPS_ADMIN_TOKEN` before running the API in any shared environment.

## API Notes

Endpoints:

- `GET /health`
- `GET /dashboard/summary`
- `GET /runs`
- `GET /runs/{run_id}/artifacts`
- `GET /jobs`
- `POST /ingestions/sync`
- `POST /jobs/{job_name}/run`

Scopes:

- `ops.read`
- `ops.write`

The API writes audit events for every request, sync action, token rotation,
scheduler run, and weekly report generation.
