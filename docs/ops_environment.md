# Ops Environment

This repository now includes an isolated ops layer under `services/ga_ops/` so the GA engine stays in `src/ga_lab/`.

## What it adds

- SQLite metadata DB for experiment, scheduler, artifact, and audit records
- Object storage abstraction with local filesystem mode and S3-compatible mode
- Audit logs and bearer-token access control for the internal ops API
- DB-backed Streamlit dashboard for recent runs, regressions, best configs, jobs, and audit events
- Scheduler jobs for nightly benchmarks, weekly reports, and release-candidate regression runs
- Optional Codex invocation endpoint for internal tools

## Local setup

Install the ops extra:

```bash
pip install -e .[ops]
```

Create an admin token or a named token:

```bash
python scripts/ops_manage_tokens.py --name local-admin --scopes ops.read ops.write codex.invoke
```

Sync existing outputs into the metadata DB and object store:

```bash
python scripts/ops_sync_results.py --results-dir outputs
```

Start the API:

```bash
python -m uvicorn ga_ops.app:app --host 0.0.0.0 --port 8000 --app-dir services
```

To use Codex through the logged-in CLI session instead of an API key:

```bash
codex login
set GA_LAB_CODEX_BACKEND=cli
set GA_LAB_CODEX_MODEL=gpt-5.4
```

You can keep those settings in a project-local env file:

```bash
copy .env.ops-cli.example .env.ops-cli
```

`GA_LAB_CODEX_BACKEND=auto` is the default. In auto mode the ops layer uses:

- the OpenAI SDK when `OPENAI_API_KEY` is set
- the logged-in `codex` CLI when no API key is present

CLI-backed Codex invocation is intended for host-local runs. Containerized services still need an
installed and logged-in `codex` CLI inside the container, or they should continue using the SDK
with `OPENAI_API_KEY`.

Windows helper scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_ops_api_cli.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_ops_api_cli.ps1 -ValidateOnly
powershell -ExecutionPolicy Bypass -File scripts/run_ops_scheduler_cli.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_ops_scheduler_cli.ps1 -Once
```

The helper scripts:

- verify `codex login status`
- default the ops stack to `GA_LAB_CODEX_BACKEND=cli`
- reuse `.env` and `.env.ops-cli` automatically when present
- fall back to local-safe defaults for the DB, logs, reports, and scheduler config

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

## Docker environment

The repository ships with a containerized local ops stack:

```bash
docker compose up --build
```

Services:

- `ops-api`: FastAPI service on `http://localhost:8000`
- `ops-dashboard`: Streamlit dashboard on `http://localhost:8502`
- `ops-scheduler`: APScheduler host for recurring jobs
- `minio`: S3-compatible object storage on `http://localhost:9000` with console on `http://localhost:9001`

The compose file uses:

- SQLite DB at `/app/var/ops/ga_ops.db`
- MinIO bucket `ga-lab-artifacts`
- Scheduler definitions from `configs/schedules/default_jobs.json`

Set `GA_LAB_OPS_ADMIN_TOKEN` before running the API in any shared environment.

## API notes

Endpoints:

- `GET /health`
- `GET /dashboard/summary`
- `GET /runs`
- `GET /runs/{run_id}/artifacts`
- `GET /jobs`
- `POST /ingestions/sync`
- `POST /jobs/{job_name}/run`
- `POST /codex/invoke`

Scopes:

- `ops.read`
- `ops.write`
- `codex.invoke`

The API writes audit events for every request, sync action, token rotation, scheduler run, and weekly report generation.
