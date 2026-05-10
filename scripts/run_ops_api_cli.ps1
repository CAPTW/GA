param(
    [string]$ListenHost = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "Codex CLI was not found on PATH. Install it or set GA_LAB_CODEX_CLI_COMMAND."
}

$codexCommand = $env:GA_LAB_CODEX_CLI_COMMAND
if (-not $codexCommand) {
    $resolved = Get-Command codex.cmd -ErrorAction SilentlyContinue
    if (-not $resolved) { $resolved = Get-Command codex.exe -ErrorAction SilentlyContinue }
    if (-not $resolved) { $resolved = Get-Command codex }
    $codexCommand = $resolved.Path
    $env:GA_LAB_CODEX_CLI_COMMAND = $codexCommand
}

$loginStatus = python -c "import subprocess, sys; completed = subprocess.run(['cmd.exe', '/c', sys.argv[1], 'login', 'status'], capture_output=True, text=True, encoding='utf-8', errors='replace'); text = (completed.stdout or completed.stderr).strip(); print(text); sys.exit(completed.returncode)" "$codexCommand"
if ($LASTEXITCODE -ne 0) {
    throw "Codex CLI is not logged in. Run `codex login` first. Details: $loginStatus"
}

if (-not $env:GA_LAB_CODEX_BACKEND) { $env:GA_LAB_CODEX_BACKEND = "cli" }
if (-not $env:GA_LAB_CODEX_MODEL) { $env:GA_LAB_CODEX_MODEL = "gpt-5.4" }
if (-not $env:GA_LAB_OPS_DB_PATH) { $env:GA_LAB_OPS_DB_PATH = Join-Path $projectRoot "var\ops\ga_ops_cli.db" }
if (-not $env:GA_LAB_LOGS_ROOT) { $env:GA_LAB_LOGS_ROOT = Join-Path $projectRoot "var\ops\logs" }
if (-not $env:GA_LAB_REPORTS_ROOT) { $env:GA_LAB_REPORTS_ROOT = Join-Path $projectRoot "outputs\reports" }
if (-not $env:GA_LAB_DASHBOARD_RESULTS_DIR) { $env:GA_LAB_DASHBOARD_RESULTS_DIR = Join-Path $projectRoot "outputs" }
if (-not $env:GA_LAB_SCHEDULER_JOBS_PATH) { $env:GA_LAB_SCHEDULER_JOBS_PATH = Join-Path $projectRoot "configs\schedules\default_jobs.json" }
if (-not $env:GA_LAB_SCHEDULER_TIMEZONE) { $env:GA_LAB_SCHEDULER_TIMEZONE = "Asia/Seoul" }
if (-not $env:GA_LAB_OPS_ADMIN_TOKEN) { $env:GA_LAB_OPS_ADMIN_TOKEN = "local-dev-admin" }

$summary = @{
    project_root = $projectRoot
    codex_backend = $env:GA_LAB_CODEX_BACKEND
    codex_model = $env:GA_LAB_CODEX_MODEL
    codex_cli_command = $env:GA_LAB_CODEX_CLI_COMMAND
    ops_db_path = $env:GA_LAB_OPS_DB_PATH
    scheduler_jobs_path = $env:GA_LAB_SCHEDULER_JOBS_PATH
    scheduler_timezone = $env:GA_LAB_SCHEDULER_TIMEZONE
    host = $ListenHost
    port = $Port
} | ConvertTo-Json -Depth 4

if ($ValidateOnly) {
    Write-Output $summary
    exit 0
}

Write-Output $summary
python -m uvicorn ga_ops.app:app --host $ListenHost --port $Port --app-dir services
