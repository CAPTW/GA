# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from ga_ops.ingestion import sync_results_dir
from ga_ops.settings import OpsSettings

DEFAULT_REGRESSION_TESTS = [
    "tests/test_baseline_regression.py",
    "tests/test_comparison.py",
    "tests/test_prompt_templates.py",
]


@dataclass(slots=True)
class StepSpec:
    name: str
    command: list[str]
    log_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local nightly regression workflow.")
    parser.add_argument(
        "--output-root",
        default="outputs/nightly",
        help="Directory where nightly run directories will be created",
    )
    parser.add_argument(
        "--collection-name",
        default="nightly_regression",
        help="Collection name used for the top-level nightly run directory and report",
    )
    parser.add_argument(
        "--baseline-manifest",
        default="configs/ci/baseline_smoke.json",
        help="Baseline smoke manifest used for nightly regression checks",
    )
    parser.add_argument(
        "--comparison-manifest",
        default="configs/comparisons/onemax_operator_compare_10seeds.json",
        help="Seed sweep manifest used for nightly operator comparison",
    )
    parser.add_argument(
        "--regression-tests",
        nargs="+",
        default=DEFAULT_REGRESSION_TESTS,
        help="Pytest paths for the fast regression gate",
    )
    parser.add_argument(
        "--git-base",
        default="HEAD~1",
        help="Git base reference used to collect a recent-change summary when git is available",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional UTC timestamp override in YYYYMMDDTHHMMSSZ form for reproducible runs",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the seed sweep comparison step",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip the top-level result summarization step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the nightly plan and metadata without executing commands",
    )
    parser.add_argument(
        "--sync-ops",
        action="store_true",
        help="Sync the nightly output directory into the ops metadata DB and object store",
    )
    parser.add_argument(
        "--ops-actor",
        default="run_nightly",
        help="Actor name recorded in ops audit logs when --sync-ops is used",
    )
    parser.add_argument(
        "--ops-db-path",
        default=None,
        help="Optional override for the ops SQLite database path",
    )
    parser.add_argument(
        "--object-store-root",
        default=None,
        help="Optional override for the local object store root",
    )
    parser.add_argument(
        "--object-store-provider",
        default=None,
        help="Optional object store provider override: local or s3",
    )
    return parser.parse_args()


def _timestamp_value(raw_timestamp: str | None) -> str:
    if raw_timestamp:
        return raw_timestamp
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _recent_changes(git_base: str) -> dict[str, object]:
    git_root = _run_command(["git", "rev-parse", "--show-toplevel"], cwd=PROJECT_ROOT)
    if git_root.returncode != 0:
        return {
            "git_available": False,
            "git_base": git_base,
            "changed_files": [],
            "detection_mode": "unavailable",
            "error": (git_root.stderr or git_root.stdout).strip(),
        }

    diff = _run_command(["git", "diff", "--name-only", f"{git_base}..HEAD"], cwd=PROJECT_ROOT)
    if diff.returncode == 0:
        changed_files = [line.strip() for line in diff.stdout.splitlines() if line.strip()]
        return {
            "git_available": True,
            "git_root": git_root.stdout.strip(),
            "git_base": git_base,
            "changed_files": changed_files,
            "detection_mode": "git_diff",
            "error": "",
        }

    status = _run_command(["git", "status", "--short"], cwd=PROJECT_ROOT)
    changed_files = []
    if status.returncode == 0:
        changed_files = [
            line[3:].strip()
            for line in status.stdout.splitlines()
            if len(line) >= 4 and line[3:].strip()
        ]

    return {
        "git_available": True,
        "git_root": git_root.stdout.strip(),
        "git_base": git_base,
        "changed_files": changed_files,
        "detection_mode": "git_status_fallback",
        "error": (diff.stderr or diff.stdout).strip(),
    }


def _build_steps(args: argparse.Namespace, nightly_dir: Path) -> list[StepSpec]:
    steps = [
        StepSpec(
            name="regression_tests",
            command=[sys.executable, "-m", "pytest", *args.regression_tests],
            log_name="01_regression_tests.log",
        ),
        StepSpec(
            name="baseline_smoke",
            command=[
                sys.executable,
                "scripts/run_baselines.py",
                "--manifest",
                args.baseline_manifest,
                "--output-root",
                str(nightly_dir),
            ],
            log_name="02_baseline_smoke.log",
        ),
    ]
    if not args.skip_comparison:
        steps.append(
            StepSpec(
                name="seed_sweep",
                command=[
                    sys.executable,
                    "scripts/run_baselines.py",
                    "--manifest",
                    args.comparison_manifest,
                    "--output-root",
                    str(nightly_dir),
                ],
                log_name="03_seed_sweep.log",
            )
        )
    if not args.skip_summary:
        steps.append(
            StepSpec(
                name="collection_summary",
                command=[
                    sys.executable,
                    "scripts/summarize_results.py",
                    "--results-dir",
                    str(nightly_dir),
                    "--collection-name",
                    args.collection_name,
                ],
                log_name="04_collection_summary.log",
            )
        )
    return steps


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_log(
    path: Path,
    *,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
) -> None:
    lines = [
        f"Command: {subprocess.list2cmdline(command)}",
        f"Return code: {completed.returncode}",
        "",
        "STDOUT",
        completed.stdout.rstrip(),
        "",
        "STDERR",
        completed.stderr.rstrip(),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown_summary(
    path: Path,
    *,
    collection_name: str,
    timestamp: str,
    recent_changes: dict[str, object],
    steps: list[dict[str, object]],
    run_dir: Path,
) -> None:
    lines = [
        f"# {collection_name}",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Output root: `{run_dir}`",
        "",
        "## Recent changes",
    ]
    if recent_changes["git_available"]:
        lines.append(
            "- Detection mode: "
            f"`{recent_changes['detection_mode']}` "
            f"from `{recent_changes['git_base']}`"
        )
        changed_files = recent_changes["changed_files"]
        if changed_files:
            lines.extend(f"- `{path}`" for path in changed_files)  # type: ignore[arg-type]
        else:
            lines.append("- No changed files were detected.")
        if recent_changes["error"]:
            lines.append(f"- Fallback note: `{recent_changes['error']}`")
    else:
        lines.append("- Git metadata unavailable in this workspace snapshot.")
        if recent_changes["error"]:
            lines.append(f"- Detail: `{recent_changes['error']}`")

    lines.extend(["", "## Steps"])
    for step in steps:
        status = step["status"]
        command = step["display_command"]
        lines.append(f"- `{step['name']}`: `{status}` (`{command}`)")
        if step["log_path"]:
            lines.append(f"  log: `{step['log_path']}`")
        if step["returncode"] is not None:
            lines.append(f"  return code: `{step['returncode']}`")
    lines.extend(
        [
            "",
            "## Expected artifacts",
            "- `SUMMARY.md`, `results_summary.json`, `RUNS.csv`, `RUNS.jsonl` at the nightly root",
            "- Suite-level `SUMMARY.md` and `suite_summary.json` files inside generated suites",
            "- `nightly_plan.json`, `recent_changes.json`, `nightly_status.json`, and `logs/*.log`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    timestamp = _timestamp_value(args.timestamp)
    output_root = Path(args.output_root)
    nightly_dir = output_root / f"{timestamp}_{args.collection_name}"
    logs_dir = nightly_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    recent_changes = _recent_changes(args.git_base)
    _write_json(nightly_dir / "recent_changes.json", recent_changes)

    step_specs = _build_steps(args, nightly_dir)
    plan_payload = {
        "collection_name": args.collection_name,
        "timestamp": timestamp,
        "output_root": str(output_root),
        "nightly_dir": str(nightly_dir),
        "baseline_manifest": args.baseline_manifest,
        "comparison_manifest": None if args.skip_comparison else args.comparison_manifest,
        "regression_tests": args.regression_tests,
        "dry_run": args.dry_run,
        "recent_changes": recent_changes,
        "steps": [
            {
                "name": step.name,
                "command": step.command,
                "display_command": subprocess.list2cmdline(step.command),
                "log_name": step.log_name,
            }
            for step in step_specs
        ],
    }
    _write_json(nightly_dir / "nightly_plan.json", plan_payload)

    step_results: list[dict[str, object]] = []
    for step in step_specs:
        log_path = logs_dir / step.log_name
        if args.dry_run:
            step_results.append(
                {
                    "name": step.name,
                    "status": "dry_run",
                    "command": step.command,
                    "display_command": subprocess.list2cmdline(step.command),
                    "returncode": None,
                    "log_path": "",
                }
            )
            continue

        completed = _run_command(step.command, cwd=PROJECT_ROOT)
        _write_log(log_path, command=step.command, completed=completed)
        step_results.append(
            {
                "name": step.name,
                "status": "passed" if completed.returncode == 0 else "failed",
                "command": step.command,
                "display_command": subprocess.list2cmdline(step.command),
                "returncode": completed.returncode,
                "log_path": str(log_path),
            }
        )

    overall_success = all(result["status"] in {"passed", "dry_run"} for result in step_results)
    status_payload = {
        "collection_name": args.collection_name,
        "timestamp": timestamp,
        "nightly_dir": str(nightly_dir),
        "dry_run": args.dry_run,
        "success": overall_success,
        "steps": step_results,
    }
    _write_json(nightly_dir / "nightly_status.json", status_payload)
    _write_markdown_summary(
        nightly_dir / "NIGHTLY.md",
        collection_name=args.collection_name,
        timestamp=timestamp,
        recent_changes=recent_changes,
        steps=step_results,
        run_dir=nightly_dir,
    )
    if args.sync_ops and not args.dry_run:
        settings = OpsSettings.from_env(project_root=PROJECT_ROOT)
        if args.ops_db_path is not None:
            settings.db_path = Path(args.ops_db_path)
        if args.object_store_root is not None:
            settings.object_store_root = Path(args.object_store_root)
        if args.object_store_provider is not None:
            settings.object_store_provider = args.object_store_provider
        sync_results_dir(
            nightly_dir,
            settings=settings,
            actor=args.ops_actor,
            source_collection=args.collection_name,
        )

    print(f"Nightly directory: {nightly_dir}")
    print(f"Plan: {nightly_dir / 'nightly_plan.json'}")
    print(f"Status: {nightly_dir / 'nightly_status.json'}")
    if args.dry_run:
        print("Dry run complete.")
        return
    if not overall_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
