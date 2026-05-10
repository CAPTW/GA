# ruff: noqa: E402

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the release candidate regression bundle and sync it into ops metadata."
    )
    parser.add_argument(
        "--manifest",
        default="configs/release_candidate/regression_manifest.json",
        help="Manifest used for release candidate regression runs",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/release_candidate",
        help="Directory where RC regression results should be written",
    )
    parser.add_argument(
        "--db-path",
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
    parser.add_argument(
        "--actor",
        default="release_candidate_regression",
        help="Actor name recorded in ops audit logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_baselines.py"),
        "--manifest",
        args.manifest,
        "--output-root",
        args.output_root,
        "--sync-ops",
        "--ops-actor",
        args.actor,
    ]
    if args.db_path is not None:
        command.extend(["--ops-db-path", args.db_path])
    if args.object_store_root is not None:
        command.extend(["--object-store-root", args.object_store_root])
    if args.object_store_provider is not None:
        command.extend(["--object-store-provider", args.object_store_provider])

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
