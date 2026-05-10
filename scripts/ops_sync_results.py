# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from ga_ops.ingestion import sync_results_dir
from ga_ops.settings import OpsSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync result directories into the ops metadata store."
    )
    parser.add_argument(
        "--results-dir",
        default="outputs",
        help="Results directory to scan",
    )
    parser.add_argument(
        "--actor",
        default="ops_sync_results",
        help="Actor name recorded in ops audit logs",
    )
    parser.add_argument(
        "--source-collection",
        default=None,
        help="Optional collection name override stored with ingested runs",
    )
    parser.add_argument(
        "--skip-artifact-upload",
        action="store_true",
        help="Store metadata only without copying artifacts into object storage",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = OpsSettings.from_env(project_root=PROJECT_ROOT)
    if args.db_path is not None:
        settings.db_path = Path(args.db_path)
    if args.object_store_root is not None:
        settings.object_store_root = Path(args.object_store_root)
    if args.object_store_provider is not None:
        settings.object_store_provider = args.object_store_provider
    summary = sync_results_dir(
        args.results_dir,
        settings=settings,
        actor=args.actor,
        upload_artifacts=not args.skip_artifact_upload,
        source_collection=args.source_collection,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
