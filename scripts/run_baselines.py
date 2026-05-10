# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from ga_lab.config import GAConfig
from ga_lab.experiment.comparison import build_baseline_suite_summary, flatten_summary
from ga_lab.experiment.grid import build_grid_summary
from ga_lab.experiment.reporting import write_collection_bundle
from ga_lab.experiment.retention import build_retention_plan, write_retention_bundle
from ga_lab.experiment.suite import SuiteEntry, load_suite_manifest
from ga_lab.experiment.tracking import write_json
from ga_lab.runner import run_experiment
from ga_ops.ingestion import sync_results_dir
from ga_ops.settings import OpsSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline suite from a manifest.")
    parser.add_argument(
        "--manifest",
        default="configs/baselines/manifest.json",
        help="Path to a baseline suite manifest",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Directory where suite outputs will be stored",
    )
    parser.add_argument(
        "--default-seeds",
        type=int,
        default=1,
        help="Fallback seed count when a manifest entry omits seeds",
    )
    parser.add_argument(
        "--retention-mode",
        choices=("history-only", "run-dirs"),
        default="history-only",
        help="How aggressively retention plans should prune older run artifacts",
    )
    parser.add_argument(
        "--keep-best-runs",
        type=int,
        default=2,
        help="How many top runs per label to keep fully in retention plans",
    )
    parser.add_argument(
        "--keep-latest-runs",
        type=int,
        default=1,
        help="How many latest runs per label to keep fully in retention plans",
    )
    parser.add_argument(
        "--sync-ops",
        action="store_true",
        help="Sync the completed suite into the ops metadata DB and object store",
    )
    parser.add_argument(
        "--ops-actor",
        default="run_baselines",
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


def _safe_label(label: str, index: int) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in label.strip()
    ).strip("_")
    return normalized or f"entry_{index}"


def _materialize_entry_config(
    suite_dir: Path,
    entry: SuiteEntry,
    index: int,
) -> Path:
    config_dir = suite_dir / "generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{index:02d}_{_safe_label(entry.label, index)}.json"
    config_path.write_text(
        json.dumps(entry.config_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config_path


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest, entries = load_suite_manifest(manifest_path, default_seeds=args.default_seeds)
    suite_name = manifest["suite_name"]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suite_dir = Path(args.output_root) / f"{timestamp}_{suite_name}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for index, entry in enumerate(entries, start=1):
        materialized_config_path = _materialize_entry_config(suite_dir, entry, index)
        base_config = GAConfig.from_dict(entry.config_data)
        base_config.seed = entry.seed_start

        entry_results = []
        for offset in range(entry.seeds):
            config = GAConfig.from_dict(entry.config_data)
            config.seed = entry.seed_start + offset
            if entry.seeds > 1:
                config.run_name = f"{base_config.run_name}_seed{config.seed}"
            result = run_experiment(config=config, output_root=suite_dir)
            entry_results.append(result)
            comparison_rows.append(
                flatten_summary(
                    result.summary,
                    config=config.to_dict(),
                    suite_name=suite_name,
                    baseline_label=entry.label,
                    config_path=str(materialized_config_path),
                    output_dir=str(result.output_dir),
                )
            )

        if entry.seeds > 1:
            grid_summary = build_grid_summary(base_config, entry_results, entry.seeds)
            write_json(
                suite_dir / f"{_safe_label(entry.label, index)}_grid_summary.json",
                grid_summary,
            )

    write_json(suite_dir / "manifest.json", manifest)
    suite_summary = build_baseline_suite_summary(
        suite_name,
        manifest_path,
        comparison_rows,
    )
    report_paths = write_collection_bundle(
        suite_dir,
        comparison_rows,
        suite_summary,
        summary_filename="suite_summary.json",
    )

    retention_plan = build_retention_plan(
        suite_name,
        comparison_rows,
        mode=args.retention_mode,
        keep_best_runs=args.keep_best_runs,
        keep_latest_runs=args.keep_latest_runs,
    )
    retention_paths = write_retention_bundle(suite_dir, retention_plan)
    if args.sync_ops:
        settings = OpsSettings.from_env(project_root=PROJECT_ROOT)
        if args.ops_db_path is not None:
            settings.db_path = Path(args.ops_db_path)
        if args.object_store_root is not None:
            settings.object_store_root = Path(args.object_store_root)
        if args.object_store_provider is not None:
            settings.object_store_provider = args.object_store_provider
        sync_results_dir(
            suite_dir,
            settings=settings,
            actor=args.ops_actor,
            source_collection=suite_name,
        )

    print(json.dumps(suite_summary, indent=2, ensure_ascii=False))
    print(f"\nSuite directory: {suite_dir}")
    print(f"Markdown summary: {report_paths['summary_md']}")
    print(f"JSON summary: {report_paths['summary_json']}")
    print(f"Retention plan: {retention_paths['retention_plan']}")


if __name__ == "__main__":
    main()
