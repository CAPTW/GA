# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.config import load_config
from ga_lab.experiment.comparison import build_collection_summary, flatten_summary
from ga_lab.experiment.grid import build_grid_summary
from ga_lab.experiment.reporting import write_collection_bundle
from ga_lab.experiment.retention import build_retention_plan, write_retention_bundle
from ga_lab.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the same GA config across multiple seeds.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run")
    parser.add_argument("--output-root", default="outputs", help="Root output directory")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    run_results = []
    comparison_rows = []
    for offset in range(args.seeds):
        config = load_config(args.config)
        config.run_name = f"{base_config.run_name}_seed{base_config.seed + offset}"
        config.seed = base_config.seed + offset
        result = run_experiment(config=config, output_root=args.output_root)
        run_results.append(result)
        comparison_rows.append(
            flatten_summary(
                result.summary,
                config=config.to_dict(),
                suite_name=base_config.run_name,
                baseline_label=base_config.run_name,
                config_path=str(Path(args.config).resolve()),
                output_dir=str(result.output_dir),
            )
        )

    aggregate = build_grid_summary(base_config, run_results, args.seeds)
    aggregate_path = Path(args.output_root) / f"{base_config.run_name}_grid_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    collection_summary = build_collection_summary(base_config.run_name, comparison_rows)
    report_paths = write_collection_bundle(
        args.output_root,
        comparison_rows,
        collection_summary,
        summary_filename="results_summary.json",
    )
    retention_plan = build_retention_plan(
        base_config.run_name,
        comparison_rows,
        mode=args.retention_mode,
        keep_best_runs=args.keep_best_runs,
        keep_latest_runs=args.keep_latest_runs,
    )
    retention_paths = write_retention_bundle(args.output_root, retention_plan)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"\nAggregate summary: {aggregate_path}")
    print(f"Markdown summary: {report_paths['summary_md']}")
    print(f"JSON summary: {report_paths['summary_json']}")
    print(f"Retention plan: {retention_paths['retention_plan']}")


if __name__ == "__main__":
    main()
