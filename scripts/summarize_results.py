# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.experiment.comparison import build_collection_summary, collect_comparison_rows
from ga_lab.experiment.reporting import write_collection_bundle
from ga_lab.experiment.retention import build_retention_plan, write_retention_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GA run results.")
    parser.add_argument("--results-dir", default="outputs", help="Directory containing run outputs")
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Optional display name for the summarized run collection",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    rows = collect_comparison_rows(results_dir)
    if not rows:
        raise SystemExit("No summary.json files found.")

    collection_name = args.collection_name or results_dir.name
    summary = build_collection_summary(collection_name, rows)
    report_paths = write_collection_bundle(
        results_dir,
        rows,
        summary,
        summary_filename="results_summary.json",
    )
    retention_plan = build_retention_plan(
        collection_name,
        rows,
        mode=args.retention_mode,
        keep_best_runs=args.keep_best_runs,
        keep_latest_runs=args.keep_latest_runs,
    )
    retention_paths = write_retention_bundle(results_dir, retention_plan)

    print(report_paths["summary_md"].read_text(encoding="utf-8"))
    print(f"\nMarkdown summary written to: {report_paths['summary_md']}")
    print(f"JSON summary written to: {report_paths['summary_json']}")
    print(f"Run table written to: {report_paths['runs_csv']}")
    print(f"Run JSONL written to: {report_paths['runs_jsonl']}")
    print(f"Retention plan written to: {retention_paths['retention_plan']}")


if __name__ == "__main__":
    main()
