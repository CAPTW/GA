from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.governance import (  # noqa: E402
    build_claim_drift_report,
    build_run_metadata,
    load_claim_registry,
    markdown_claim_drift_report,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether benchmark summaries still satisfy "
            "the machine-readable claim registry."
        )
    )
    parser.add_argument(
        "--registry",
        default=str(PROJECT_ROOT / "claims" / "claim_registry.json"),
        help="Path to the machine-readable claim registry.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        dest="summaries",
        help="Benchmark summary JSON path. May be repeated.",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(PROJECT_ROOT / "outputs" / "benchmark_summary"),
        help="Directory scanned for *_summary.json files when --summary is omitted.",
    )
    parser.add_argument(
        "--reference-summary",
        action="append",
        dest="reference_summaries",
        help="Optional historical or release summary path for metadata purposes.",
    )
    parser.add_argument(
        "--output-json",
        default=str(PROJECT_ROOT / "outputs" / "benchmark_summary" / "claim_drift_report.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(PROJECT_ROOT / "outputs" / "benchmark_summary" / "claim_drift_report.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--output-metadata",
        default=str(PROJECT_ROOT / "outputs" / "benchmark_summary" / "run_metadata.json"),
        help="Run metadata output path.",
    )
    parser.add_argument(
        "--fail-on",
        choices=("FAIL", "WARN", "never"),
        default="FAIL",
        help="Exit policy for ci_gated claims.",
    )
    return parser.parse_args()


def _summary_paths(args: argparse.Namespace) -> list[Path]:
    if args.summaries:
        return [Path(path).resolve() for path in args.summaries]
    summary_dir = Path(args.summary_dir).resolve()
    return sorted(summary_dir.glob("*_summary.json"))


def main() -> int:
    args = parse_args()
    registry = load_claim_registry(args.registry)
    summary_paths = _summary_paths(args)
    summaries = {}
    for path in summary_paths:
        if path.exists():
            summaries[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    report = build_claim_drift_report(
        registry,
        summaries,
        reference_summary_paths=args.reference_summaries or [],
    )

    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(markdown_claim_drift_report(report), encoding="utf-8")

    metadata = build_run_metadata(
        project_root=PROJECT_ROOT,
        summary_stem="claim_drift_report",
        output_root=output_json.parent,
        manifest_paths=[args.registry, *summary_paths],
        extra={
            "reference_summary_paths": args.reference_summaries or [],
            "report_counts": report["counts"],
            "overall_gated_status": report["overall_gated_status"],
        },
    )
    write_run_metadata(args.output_metadata, metadata)

    if args.fail_on == "never":
        return 0
    if args.fail_on == "FAIL" and report["overall_gated_status"] == "FAIL":
        return 1
    if args.fail_on == "WARN" and report["overall_gated_status"] in {"FAIL", "WARN"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
