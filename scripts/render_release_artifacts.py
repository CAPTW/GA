from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.governance.claim_registry import load_claim_registry  # noqa: E402
from ga_lab.governance.release_artifacts import (  # noqa: E402
    RELEASE_ARTIFACTS_SCHEMA_VERSION,
    build_badges,
    build_claim_matrix,
    build_release_snapshot,
    build_solver_matrix,
    build_summary_inventory,
    build_validated_ranges,
    load_json,
    load_release_manifest,
    load_summary_payloads,
    render_benchmark_card,
    render_readme_evidence_snapshot,
    render_readme_release_status,
    render_readme_solver_preview,
    render_release_status_markdown,
    render_solver_matrix_markdown,
    update_markdown_file,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render public release artifacts from the claim registry and latest evidence."
    )
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "configs" / "release" / "release_artifacts_manifest.json"),
        help="Release artifact manifest path.",
    )
    parser.add_argument(
        "--registry",
        default=str(PROJECT_ROOT / "claims" / "claim_registry.json"),
        help="Claim registry path.",
    )
    parser.add_argument(
        "--drift-report",
        default=str(PROJECT_ROOT / "outputs" / "benchmark_summary" / "claim_drift_report.json"),
        help="Latest claim drift report path.",
    )
    parser.add_argument(
        "--docs-dir",
        default=str(PROJECT_ROOT / "docs"),
        help="Docs directory containing marker templates.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory where JSON/CSV release artifacts will be written.",
    )
    parser.add_argument(
        "--readme-path",
        default=str(PROJECT_ROOT / "README.md"),
        help="README path with marker blocks to refresh.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_release_manifest(args.manifest)
    registry = load_claim_registry(args.registry)
    drift_report = load_json(args.drift_report)

    docs_dir = Path(args.docs_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    badges_dir = artifacts_dir / "badges"
    readme_path = Path(args.readme_path).resolve()

    summary_payloads = load_summary_payloads(PROJECT_ROOT, manifest["summary_paths"])
    summary_inventory = build_summary_inventory(
        PROJECT_ROOT,
        manifest["summary_paths"],
        summary_payloads,
    )
    validated_ranges = build_validated_ranges(summary_payloads)

    claim_matrix = build_claim_matrix(registry, drift_report)
    solver_matrix = build_solver_matrix(registry, drift_report, manifest)
    release_snapshot = build_release_snapshot(
        PROJECT_ROOT,
        registry,
        drift_report,
        manifest,
        claim_matrix,
        solver_matrix,
        summary_inventory,
        validated_ranges,
    )
    badges = build_badges(release_snapshot, claim_matrix)

    claim_payload = {
        "schema_version": RELEASE_ARTIFACTS_SCHEMA_VERSION,
        "generated_at_utc": release_snapshot["generated_at_utc"],
        "rows": claim_matrix,
    }
    solver_payload = {
        "schema_version": RELEASE_ARTIFACTS_SCHEMA_VERSION,
        "generated_at_utc": release_snapshot["generated_at_utc"],
        "rows": solver_matrix,
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    badges_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifacts_dir / "claim_matrix.json", claim_payload)
    write_csv(artifacts_dir / "claim_matrix.csv", claim_matrix)
    write_json(artifacts_dir / "solver_matrix.json", solver_payload)
    write_csv(artifacts_dir / "solver_matrix.csv", solver_matrix)
    write_json(artifacts_dir / "release_snapshot.json", release_snapshot)
    for badge_name, badge_payload in badges.items():
        write_json(badges_dir / f"{badge_name}.json", badge_payload)

    update_markdown_file(
        docs_dir / "benchmark_card.md",
        {
            "benchmark-card": render_benchmark_card(release_snapshot, claim_matrix),
        },
    )
    update_markdown_file(
        docs_dir / "solver_matrix.md",
        {
            "solver-matrix": render_solver_matrix_markdown(solver_matrix),
        },
    )
    update_markdown_file(
        docs_dir / "release_status.md",
        {
            "release-status": render_release_status_markdown(
                release_snapshot,
                claim_matrix,
            ),
        },
    )
    update_markdown_file(
        readme_path,
        {
            "evidence-snapshot": render_readme_evidence_snapshot(
                release_snapshot,
                claim_matrix,
                manifest,
            ),
            "solver-matrix-preview": render_readme_solver_preview(
                solver_matrix,
                manifest,
            ),
            "release-status": render_readme_release_status(release_snapshot),
        },
    )

    payload = {
        "schema_version": RELEASE_ARTIFACTS_SCHEMA_VERSION,
        "generated_at_utc": release_snapshot["generated_at_utc"],
        "artifacts_dir": str(artifacts_dir),
        "docs_dir": str(docs_dir),
        "readme_path": str(readme_path),
        "generated_files": [
            str(artifacts_dir / "claim_matrix.json"),
            str(artifacts_dir / "claim_matrix.csv"),
            str(artifacts_dir / "solver_matrix.json"),
            str(artifacts_dir / "solver_matrix.csv"),
            str(artifacts_dir / "release_snapshot.json"),
            str(docs_dir / "benchmark_card.md"),
            str(docs_dir / "solver_matrix.md"),
            str(docs_dir / "release_status.md"),
            str(readme_path),
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
