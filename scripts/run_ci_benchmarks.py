from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class TierSpec:
    name: str
    purpose: str
    runtime_class: str
    cadence: str
    artifact_outputs: tuple[str, ...]
    failure_policy: str
    commands: tuple[list[str], ...]


def _python(*args: str) -> list[str]:
    return [sys.executable, *args]


def _tier_specs(output_root: Path) -> dict[str, TierSpec]:
    return {
        "tier0": TierSpec(
            name="tier0",
            purpose=(
                "lint-adjacent governance checks, registry/schema validation, "
                "and drift-check logic smoke tests."
            ),
            runtime_class="seconds",
            cadence="pre-commit / PR",
            artifact_outputs=(
                "claim_drift_report.json",
                "claim_drift_report.md",
                "run_metadata.json",
            ),
            failure_policy="hard fail on test or ci_gated claim drift failure.",
            commands=(
                _python(
                    "-m",
                    "pytest",
                    "tests/test_claim_governance.py",
                    "tests/test_external_benchmark_suite.py",
                    "tests/test_recommend_solver.py",
                ),
                _python(
                    "scripts/check_claim_drift.py",
                    "--summary-dir",
                    "outputs/benchmark_summary",
                    "--output-json",
                    str(output_root / "claim_drift_report.json"),
                    "--output-md",
                    str(output_root / "claim_drift_report.md"),
                    "--output-metadata",
                    str(output_root / "run_metadata.json"),
                ),
            ),
        ),
        "tier1": TierSpec(
            name="tier1",
            purpose=(
                "fast smoke benchmark that exercises suite runners, benchmark "
                "fetch planning, and artifact generation."
            ),
            runtime_class="low minutes",
            cadence="PR",
            artifact_outputs=("ci_benchmark_plan.json", "tier1/benchmark_smoke/*"),
            failure_policy="hard fail on runner or smoke-suite failure.",
            commands=(
                _python("scripts/fetch_benchmarks.py", "--dry-run", "--format", "text"),
                _python(
                    "scripts/run_baselines.py",
                    "--manifest",
                    "configs/ci/baseline_smoke.json",
                    "--output-root",
                    str(output_root / "tier1" / "benchmark_smoke"),
                ),
            ),
        ),
        "tier2": TierSpec(
            name="tier2",
            purpose="internal validated regression subset for current official internal claims.",
            runtime_class="minutes",
            cadence="nightly",
            artifact_outputs=("ablation_summary.json", "claim_drift_report.json"),
            failure_policy="hard fail on ci_gated internal claim drift regression.",
            commands=(
                _python(
                    "scripts/run_ablation.py",
                    "--manifest",
                    "configs/benchmarks/ablation_manifest.json",
                    "--output-root",
                    str(output_root),
                    "--summary-stem",
                    "ablation_summary",
                ),
                _python(
                    "scripts/check_claim_drift.py",
                    "--summary",
                    str(output_root / "ablation_summary.json"),
                    "--summary",
                    "outputs/benchmark_summary/external_family_summary.json",
                    "--output-json",
                    str(output_root / "claim_drift_report.json"),
                    "--output-md",
                    str(output_root / "claim_drift_report.md"),
                    "--output-metadata",
                    str(output_root / "run_metadata.json"),
                ),
            ),
        ),
        "tier3": TierSpec(
            name="tier3",
            purpose=(
                "external support regression subset for current external and "
                "family-conditional claims."
            ),
            runtime_class="tens of minutes",
            cadence="manual / release-candidate",
            artifact_outputs=("external_family_summary.json", "claim_drift_report.json"),
            failure_policy="hard fail on ci_gated external claim drift regression.",
            commands=(
                _python("scripts/fetch_benchmarks.py", "--format", "text"),
                _python(
                    "scripts/run_external_benchmarks.py",
                    "--manifest",
                    "configs/benchmarks/external_family_manifest.json",
                    "--output-root",
                    str(output_root),
                    "--summary-stem",
                    "external_family_summary",
                ),
                _python(
                    "scripts/check_claim_drift.py",
                    "--summary",
                    str(output_root / "external_family_summary.json"),
                    "--summary",
                    "outputs/benchmark_summary/ablation_summary.json",
                    "--output-json",
                    str(output_root / "claim_drift_report.json"),
                    "--output-md",
                    str(output_root / "claim_drift_report.md"),
                    "--output-metadata",
                    str(output_root / "run_metadata.json"),
                ),
            ),
        ),
        "tier4": TierSpec(
            name="tier4",
            purpose=(
                "weekly deeper suite for confirm manifests, external family "
                "refresh, and docs-facing governance artifacts."
            ),
            runtime_class="heavy / scheduled",
            cadence="weekly / manual",
            artifact_outputs=(
                "baseline_comparison_summary.json",
                "hybrid_comparison_summary.json",
                "ablation_summary.json",
                "external_family_summary.json",
                "claim_drift_report.json",
            ),
            failure_policy=(
                "hard fail on ci_gated claim drift failure; report-only claims "
                "may warn without blocking release notes."
            ),
            commands=(
                _python(
                    "scripts/run_baseline_comparison.py",
                    "--manifest",
                    "configs/benchmarks/baseline_confirm_manifest.json",
                    "--output-root",
                    str(output_root),
                    "--summary-stem",
                    "baseline_comparison_summary",
                ),
                _python(
                    "scripts/run_hybrid_comparison.py",
                    "--manifest",
                    "configs/benchmarks/hybrid_confirm_manifest.json",
                    "--output-root",
                    str(output_root),
                    "--summary-stem",
                    "hybrid_comparison_summary",
                ),
                _python(
                    "scripts/run_ablation.py",
                    "--manifest",
                    "configs/benchmarks/ablation_confirm_manifest.json",
                    "--output-root",
                    str(output_root),
                    "--summary-stem",
                    "ablation_summary",
                ),
                _python("scripts/fetch_benchmarks.py", "--format", "text"),
                _python(
                    "scripts/run_external_benchmarks.py",
                    "--manifest",
                    "configs/benchmarks/external_confirm_manifest.json",
                    "--manifest",
                    "configs/benchmarks/external_family_confirm_manifest.json",
                    "--output-root",
                    str(output_root),
                    "--summary-stem",
                    "external_family_summary",
                ),
                _python(
                    "scripts/check_claim_drift.py",
                    "--summary",
                    str(output_root / "baseline_comparison_summary.json"),
                    "--summary",
                    str(output_root / "hybrid_comparison_summary.json"),
                    "--summary",
                    str(output_root / "ablation_summary.json"),
                    "--summary",
                    str(output_root / "external_family_summary.json"),
                    "--output-json",
                    str(output_root / "claim_drift_report.json"),
                    "--output-md",
                    str(output_root / "claim_drift_report.md"),
                    "--output-metadata",
                    str(output_root / "run_metadata.json"),
                ),
            ),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or plan benchmark governance tiers for CI, nightly, and weekly automation."
    )
    parser.add_argument(
        "--tier",
        choices=("tier0", "tier1", "tier2", "tier3", "tier4"),
        required=True,
        help="Benchmark governance tier to execute.",
    )
    parser.add_argument(
        "--output-root",
        default=".ci-artifacts/benchmark-governance",
        help="Directory where tier outputs and plans are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the execution plan without running subprocesses.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = _tier_specs(output_root)
    spec = specs[args.tier]

    plan = {
        "tier": spec.name,
        "purpose": spec.purpose,
        "runtime_class": spec.runtime_class,
        "cadence": spec.cadence,
        "artifact_outputs": list(spec.artifact_outputs),
        "failure_policy": spec.failure_policy,
        "commands": spec.commands,
        "dry_run": args.dry_run,
    }
    (output_root / "ci_benchmark_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    for command in spec.commands:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
