from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.experiment.external_benchmark_suite import load_manifest as load_external_manifest
from ga_lab.experiment.suite import load_suite_manifest
from ga_lab.governance.claim_drift import (
    FAIL,
    NOT_EVALUATED,
    PASS,
    WARN,
    build_claim_drift_report,
)
from ga_lab.governance.claim_registry import load_claim_registry
from ga_lab.governance.run_metadata import build_run_metadata


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_claim_registry_loads_and_has_unique_claim_ids() -> None:
    registry = load_claim_registry(_project_root() / "claims" / "claim_registry.json")
    claim_ids = [claim["claim_id"] for claim in registry["claims"]]

    assert registry["registry_schema_version"] == 1
    assert len(claim_ids) == len(set(claim_ids))
    assert "tsp_external_nn2opt_default" in claim_ids
    assert "zdt_external_nsga2_over_random_archive" in claim_ids


def test_claim_drift_report_marks_pass_warn_fail_and_not_evaluated() -> None:
    registry = {
        "registry_schema_version": 1,
        "claims": [
            {
                "claim_id": "pass_claim",
                "label": "pass",
                "status": "official",
                "evidence_scope": "internal",
                "problem_family": "demo",
                "validated_ranges": ["demo"],
                "comparators": ["a", "b"],
                "metrics": ["metric"],
                "pass_condition": "positive and significant",
                "warning_threshold": "positive only",
                "source_summary_paths": ["demo_summary.json"],
                "doc_locations": ["README.md"],
                "notes": "",
                "governance": {"mode": "ci_gated", "tier": "tier0"},
                "checks": [
                    {
                        "summary_stem": "demo_summary",
                        "rowset": "comparison_rows",
                        "match": {"comparison_id": "pass_row"},
                        "pass_if": [
                            {"field": "p_value", "op": "<=", "value": 0.05},
                            {"field": "oriented_diff_ci_low", "op": ">", "value": 0.0},
                        ],
                        "warn_if": [{"field": "oriented_mean_diff", "op": ">", "value": 0.0}],
                    }
                ],
            },
            {
                "claim_id": "warn_claim",
                "label": "warn",
                "status": "official",
                "evidence_scope": "internal",
                "problem_family": "demo",
                "validated_ranges": ["demo"],
                "comparators": ["a", "b"],
                "metrics": ["metric"],
                "pass_condition": "positive and significant",
                "warning_threshold": "positive only",
                "source_summary_paths": ["demo_summary.json"],
                "doc_locations": ["README.md"],
                "notes": "",
                "governance": {"mode": "ci_gated", "tier": "tier0"},
                "checks": [
                    {
                        "summary_stem": "demo_summary",
                        "rowset": "comparison_rows",
                        "match": {"comparison_id": "warn_row"},
                        "pass_if": [
                            {"field": "p_value", "op": "<=", "value": 0.05},
                            {"field": "oriented_diff_ci_low", "op": ">", "value": 0.0},
                        ],
                        "warn_if": [{"field": "oriented_mean_diff", "op": ">", "value": 0.0}],
                    }
                ],
            },
            {
                "claim_id": "fail_claim",
                "label": "fail",
                "status": "official",
                "evidence_scope": "internal",
                "problem_family": "demo",
                "validated_ranges": ["demo"],
                "comparators": ["a", "b"],
                "metrics": ["metric"],
                "pass_condition": "positive and significant",
                "warning_threshold": "positive only",
                "source_summary_paths": ["demo_summary.json"],
                "doc_locations": ["README.md"],
                "notes": "",
                "governance": {"mode": "ci_gated", "tier": "tier0"},
                "checks": [
                    {
                        "summary_stem": "demo_summary",
                        "rowset": "comparison_rows",
                        "match": {"comparison_id": "fail_row"},
                        "pass_if": [
                            {"field": "p_value", "op": "<=", "value": 0.05},
                            {"field": "oriented_diff_ci_low", "op": ">", "value": 0.0},
                        ],
                        "warn_if": [{"field": "oriented_mean_diff", "op": ">", "value": 0.0}],
                    }
                ],
            },
            {
                "claim_id": "missing_claim",
                "label": "missing",
                "status": "experimental",
                "evidence_scope": "internal",
                "problem_family": "demo",
                "validated_ranges": ["demo"],
                "comparators": ["a", "b"],
                "metrics": ["metric"],
                "pass_condition": "exists",
                "warning_threshold": "missing",
                "source_summary_paths": ["demo_summary.json"],
                "doc_locations": ["README.md"],
                "notes": "",
                "governance": {"mode": "report_only", "tier": "tier0"},
                "checks": [
                    {
                        "summary_stem": "missing_summary",
                        "rowset": "comparison_rows",
                        "match": {"comparison_id": "missing_row"},
                        "pass_if": [{"field": "p_value", "op": "<=", "value": 0.05}],
                    }
                ],
            },
        ],
    }
    summaries = {
        "demo_summary": {
            "comparison_rows": [
                {
                    "comparison_id": "pass_row",
                    "p_value": 0.01,
                    "oriented_diff_ci_low": 1.0,
                    "oriented_mean_diff": 2.0,
                },
                {
                    "comparison_id": "warn_row",
                    "p_value": 0.2,
                    "oriented_diff_ci_low": -0.5,
                    "oriented_mean_diff": 1.0,
                },
                {
                    "comparison_id": "fail_row",
                    "p_value": 0.3,
                    "oriented_diff_ci_low": -1.0,
                    "oriented_mean_diff": -0.5,
                },
            ]
        }
    }

    report = build_claim_drift_report(registry, summaries)
    statuses = {row["claim_id"]: row["status"] for row in report["claim_results"]}

    assert statuses["pass_claim"] == PASS
    assert statuses["warn_claim"] == WARN
    assert statuses["fail_claim"] == FAIL
    assert statuses["missing_claim"] == NOT_EVALUATED
    assert report["overall_gated_status"] == FAIL


def test_run_metadata_contains_manifest_hash_and_environment_snapshot(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"suite_name": "smoke", "entries": [{"entry_id": "demo"}]}),
        encoding="utf-8",
    )

    metadata = build_run_metadata(
        project_root=_project_root(),
        summary_stem="demo_summary",
        output_root=tmp_path,
        manifest_paths=[manifest_path],
        extra={"suite_kind": "test"},
    )

    assert metadata["run_metadata_schema_version"] == 1
    assert metadata["manifest_snapshots"][0]["sha256"]
    assert metadata["python"]["version_info"]["major"] >= 3
    assert "platform" in metadata
    assert "git" in metadata


def test_manifest_loaders_accept_governed_suite_inputs() -> None:
    project_root = _project_root()
    _, external_entries = load_external_manifest(
        project_root / "configs" / "benchmarks" / "external_family_manifest.json"
    )
    suite_manifest, _ = load_suite_manifest(
        project_root / "configs" / "ci" / "baseline_smoke.json"
    )

    assert external_entries
    assert suite_manifest["suite_name"] == "ci_benchmark_smoke"


def test_run_ci_benchmarks_dry_run_writes_plan(tmp_path) -> None:
    project_root = _project_root()
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_ci_benchmarks.py"),
            "--tier",
            "tier3",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads((tmp_path / "ci_benchmark_plan.json").read_text(encoding="utf-8"))

    assert plan["tier"] == "tier3"
    assert plan["cadence"] == "manual / release-candidate"
    assert "external_family_summary.json" in json.dumps(plan, ensure_ascii=False)
    assert "tier3" in completed.stdout
