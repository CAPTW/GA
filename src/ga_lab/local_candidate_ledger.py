from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ga_lab.local_baseline import PROJECT_ROOT as DEFAULT_PROJECT_ROOT

DECISION_LABEL_TO_LIFECYCLE_STATE = {
    "reject_regression": "rejected",
    "reject_no_material_gain": "rejected",
    "note_only_stress_slice": "note_only",
    "monitor_only": "monitor",
    "candidate_promising_needs_confirm": "promising_needs_confirm",
    "candidate_passes_local_guard": "passed_local_guard",
    "candidate_requires_new_mechanism_hypothesis": "requires_new_mechanism",
    "baseline_drift_detected": "blocked_by_baseline_drift",
    "intentional_baseline_change_required": "ready_for_change_request",
}

READY_FOR_CHANGE_REQUEST_STATES = {
    "passed_local_guard",
    "ready_for_change_request",
}

LIFECYCLE_STATE_ORDER = [
    "rejected",
    "note_only",
    "monitor",
    "promising_needs_confirm",
    "passed_local_guard",
    "blocked_by_baseline_drift",
    "requires_new_mechanism",
    "ready_for_change_request",
    "archived",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _relative_str(path: str | Path | None, *, project_root: Path) -> str | None:
    if path in {None, ""}:
        return None
    raw_path = Path(path)
    candidate = raw_path if raw_path.is_absolute() else (project_root / raw_path)
    try:
        return str(candidate.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(candidate.resolve())


def _json_compact(value: Any) -> str:
    if value in (None, "", {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def lifecycle_state_for_decision_label(decision_label: str) -> str:
    return DECISION_LABEL_TO_LIFECYCLE_STATE.get(decision_label, "archived")


def baseline_change_candidate_for_state(lifecycle_state: str) -> bool:
    return lifecycle_state in READY_FOR_CHANGE_REQUEST_STATES


def reviewer_note_required_for_state(lifecycle_state: str) -> bool:
    return lifecycle_state not in {"rejected", "archived"}


def discover_candidate_report_paths(
    *,
    project_root: Path | None = None,
    reports_root: str | Path = "outputs/local_candidates",
) -> list[Path]:
    root = project_root or DEFAULT_PROJECT_ROOT
    reports_dir = Path(reports_root)
    if not reports_dir.is_absolute():
        reports_dir = root / reports_dir
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob("**/candidate_report.json"))


def _ledger_row_from_report(report: dict[str, Any], report_path: Path, *, project_root: Path) -> dict[str, Any]:
    decision_label = str(report.get("decision_label", ""))
    lifecycle_state = lifecycle_state_for_decision_label(decision_label)
    candidate_manifest_path = _relative_str(report.get("candidate_source_path"), project_root=project_root)
    candidate_output_dir = _relative_str(report.get("candidate_output_dir"), project_root=project_root)
    return {
        "candidate_id": report.get("candidate_id"),
        "problem": report.get("problem"),
        "target_id": report.get("target_id"),
        "hypothesis_id": report.get("hypothesis_id"),
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_report_path": _relative_str(report_path, project_root=project_root),
        "baseline_snapshot_hash": report.get("baseline_snapshot_hash"),
        "baseline_status": report.get("baseline_status"),
        "decision_label": decision_label,
        "lifecycle_state": lifecycle_state,
        "primary_metric_delta": report.get("primary_metric_delta", {}),
        "non_regression_summary": report.get("non_regression_results", {}),
        "stress_slice_summary": report.get("stress_slice_results", {}),
        "actual_evaluations_used": report.get("actual_evaluations_used"),
        "runtime_seconds": report.get("runtime_seconds"),
        "recommended_next_action": report.get("recommended_next_action"),
        "reviewer_note_required": reviewer_note_required_for_state(lifecycle_state),
        "baseline_change_candidate": baseline_change_candidate_for_state(lifecycle_state),
        "report_timestamp": report.get("generated_at"),
        "evidence_source": candidate_output_dir or _relative_str(report_path, project_root=project_root),
        "decision_matches_expectation": report.get("decision_matches_expectation"),
    }


def build_candidate_ledger(
    *,
    project_root: Path | None = None,
    reports_root: str | Path = "outputs/local_candidates",
    ledger_json_path: str | Path = "artifacts/local_candidate_ledger.json",
    ledger_csv_path: str | Path = "artifacts/local_candidate_ledger.csv",
    ledger_md_path: str | Path = "artifacts/local_candidate_ledger.md",
    summary_json_path: str | Path = "artifacts/local_candidate_summary.json",
    summary_md_path: str | Path = "artifacts/local_candidate_summary.md",
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT
    report_paths = discover_candidate_report_paths(project_root=root, reports_root=reports_root)
    rows = []
    for report_path in report_paths:
        report = _load_json(report_path)
        if not isinstance(report, dict):
            raise ValueError(f"candidate report must be a JSON object: {report_path}")
        rows.append(_ledger_row_from_report(report, report_path, project_root=root))

    rows.sort(key=lambda row: (str(row.get("report_timestamp") or ""), str(row.get("candidate_id") or "")))

    ledger_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "reports_root": str((root / reports_root).resolve()) if not Path(reports_root).is_absolute() else str(Path(reports_root).resolve()),
        "rows": rows,
    }

    counts_by_problem = Counter(str(row.get("problem", "")) for row in rows)
    counts_by_decision = Counter(str(row.get("decision_label", "")) for row in rows)
    counts_by_lifecycle = Counter(str(row.get("lifecycle_state", "")) for row in rows)

    target_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target_buckets[(str(row.get("problem", "")), str(row.get("target_id", "")))].append(row)

    target_summary_rows = []
    for (problem, target_id), bucket in sorted(target_buckets.items()):
        latest = sorted(bucket, key=lambda row: str(row.get("report_timestamp", "")))[-1]
        target_summary_rows.append(
            {
                "problem": problem,
                "target_id": target_id,
                "candidate_count": len(bucket),
                "latest_candidate_id": latest.get("candidate_id"),
                "latest_decision_label": latest.get("decision_label"),
                "latest_lifecycle_state": latest.get("lifecycle_state"),
                "ready_for_change_request_count": sum(1 for row in bucket if row.get("lifecycle_state") == "ready_for_change_request"),
                "passed_local_guard_count": sum(1 for row in bucket if row.get("lifecycle_state") == "passed_local_guard"),
                "note_only_count": sum(1 for row in bucket if row.get("lifecycle_state") == "note_only"),
                "rejected_count": sum(1 for row in bucket if row.get("lifecycle_state") == "rejected"),
            }
        )

    summary_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_candidates": len(rows),
        "counts_by_problem": dict(sorted(counts_by_problem.items())),
        "counts_by_decision_label": dict(sorted(counts_by_decision.items())),
        "counts_by_lifecycle_state": {
            state: counts_by_lifecycle.get(state, 0)
            for state in LIFECYCLE_STATE_ORDER
            if counts_by_lifecycle.get(state, 0)
        },
        "targets": target_summary_rows,
        "ready_for_change_request_candidates": [
            row["candidate_id"] for row in rows if row["lifecycle_state"] == "ready_for_change_request"
        ],
        "passed_local_guard_candidates": [
            row["candidate_id"] for row in rows if row["lifecycle_state"] == "passed_local_guard"
        ],
    }

    ledger_json = Path(ledger_json_path)
    if not ledger_json.is_absolute():
        ledger_json = root / ledger_json
    ledger_csv = Path(ledger_csv_path)
    if not ledger_csv.is_absolute():
        ledger_csv = root / ledger_csv
    ledger_md = Path(ledger_md_path)
    if not ledger_md.is_absolute():
        ledger_md = root / ledger_md
    summary_json = Path(summary_json_path)
    if not summary_json.is_absolute():
        summary_json = root / summary_json
    summary_md = Path(summary_md_path)
    if not summary_md.is_absolute():
        summary_md = root / summary_md

    _write_json(ledger_json, ledger_payload)
    csv_rows = [
        {
            **row,
            "primary_metric_delta": _json_compact(row["primary_metric_delta"]),
            "non_regression_summary": _json_compact(row["non_regression_summary"]),
            "stress_slice_summary": _json_compact(row["stress_slice_summary"]),
        }
        for row in rows
    ]
    fieldnames = [
        "candidate_id",
        "problem",
        "target_id",
        "hypothesis_id",
        "candidate_manifest_path",
        "candidate_report_path",
        "baseline_snapshot_hash",
        "baseline_status",
        "decision_label",
        "lifecycle_state",
        "primary_metric_delta",
        "non_regression_summary",
        "stress_slice_summary",
        "actual_evaluations_used",
        "runtime_seconds",
        "recommended_next_action",
        "reviewer_note_required",
        "baseline_change_candidate",
        "report_timestamp",
        "evidence_source",
        "decision_matches_expectation",
    ]
    _write_rows_csv(ledger_csv, csv_rows, fieldnames)
    ledger_md.write_text(
        "\n".join(
            [
                "# Local Candidate Ledger",
                "",
                f"- Generated at: `{ledger_payload['generated_at']}`",
                f"- Total candidates: `{len(rows)}`",
                "",
                _markdown_table(
                    [
                        {
                            "candidate_id": row["candidate_id"],
                            "problem": row["problem"],
                            "target_id": row["target_id"],
                            "decision_label": row["decision_label"],
                            "lifecycle_state": row["lifecycle_state"],
                            "baseline_change_candidate": row["baseline_change_candidate"],
                            "recommended_next_action": row["recommended_next_action"],
                        }
                        for row in rows
                    ],
                    [
                        "candidate_id",
                        "problem",
                        "target_id",
                        "decision_label",
                        "lifecycle_state",
                        "baseline_change_candidate",
                        "recommended_next_action",
                    ],
                )
                if rows
                else "_No candidate reports found._",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_json(summary_json, summary_payload)
    summary_md.write_text(
        "\n".join(
            [
                "# Local Candidate Summary",
                "",
                f"- Generated at: `{summary_payload['generated_at']}`",
                f"- Total candidates: `{summary_payload['total_candidates']}`",
                "",
                "## Counts By Problem",
                "",
                _markdown_table(
                    [{"problem": key, "count": value} for key, value in summary_payload["counts_by_problem"].items()],
                    ["problem", "count"],
                )
                if summary_payload["counts_by_problem"]
                else "_No candidate reports found._",
                "",
                "## Counts By Decision Label",
                "",
                _markdown_table(
                    [{"decision_label": key, "count": value} for key, value in summary_payload["counts_by_decision_label"].items()],
                    ["decision_label", "count"],
                )
                if summary_payload["counts_by_decision_label"]
                else "_No candidate reports found._",
                "",
                "## Counts By Lifecycle State",
                "",
                _markdown_table(
                    [{"lifecycle_state": key, "count": value} for key, value in summary_payload["counts_by_lifecycle_state"].items()],
                    ["lifecycle_state", "count"],
                )
                if summary_payload["counts_by_lifecycle_state"]
                else "_No candidate reports found._",
                "",
                "## Target Overview",
                "",
                _markdown_table(
                    target_summary_rows,
                    [
                        "problem",
                        "target_id",
                        "candidate_count",
                        "latest_candidate_id",
                        "latest_decision_label",
                        "latest_lifecycle_state",
                    ],
                )
                if target_summary_rows
                else "_No target summaries available._",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "ledger": ledger_payload,
        "summary": summary_payload,
        "output_paths": {
            "ledger_json": str(ledger_json.resolve()),
            "ledger_csv": str(ledger_csv.resolve()),
            "ledger_md": str(ledger_md.resolve()),
            "summary_json": str(summary_json.resolve()),
            "summary_md": str(summary_md.resolve()),
        },
    }
