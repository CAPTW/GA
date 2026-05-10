from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DRIFT_REPORT_SCHEMA_VERSION = 1
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
NOT_EVALUATED = "NOT_EVALUATED"


def _condition_matches(value: Any, operator: str, expected: Any) -> bool:
    if value is None:
        return False
    if operator == ">":
        return value > expected
    if operator == ">=":
        return value >= expected
    if operator == "<":
        return value < expected
    if operator == "<=":
        return value <= expected
    if operator == "==":
        return value == expected
    if operator == "!=":
        return value != expected
    if operator == "contains":
        return str(expected) in str(value)
    if operator == "in":
        return value in expected
    raise ValueError(f"Unsupported operator: {operator}")


def _conditions_pass(
    row: dict[str, Any],
    conditions: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for condition in conditions:
        field = condition["field"]
        operator = condition["op"]
        expected = condition["value"]
        actual = row.get(field)
        passed = _condition_matches(actual, operator, expected)
        reasons.append(
            f"{field}={actual!r} {operator} {expected!r} -> {'ok' if passed else 'miss'}"
        )
        if not passed:
            return False, reasons
    return True, reasons


def load_summary_payloads(paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        payloads[path.stem] = payload
    return payloads


def _find_row(summary: dict[str, Any], rowset: str, match: dict[str, Any]) -> dict[str, Any] | None:
    rows = summary.get(rowset, [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if all(row.get(key) == value for key, value in match.items()):
            return row
    return None


def _evaluate_check(
    *,
    claim_id: str,
    check: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary_stem = str(check["summary_stem"])
    summary = summaries.get(summary_stem)
    if summary is None:
        return {
            "claim_id": claim_id,
            "summary_stem": summary_stem,
            "status": NOT_EVALUATED,
            "reason": f"summary `{summary_stem}` not loaded",
        }

    row = _find_row(summary, str(check["rowset"]), check["match"])
    if row is None:
        return {
            "claim_id": claim_id,
            "summary_stem": summary_stem,
            "status": NOT_EVALUATED,
            "reason": f"row not found in `{summary_stem}` for match {check['match']}",
        }

    pass_ok, pass_reasons = _conditions_pass(row, check["pass_if"])
    if pass_ok:
        return {
            "claim_id": claim_id,
            "summary_stem": summary_stem,
            "status": PASS,
            "reason": "; ".join(pass_reasons),
            "row_excerpt": {key: row.get(key) for key in check.get("report_fields", [])},
        }

    warn_if = check.get("warn_if", [])
    if warn_if:
        warn_ok, warn_reasons = _conditions_pass(row, warn_if)
        if warn_ok:
            return {
                "claim_id": claim_id,
                "summary_stem": summary_stem,
                "status": WARN,
                "reason": "; ".join(pass_reasons + warn_reasons),
                "row_excerpt": {key: row.get(key) for key in check.get("report_fields", [])},
            }

    return {
        "claim_id": claim_id,
        "summary_stem": summary_stem,
        "status": FAIL,
        "reason": "; ".join(pass_reasons),
        "row_excerpt": {key: row.get(key) for key in check.get("report_fields", [])},
    }


def _aggregate_claim_status(check_statuses: list[str]) -> str:
    if any(status == FAIL for status in check_statuses):
        return FAIL
    if any(status == WARN for status in check_statuses):
        return WARN
    evaluated = [status for status in check_statuses if status != NOT_EVALUATED]
    if not evaluated:
        return NOT_EVALUATED
    if len(evaluated) != len(check_statuses):
        return WARN
    return PASS


def build_claim_drift_report(
    registry: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    *,
    reference_summary_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    claim_results: list[dict[str, Any]] = []
    overall_gated_status = PASS
    for claim in registry["claims"]:
        checks = claim.get("checks", [])
        check_results = [
            _evaluate_check(claim_id=claim["claim_id"], check=check, summaries=summaries)
            for check in checks
        ]
        claim_status = _aggregate_claim_status([result["status"] for result in check_results])
        governance_mode = claim["governance"]["mode"]
        if governance_mode == "ci_gated":
            if claim_status == FAIL:
                overall_gated_status = FAIL
            elif claim_status in {WARN, NOT_EVALUATED} and overall_gated_status != FAIL:
                overall_gated_status = WARN
        claim_results.append(
            {
                "claim_id": claim["claim_id"],
                "label": claim["label"],
                "status": claim_status,
                "scope": claim["evidence_scope"],
                "classification": claim["status"],
                "governance_mode": governance_mode,
                "tier": claim["governance"].get("tier"),
                "doc_locations": claim["doc_locations"],
                "source_summary_paths": claim["source_summary_paths"],
                "check_results": check_results,
            }
        )

    counts = {
        PASS: sum(1 for result in claim_results if result["status"] == PASS),
        WARN: sum(1 for result in claim_results if result["status"] == WARN),
        FAIL: sum(1 for result in claim_results if result["status"] == FAIL),
        NOT_EVALUATED: sum(1 for result in claim_results if result["status"] == NOT_EVALUATED),
    }
    return {
        "report_schema_version": DRIFT_REPORT_SCHEMA_VERSION,
        "registry_schema_version": registry["registry_schema_version"],
        "overall_gated_status": overall_gated_status,
        "reference_summary_paths": [
            str(Path(path).resolve()) for path in (reference_summary_paths or [])
        ],
        "loaded_summary_stems": sorted(summaries),
        "counts": counts,
        "claim_results": claim_results,
    }


def markdown_claim_drift_report(report: dict[str, Any]) -> str:
    lines = [
        "# Claim Drift Report",
        "",
        f"- overall gated status: `{report['overall_gated_status']}`",
        f"- loaded summary stems: `{', '.join(report['loaded_summary_stems'])}`",
        (
            f"- pass / warn / fail / not evaluated: `{report['counts'][PASS]}` / "
            f"`{report['counts'][WARN]}` / `{report['counts'][FAIL]}` / "
            f"`{report['counts'][NOT_EVALUATED]}`"
        ),
        "",
        "| claim_id | status | scope | governance | tier |",
        "| --- | --- | --- | --- | --- |",
    ]
    for claim in report["claim_results"]:
        lines.append(
            f"| `{claim['claim_id']}` | `{claim['status']}` | `{claim['scope']}` | "
            f"`{claim['governance_mode']}` | `{claim['tier'] or ''}` |"
        )

    focus_rows = [
        claim for claim in report["claim_results"] if claim["status"] in {WARN, FAIL, NOT_EVALUATED}
    ]
    if focus_rows:
        lines.extend(["", "## Follow-up", ""])
        for claim in focus_rows:
            lines.append(f"### `{claim['claim_id']}`")
            lines.append("")
            lines.append(f"- status: `{claim['status']}`")
            lines.append(f"- docs to review: `{', '.join(claim['doc_locations'])}`")
            for check in claim["check_results"]:
                lines.append(f"- {check['summary_stem']}: {check['status']} - {check['reason']}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
