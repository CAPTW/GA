from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ga_lab.governance.run_metadata import file_sha256

RELEASE_ARTIFACTS_SCHEMA_VERSION = 1
MARKER_TEMPLATE_BEGIN = "<!-- BEGIN AUTO-GENERATED: {name} -->"
MARKER_TEMPLATE_END = "<!-- END AUTO-GENERATED: {name} -->"


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_release_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("manifest_version") != 1:
        raise ValueError("release artifact manifest_version must be 1")
    if not isinstance(manifest.get("solver_rows"), list) or not manifest["solver_rows"]:
        raise ValueError("release artifact manifest requires non-empty solver_rows")
    return manifest


def _utc_timestamp(epoch_seconds: float | None = None) -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() if epoch_seconds is None else epoch_seconds),
    )


def load_summary_payloads(
    project_root: Path,
    summary_paths: dict[str, str],
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for key, rel_path in summary_paths.items():
        path = (project_root / rel_path).resolve()
        if path.exists():
            payloads[key] = load_json(path)
    return payloads


def build_validated_ranges(
    summary_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    for payload in summary_payloads.values():
        scope = payload.get("validated_scope")
        if not isinstance(scope, dict):
            continue
        rows: list[dict[str, str]] = []
        for problem, entry in scope.items():
            if not isinstance(entry, dict):
                continue
            size_key = str(entry.get("size_key", ""))
            sizes = entry.get("sizes", {})
            size_values = ""
            raw_values: list[Any] = []
            if isinstance(sizes, dict):
                raw_values = list(sizes.keys())
            elif isinstance(sizes, (list, tuple, set)):
                raw_values = list(sizes)

            if raw_values:
                numeric_values: list[int] = []
                string_values: list[str] = []
                for value in raw_values:
                    try:
                        numeric_values.append(int(value))
                    except (TypeError, ValueError):
                        string_values.append(str(value))
                ordered_values = [str(value) for value in sorted(numeric_values)]
                ordered_values.extend(sorted(string_values))
                size_values = ", ".join(ordered_values)
            rows.append(
                {
                    "problem": str(problem),
                    "size_key": size_key,
                    "validated_sizes": size_values,
                }
            )
        if rows:
            return rows
    return []


def build_summary_inventory(
    project_root: Path,
    summary_paths: dict[str, str],
    summary_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, rel_path in summary_paths.items():
        path = (project_root / rel_path).resolve()
        payload = summary_payloads.get(key, {})
        stat = path.stat() if path.exists() else None
        rows.append(
            {
                "summary_key": key,
                "path": rel_path,
                "exists": path.exists(),
                "size_bytes": stat.st_size if stat is not None else None,
                "sha256": file_sha256(path) if path.exists() else None,
                "modified_at_utc": _utc_timestamp(stat.st_mtime) if stat is not None else None,
                "summary_version": payload.get("summary_schema_version")
                or payload.get("summary_version"),
                "generated_at_utc": payload.get("generated_at_utc")
                or payload.get("run_metadata", {}).get("generated_at_utc"),
                "has_run_metadata": isinstance(payload.get("run_metadata"), dict),
                "aggregate_rows": len(payload.get("aggregate_rows", []))
                if isinstance(payload.get("aggregate_rows"), list)
                else 0,
                "comparison_rows": len(payload.get("comparison_rows", []))
                if isinstance(payload.get("comparison_rows"), list)
                else 0,
                "claim_rows": len(payload.get("claim_rows", []))
                if isinstance(payload.get("claim_rows"), list)
                else 0,
                "freeze_rows": len(payload.get("freeze_rows", []))
                if isinstance(payload.get("freeze_rows"), list)
                else 0,
            }
        )
    return rows


def build_claim_matrix(
    registry: dict[str, Any],
    drift_report: dict[str, Any],
) -> list[dict[str, Any]]:
    drift_index = {row["claim_id"]: row for row in drift_report.get("claim_results", [])}
    rows: list[dict[str, Any]] = []
    for claim in registry["claims"]:
        drift_row = drift_index.get(
            claim["claim_id"],
            {
                "status": "NOT_EVALUATED",
                "check_results": [],
            },
        )
        latest_reason = " / ".join(
            f"{check['summary_stem']}:{check['status']}"
            for check in drift_row.get("check_results", [])
        )
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "label": claim["label"],
                "status": claim["status"],
                "evidence_scope": claim["evidence_scope"],
                "problem_family": claim["problem_family"],
                "validated_ranges": claim["validated_ranges"],
                "strongest_comparator": claim["comparators"],
                "metrics": claim["metrics"],
                "pass_condition": claim["pass_condition"],
                "warning_threshold": claim["warning_threshold"],
                "source_summary_paths": claim["source_summary_paths"],
                "doc_locations": claim["doc_locations"],
                "governance_mode": claim["governance"]["mode"],
                "governance_tier": claim["governance"].get("tier"),
                "latest_status": drift_row["status"],
                "latest_reason": latest_reason,
                "notes": claim["notes"],
            }
        )
    return rows


def build_solver_matrix(
    registry: dict[str, Any],
    drift_report: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    claim_index = {claim["claim_id"]: claim for claim in registry["claims"]}
    drift_index = {row["claim_id"]: row for row in drift_report.get("claim_results", [])}
    rows: list[dict[str, Any]] = []

    for entry in manifest["solver_rows"]:
        claim = claim_index[entry["claim_id"]]
        drift_row = drift_index.get(entry["claim_id"], {"status": "NOT_EVALUATED"})
        rows.append(
            {
                "row_id": entry["row_id"],
                "claim_id": entry["claim_id"],
                "problem_family": entry["problem_family"],
                "family_or_instance_scope": entry["family_or_instance_scope"],
                "size_tier": entry["size_tier"],
                "objective_priority": entry["objective_priority"],
                "recommended_solver": entry["recommended_solver"],
                "recommended_preset_or_script": entry["recommended_preset_or_script"],
                "status": entry.get("status", claim["status"]),
                "evidence_scope": entry.get("evidence_scope", claim["evidence_scope"]),
                "strongest_comparator": entry.get(
                    "strongest_comparator",
                    " / ".join(claim["comparators"]),
                ),
                "latest_status": drift_row["status"],
                "notes": entry["notes"],
            }
        )
    return rows


def build_release_snapshot(
    project_root: Path,
    registry: dict[str, Any],
    drift_report: dict[str, Any],
    manifest: dict[str, Any],
    claim_matrix: list[dict[str, Any]],
    solver_matrix: list[dict[str, Any]],
    summary_inventory: list[dict[str, Any]],
    validated_ranges: list[dict[str, str]],
) -> dict[str, Any]:
    status_counts = Counter(row["latest_status"] for row in claim_matrix)
    scope_counts = Counter(row["evidence_scope"] for row in claim_matrix)
    class_counts = Counter(row["status"] for row in claim_matrix)
    ci_gated = [row for row in claim_matrix if row["governance_mode"] == "ci_gated"]
    ci_status_counts = Counter(row["latest_status"] for row in ci_gated)
    return {
        "snapshot_schema_version": RELEASE_ARTIFACTS_SCHEMA_VERSION,
        "generated_at_utc": _utc_timestamp(),
        "project_root": str(project_root),
        "registry_path": "claims/claim_registry.json",
        "drift_report_path": "outputs/benchmark_summary/claim_drift_report.json",
        "drift_overall_status": drift_report["overall_gated_status"],
        "drift_counts": dict(drift_report["counts"]),
        "claim_counts_by_latest_status": dict(status_counts),
        "claim_counts_by_evidence_scope": dict(scope_counts),
        "claim_counts_by_classification": dict(class_counts),
        "ci_gated_claims": {
            "total": len(ci_gated),
            "counts": dict(ci_status_counts),
        },
        "validated_ranges": validated_ranges,
        "summary_inventory": summary_inventory,
        "readme_claim_highlight_ids": manifest.get("readme_claim_highlight_ids", []),
        "readme_solver_preview_row_ids": manifest.get("readme_solver_preview_row_ids", []),
        "solver_matrix_row_count": len(solver_matrix),
    }


def _badge_payload(label: str, message: str, color: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label": label,
        "message": message,
        "color": color,
        "details": details,
    }


def build_badges(
    release_snapshot: dict[str, Any],
    claim_matrix: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    ci_status = release_snapshot["drift_overall_status"]
    external_official = [
        row
        for row in claim_matrix
        if row["status"] == "official"
        and row["evidence_scope"] == "external"
        and row["latest_status"] == "PASS"
    ]
    family_conditional = [
        row
        for row in claim_matrix
        if row["evidence_scope"] == "family_conditional" and row["latest_status"] == "PASS"
    ]
    governance_color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(
        ci_status,
        "lightgrey",
    )
    return {
        "governance_status": _badge_payload(
            "governance",
            ci_status.lower(),
            governance_color,
            {
                "ci_gated_counts": release_snapshot["ci_gated_claims"]["counts"],
            },
        ),
        "external_support": _badge_payload(
            "external-support",
            f"{len(external_official)} official / {len(family_conditional)} family",
            "green" if external_official else "yellow",
            {
                "official_external_pass_claims": [row["claim_id"] for row in external_official],
                "family_conditional_pass_claims": [
                    row["claim_id"] for row in family_conditional
                ],
            },
        ),
        "solver_choice_guidance": _badge_payload(
            "solver-choice",
            "scoped-and-governed",
            "green",
            {
                "note": (
                    "Solver guidance is split into internal, external, "
                    "family-conditional, and experimental scopes."
                ),
            },
        ),
        "validated_ranges": _badge_payload(
            "validated-ranges",
            f"{len(release_snapshot['validated_ranges'])} families",
            "blue",
            {
                "ranges": release_snapshot["validated_ranges"],
            },
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _csv_value(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _rows_by_ids(rows: list[dict[str, Any]], key: str, ids: list[str]) -> list[dict[str, Any]]:
    index = {row[key]: row for row in rows}
    return [index[row_id] for row_id in ids if row_id in index]


def _humanize_scope(value: str) -> str:
    return value.replace("_", " ")


def render_benchmark_card(
    release_snapshot: dict[str, Any],
    claim_matrix: list[dict[str, Any]],
) -> str:
    external_rows = [
        row
        for row in claim_matrix
        if row["status"] == "official"
        and row["evidence_scope"] == "external"
        and row["latest_status"] == "PASS"
    ]
    family_rows = [
        row
        for row in claim_matrix
        if row["evidence_scope"] == "family_conditional"
        and row["status"] != "experimental"
        and row["latest_status"] == "PASS"
    ]
    experimental_rows = [
        row for row in claim_matrix if row["status"] == "experimental"
    ]
    internal_only = [
        row
        for row in claim_matrix
        if row["status"] == "official" and row["evidence_scope"] == "internal"
    ]
    lines = [
        "## Benchmark Fairness",
        "",
        "- Primary comparison basis: matched function evaluation budget.",
        "- `configured_budget` is the planned objective-evaluation budget for a solver path.",
        "- `actual_evaluations_used` is the objective-call count actually spent by the run.",
        (
            "- `extra_evaluations_from_hybrid` counts extra objective calls used by local "
            "search or refinement."
        ),
        (
            "- Initial population evaluation and final re-evaluation stay inside the "
            "reported budget policy."
        ),
        "- Wall-clock time is recorded, but it is secondary to matched-budget quality.",
        "",
        "## Validated Ranges",
        "",
    ]
    lines.append(
        _markdown_table(
            ["Problem", "Validated sizes", "Size key"],
            [
                [row["problem"], f"`{row['validated_sizes']}`", f"`{row['size_key']}`"]
                for row in release_snapshot["validated_ranges"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Evidence Snapshot",
            "",
            _markdown_table(
                ["Bucket", "Count"],
                [
                    [
                        "Official",
                        str(
                            release_snapshot["claim_counts_by_classification"].get(
                                "official",
                                0,
                            )
                        ),
                    ],
                    [
                        "Family-conditional",
                        str(
                            release_snapshot["claim_counts_by_classification"].get(
                                "family_conditional",
                                0,
                            )
                        ),
                    ],
                    [
                        "Experimental",
                        str(
                            release_snapshot["claim_counts_by_classification"].get(
                                "experimental",
                                0,
                            )
                        ),
                    ],
                    [
                        "ci-gated PASS",
                        str(release_snapshot["ci_gated_claims"]["counts"].get("PASS", 0)),
                    ],
                ],
            ),
            "",
            "## Strongest Externally Supported Claims",
            "",
            _markdown_table(
                ["Claim", "Comparator", "Latest status"],
                [
                    [
                        row["label"],
                        f"`{' / '.join(row['strongest_comparator'])}`",
                        f"`{row['latest_status']}`",
                    ]
                    for row in external_rows
                ],
            ),
            "",
            "## Family-Conditional Evidence",
            "",
            _markdown_table(
                ["Claim", "Scope", "Latest status"],
                [
                    [
                        row["label"],
                        _humanize_scope(row["problem_family"]),
                        f"`{row['latest_status']}`",
                    ]
                    for row in family_rows
                ],
            ),
            "",
            "## Experimental Only",
            "",
            _markdown_table(
                ["Claim", "Reading", "Latest status"],
                [
                    [row["label"], row["notes"], f"`{row['latest_status']}`"]
                    for row in experimental_rows
                ],
            ),
            "",
            "## Important Limitations",
            "",
            f"- `{len(internal_only)}` official claims remain intentionally internal-only.",
            (
                "- Bitstring and knapsack should be read through family-conditioned "
                "guidance, not broad external defaults."
            ),
            "- `tsp_medium_hybrid.json` remains a narrow internal quality-first path.",
            "- Large-tier ZDT still needs a tradeoff note instead of a scalar one-best claim.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_solver_matrix_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "problem_family",
        "family_or_instance_scope",
        "size_tier",
        "objective_priority",
        "recommended_solver",
        "recommended_preset_or_script",
        "status",
        "evidence_scope",
        "strongest_comparator",
        "latest_status",
        "notes",
    ]
    table_rows = [
        [
            row["problem_family"],
            row["family_or_instance_scope"],
            row["size_tier"],
            row["objective_priority"],
            f"`{row['recommended_solver']}`",
            f"`{row['recommended_preset_or_script']}`",
            f"`{row['status']}`",
            f"`{row['evidence_scope']}`",
            f"`{row['strongest_comparator']}`",
            f"`{row['latest_status']}`",
            row["notes"],
        ]
        for row in rows
    ]
    return _markdown_table(headers, table_rows) + "\n"


def render_release_status_markdown(
    release_snapshot: dict[str, Any],
    claim_matrix: list[dict[str, Any]],
) -> str:
    ci_rows = [row for row in claim_matrix if row["governance_mode"] == "ci_gated"]
    report_rows = [row for row in claim_matrix if row["governance_mode"] == "report_only"]
    lines = [
        "## Current Drift Status",
        "",
        f"- overall ci-gated status: `{release_snapshot['drift_overall_status']}`",
        f"- ci-gated claims: `{release_snapshot['ci_gated_claims']['total']}`",
        "",
        "## CI-Gated Claims",
        "",
        _markdown_table(
            ["claim_id", "status", "scope", "tier"],
            [
                [
                    f"`{row['claim_id']}`",
                    f"`{row['latest_status']}`",
                    f"`{row['evidence_scope']}`",
                    f"`{row['governance_tier']}`",
                ]
                for row in ci_rows
            ],
        ),
        "",
        "## Report-Only Claims",
        "",
        _markdown_table(
            ["claim_id", "status", "scope"],
            [
                [
                    f"`{row['claim_id']}`",
                    f"`{row['latest_status']}`",
                    f"`{row['evidence_scope']}`",
                ]
                for row in report_rows
            ],
        ),
        "",
        "## Latest Evidence Snapshot",
        "",
        _markdown_table(
            ["summary", "exists", "version", "generated_at", "modified_at"],
            [
                [
                    f"`{row['summary_key']}`",
                    f"`{'yes' if row['exists'] else 'no'}`",
                    f"`{row['summary_version'] or 'n/a'}`",
                    f"`{row['generated_at_utc'] or 'n/a'}`",
                    f"`{row['modified_at_utc'] or 'n/a'}`",
                ]
                for row in release_snapshot["summary_inventory"]
            ],
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_readme_evidence_snapshot(
    release_snapshot: dict[str, Any],
    claim_matrix: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    highlight_rows = _rows_by_ids(
        claim_matrix,
        "claim_id",
        manifest.get("readme_claim_highlight_ids", []),
    )
    lines = [
        f"- drift-governed ci status: `{release_snapshot['drift_overall_status']}`",
        (
            f"- ci-gated claims passing: "
            f"`{release_snapshot['ci_gated_claims']['counts'].get('PASS', 0)}` / "
            f"`{release_snapshot['ci_gated_claims']['total']}`"
        ),
        "- strongest current readings:",
    ]
    for row in highlight_rows:
        lines.append(
            f"  - `{row['claim_id']}`: {row['label']} (`{row['latest_status']}`)"
        )
    return "\n".join(lines) + "\n"


def render_readme_solver_preview(
    solver_matrix: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    preview_rows = _rows_by_ids(
        solver_matrix,
        "row_id",
        manifest.get("readme_solver_preview_row_ids", []),
    )
    return _markdown_table(
        ["Problem family", "Priority", "Recommended solver", "Scope", "Latest status"],
        [
            [
                row["problem_family"],
                row["objective_priority"],
                f"`{row['recommended_solver']}`",
                f"`{row['evidence_scope']}`",
                f"`{row['latest_status']}`",
            ]
            for row in preview_rows
        ],
    ) + "\n"


def render_readme_release_status(release_snapshot: dict[str, Any]) -> str:
    lines = [
        f"- overall drift status: `{release_snapshot['drift_overall_status']}`",
        (
            f"- ci-gated PASS / WARN / FAIL / NOT_EVALUATED: "
            f"`{release_snapshot['ci_gated_claims']['counts'].get('PASS', 0)}` / "
            f"`{release_snapshot['ci_gated_claims']['counts'].get('WARN', 0)}` / "
            f"`{release_snapshot['ci_gated_claims']['counts'].get('FAIL', 0)}` / "
            f"`{release_snapshot['ci_gated_claims']['counts'].get('NOT_EVALUATED', 0)}`"
        ),
        (
            f"- latest release snapshot: "
            f"`{release_snapshot['generated_at_utc']}`"
        ),
    ]
    return "\n".join(lines) + "\n"


def replace_marker_block(text: str, marker_name: str, content: str) -> str:
    begin = MARKER_TEMPLATE_BEGIN.format(name=marker_name)
    end = MARKER_TEMPLATE_END.format(name=marker_name)
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    replacement = f"{begin}\n{content.rstrip()}\n{end}"
    if not pattern.search(text):
        raise ValueError(f"Marker block `{marker_name}` not found")
    return pattern.sub(replacement, text)


def update_markdown_file(path: Path, replacements: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    for marker_name, content in replacements.items():
        text = replace_marker_block(text, marker_name, content)
    path.write_text(text, encoding="utf-8")
