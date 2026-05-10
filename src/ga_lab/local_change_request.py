from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ga_lab.local_baseline import PROJECT_ROOT as DEFAULT_PROJECT_ROOT
from ga_lab.local_candidate import load_candidate_manifest
from ga_lab.local_candidate_ledger import (
    baseline_change_candidate_for_state,
    lifecycle_state_for_decision_label,
)

ELIGIBLE_DECISION_LABELS = {
    "candidate_passes_local_guard",
    "intentional_baseline_change_required",
}

PROFILE_CHANGE_PATHS = {
    "tsp": "configs/local_profiles/tsp_seeded_swap_local_fast.json",
    "zdt1": "configs/local_profiles/zdt1_diversity_injection_fast.json",
    "knapsack": "configs/local_profiles/knapsack_repair_local_experimental.json",
    "onemax": None,
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
    raw = Path(path)
    candidate = raw if raw.is_absolute() else (project_root / raw)
    try:
        return str(candidate.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(candidate.resolve())


def _change_request_output_dir(
    candidate_id: str,
    *,
    project_root: Path,
    output_root: str | Path,
) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = project_root / root
    safe_candidate_id = "".join(ch.lower() if ch.isalnum() else "_" for ch in candidate_id).strip("_")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return root / f"{timestamp}_{safe_candidate_id}"


def _proposed_files_to_change(problem: str) -> list[str]:
    files = [
        "configs/local_protocols/local_operating_protocols.json",
        "docs/local_candidate_workflow.md",
        "docs/local_change_control.md",
        "docs/local_protocol_guide.md",
        "docs/local_experiment_guide.md",
        "README.md",
        "examples/README.md",
        "artifacts/local_baseline_snapshot.json",
        "artifacts/local_baseline_snapshot.md",
    ]
    profile_path = PROFILE_CHANGE_PATHS.get(problem)
    if profile_path:
        files.insert(0, profile_path)
    return files


def _profile_change_summary(
    report: dict[str, Any],
    manifest: dict[str, Any] | None,
    *,
    force_draft: bool,
) -> str:
    candidate_variant = None
    compared_profiles = report.get("compared_profiles", {})
    if isinstance(compared_profiles, dict):
        candidate_variant = compared_profiles.get("candidate_variant")
    if candidate_variant is None and manifest is not None:
        candidate_variant = manifest.get("candidate_variant")
    problem = str(report.get("problem", ""))
    profile_path = PROFILE_CHANGE_PATHS.get(problem)
    if profile_path is None:
        return "No profile replacement is defined for this control-only branch."
    if force_draft:
        return (
            f"Draft only: if reviewers ever choose to promote `{candidate_variant}` for `{problem}`, "
            f"the manual replacement target would be `{profile_path}` plus the protocol/docs/snapshot follow-up."
        )
    return (
        f"If approved after manual review, replace `{profile_path}` with the candidate-equivalent settings "
        f"for `{candidate_variant}` and refresh the linked protocol/docs/snapshot files."
    )


def _risk_list(report: dict[str, Any], lifecycle_state: str, *, force_draft: bool) -> list[str]:
    risks = []
    if report.get("baseline_status") != "PASS":
        risks.append("Baseline check is not PASS; candidate evidence is not merge-ready.")
    if lifecycle_state not in {"passed_local_guard", "ready_for_change_request"}:
        risks.append(
            "This candidate has not cleared the local promotion gate; the pack is informational only."
        )
    non_regression = report.get("non_regression_results", {})
    if isinstance(non_regression, dict):
        for key, value in non_regression.items():
            if isinstance(value, (int, float)) and value < 0:
                risks.append(f"Non-regression metric `{key}` is still negative ({value}).")
    if force_draft:
        risks.append("Force-draft was used; do not treat this pack as approval for baseline replacement.")
    return risks or ["No additional risks were detected beyond the normal manual review requirement."]


def build_local_baseline_change_request(
    candidate_report_ref: str | Path,
    *,
    project_root: Path | None = None,
    output_root: str | Path = "outputs/local_change_requests",
    force_draft: bool = False,
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT
    candidate_report_path = Path(candidate_report_ref)
    if not candidate_report_path.is_absolute():
        candidate_report_path = (root / candidate_report_path).resolve()
    report = _load_json(candidate_report_path)
    if not isinstance(report, dict):
        raise ValueError("candidate report must be a JSON object")

    current_snapshot_path = root / "artifacts" / "local_baseline_snapshot.json"
    current_snapshot_hash = _sha256(current_snapshot_path)
    candidate_report_snapshot_hash = report.get("baseline_snapshot_hash")

    decision_label = str(report.get("decision_label", ""))
    lifecycle_state = lifecycle_state_for_decision_label(decision_label)
    eligible = (
        decision_label in ELIGIBLE_DECISION_LABELS
        or lifecycle_state == "ready_for_change_request"
        or force_draft
    )
    if not eligible:
        raise ValueError(
            f"candidate report `{decision_label}` is not eligible for a change-request pack without --force-draft"
        )

    manifest = None
    manifest_path = report.get("candidate_source_path")
    if manifest_path:
        try:
            manifest = load_candidate_manifest(manifest_path, project_root=root)
        except Exception:
            manifest = None

    output_dir = _change_request_output_dir(
        str(report.get("candidate_id", "candidate")),
        project_root=root,
        output_root=output_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rerun_candidate_command = (
        f"python scripts/run_local_candidate.py --candidate {_relative_str(manifest_path, project_root=root) or manifest_path}"
    )
    if manifest is not None and manifest.get("existing_output_dir"):
        rerun_candidate_command += " --use-existing-output"

    baseline_change_candidate = baseline_change_candidate_for_state(lifecycle_state)
    draft_only = force_draft and not baseline_change_candidate
    required_commands = [
        "python scripts/check_local_baseline.py",
        f"python scripts/run_local_sweep.py --study {report.get('baseline_guard_study')}",
        rerun_candidate_command,
        "python scripts/summarize_local_candidates.py",
    ]
    if baseline_change_candidate:
        required_commands.extend(
            [
                "python scripts/check_local_baseline.py --write-snapshot",
                "python scripts/check_local_baseline.py",
            ]
        )

    risks = _risk_list(report, lifecycle_state, force_draft=force_draft)
    if candidate_report_snapshot_hash and candidate_report_snapshot_hash != current_snapshot_hash:
        risks.append(
            "The candidate report was produced against an older baseline snapshot hash; refresh review context before any promotion decision."
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_id": report.get("candidate_id"),
        "problem": report.get("problem"),
        "target_id": report.get("target_id"),
        "current_baseline_snapshot_hash": current_snapshot_hash,
        "candidate_report_baseline_snapshot_hash": candidate_report_snapshot_hash,
        "candidate_decision_label": decision_label,
        "candidate_lifecycle_state": lifecycle_state,
        "proposed_files_to_change": _proposed_files_to_change(str(report.get("problem", ""))),
        "proposed_profile_change_summary": _profile_change_summary(
            report,
            manifest,
            force_draft=force_draft,
        ),
        "expected_metric_improvement": report.get("primary_metric_delta", {}),
        "non_regression_evidence": report.get("non_regression_results", {}),
        "stress_slice_evidence": report.get("stress_slice_results", {}),
        "risks": risks,
        "required_manual_review": True,
        "required_commands_before_merge": required_commands,
        "snapshot_update_required": baseline_change_candidate,
        "docs_update_required": baseline_change_candidate,
        "final_decision_placeholder": "pending_manual_review",
        "draft_only": draft_only,
        "force_draft_used": force_draft,
        "baseline_change_candidate": baseline_change_candidate,
        "candidate_manifest_path": _relative_str(manifest_path, project_root=root),
        "candidate_report_path": _relative_str(candidate_report_path, project_root=root),
    }

    change_request_json = output_dir / "change_request.json"
    change_request_md = output_dir / "change_request.md"
    baseline_diff_md = output_dir / "baseline_diff_summary.md"
    followup_md = output_dir / "required_followup_checks.md"

    _write_json(change_request_json, payload)
    change_request_md.write_text(
        "\n".join(
            [
                "# Local Baseline Change Request",
                "",
                f"- Candidate ID: `{payload['candidate_id']}`",
                f"- Problem: `{payload['problem']}`",
                f"- Target: `{payload['target_id']}`",
                f"- Decision label: `{payload['candidate_decision_label']}`",
                f"- Lifecycle state: `{payload['candidate_lifecycle_state']}`",
                f"- Draft only: `{payload['draft_only']}`",
                "",
                "## Proposed Profile Change",
                "",
                payload["proposed_profile_change_summary"],
                "",
                "## Risks",
                "",
                "\n".join(f"- {risk}" for risk in payload["risks"]),
                "",
            ]
        ),
        encoding="utf-8",
    )

    metric_rows = report.get("candidate_vs_baseline_rows", [])
    baseline_diff_md.write_text(
        "\n".join(
            [
                "# Baseline Diff Summary",
                "",
                f"- Candidate report: `{payload['candidate_report_path']}`",
                f"- Baseline snapshot hash: `{payload['current_baseline_snapshot_hash']}`",
                f"- Candidate report baseline hash: `{payload['candidate_report_baseline_snapshot_hash']}`",
                "",
                "## Primary Metric Delta",
                "",
                f"```json\n{json.dumps(payload['expected_metric_improvement'], indent=2, ensure_ascii=False)}\n```",
                "",
                "## Non-Regression Evidence",
                "",
                f"```json\n{json.dumps(payload['non_regression_evidence'], indent=2, ensure_ascii=False)}\n```",
                "",
                "## Stress Slice Evidence",
                "",
                f"```json\n{json.dumps(payload['stress_slice_evidence'], indent=2, ensure_ascii=False)}\n```",
                "",
                "## Candidate vs Baseline Rows",
                "",
                _markdown_table(
                    metric_rows,
                    ["metric", "baseline_value", "candidate_value", "delta"],
                )
                if metric_rows
                else "_No candidate-vs-baseline rows were available._",
                "",
            ]
        ),
        encoding="utf-8",
    )

    followup_md.write_text(
        "\n".join(
            [
                "# Required Follow-Up Checks",
                "",
                "Run these commands before any manual baseline change review:",
                "",
                "\n".join(f"- `{command}`" for command in required_commands),
                "",
                "Manual follow-up reminders:",
                "",
                "- update the relevant docs only after a human baseline decision",
                "- regenerate the baseline snapshot only after approval",
                "- do not treat this draft as an automatic profile replacement",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "generated_at": payload["generated_at"],
        "candidate_id": payload["candidate_id"],
        "decision_label": payload["candidate_decision_label"],
        "lifecycle_state": payload["candidate_lifecycle_state"],
        "draft_only": payload["draft_only"],
        "output_paths": {
            "change_request_json": str(change_request_json.resolve()),
            "change_request_md": str(change_request_md.resolve()),
            "baseline_diff_summary_md": str(baseline_diff_md.resolve()),
            "required_followup_checks_md": str(followup_md.resolve()),
        },
    }
