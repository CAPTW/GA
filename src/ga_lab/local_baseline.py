from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROFILE_PATHS = {
    "tsp_q": "configs/local_profiles/tsp_seeded_swap_local.json",
    "tsp_f": "configs/local_profiles/tsp_seeded_swap_local_fast.json",
    "zdt1_q": "configs/local_profiles/zdt1_diversity_injection.json",
    "zdt1_f": "configs/local_profiles/zdt1_diversity_injection_fast.json",
    "knapsack_note": "configs/local_profiles/knapsack_repair_local_experimental.json",
}

DOC_PATHS = (
    "README.md",
    "docs/local_experiment_guide.md",
    "docs/local_protocol_guide.md",
    "docs/local_candidate_workflow.md",
    "docs/local_change_control.md",
    "docs/local_reopen_criteria.md",
    "examples/README.md",
    "configs/local_protocols/local_operating_protocols.json",
)

CANDIDATE_SCHEMA_PATH = "configs/local_candidates/candidate_schema.json"

GUARD_STUDIES = {
    "tsp": "local_baseline_guard_tsp",
    "zdt1": "local_baseline_guard_zdt1",
    "knapsack": "local_baseline_guard_knapsack",
    "onemax": "local_baseline_guard_onemax",
}

TARGET_STATUS_OVERRIDES: dict[str, dict[str, Any]] = {
    "tsp_fast_anti_case_tail": {
        "latest_decision": "freeze_as_protocol_limitation",
        "keep_as_regression_case": True,
        "next_recommended_action": "reopen_only_with_new_mechanism_hypothesis",
        "recommended_next_action": "reopen_only_with_new_mechanism_hypothesis",
    },
    "tsp_rescue_target_ambiguity": {
        "latest_decision": "secondary_regression_slice",
        "keep_as_regression_case": True,
        "next_recommended_action": "keep_secondary_regression_slice",
        "recommended_next_action": "keep_secondary_regression_slice",
    },
    "zdt1_fast_spread_safety_fail": {
        "latest_decision": "note_only_stress_slice",
        "keep_as_regression_case": True,
        "next_recommended_action": "only_validate_if_candidate_generalizes_beyond_stress_slice",
        "recommended_next_action": "only_validate_if_candidate_generalizes_beyond_stress_slice",
    },
    "zdt1_fast_joint_safety_fail": {
        "latest_decision": "monitor_only",
        "keep_as_regression_case": True,
        "next_recommended_action": "use_Q_for_final_safety",
        "recommended_next_action": "use_Q_for_final_safety",
    },
    "knapsack_repair_boundary_subset_sum_tight_capacity": {
        "latest_decision": "narrow_note_only",
        "keep_as_regression_case": True,
        "next_recommended_action": "keep_narrow_note_on_subset_sum_tight_capacity_only",
        "recommended_next_action": "keep_narrow_note_on_subset_sum_tight_capacity_only",
    },
    "onemax_no_active_target": {
        "latest_decision": "no_active_target",
        "keep_as_regression_case": False,
        "next_recommended_action": "no_active_target",
        "recommended_next_action": "no_active_target",
    },
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows_to_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
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


def _latest_study_dir(project_root: Path, suffix: str) -> Path | None:
    matches = sorted((project_root / "outputs" / "local_studies").glob(f"*_{suffix}"))
    return matches[-1] if matches else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _current_problem_decisions() -> dict[str, dict[str, Any]]:
    return {
        "tsp": {
            "q_profile": PROFILE_PATHS["tsp_q"],
            "f_profile": PROFILE_PATHS["tsp_f"],
            "f_role": "budget-first / exploratory",
            "final_rule": "anti-case/corridor suspicion or quality-sensitive final -> Q 8-10",
            "target_status": "tsp_fast_anti_case_tail = freeze_as_protocol_limitation",
        },
        "zdt1": {
            "q_profile": PROFILE_PATHS["zdt1_q"],
            "f_profile": PROFILE_PATHS["zdt1_f"],
            "f_role": "exploratory / budget-first",
            "final_rule": "final safety -> Q",
            "spread_candidate_status": "spread_pg_pop41_gen88 = note_only_stress_slice",
            "joint_target_status": "monitor_only + Q final safety",
        },
        "knapsack": {
            "default_rule": "no broad default",
            "repair_note": "repair_only narrow note only",
            "scope": "subset-sum/tight-capacity-like only",
        },
        "onemax": {
            "default_rule": "none control only",
            "target_status": "no_active_target",
        },
    }


def _candidate_admission_rules() -> dict[str, dict[str, Any]]:
    return {
        "tsp": {
            "current_baseline_profiles": {
                "quality_first": PROFILE_PATHS["tsp_q"],
                "budget_first": PROFILE_PATHS["tsp_f"],
            },
            "frozen_decision": "budget-first F stays exploratory; anti-case tail is frozen as a protocol limitation",
            "allowed_candidate_type": (
                "new_mechanism_hypothesis only; same-budget PG contour reruns without a new mechanism stay note/reject"
            ),
            "promotion_threshold": (
                "Only consider a same-name replacement if a new mechanism hypothesis clearly lowers anti-case p95/max without materially worsening rescue-target mean or budget."
            ),
            "note_only_threshold": (
                "Stress-slice-only gain may be documented, but it does not reopen the frozen fast default by itself."
            ),
            "rejection_condition": (
                "Reject if anti-case p95/max worsens, rescue-target mean worsens materially, or no new mechanism hypothesis is present."
            ),
        },
        "zdt1": {
            "current_baseline_profiles": {
                "quality_first": PROFILE_PATHS["zdt1_q"],
                "budget_first": PROFILE_PATHS["zdt1_f"],
            },
            "frozen_decision": "budget-first F stays exploratory; final safety stays on Q",
            "allowed_candidate_type": (
                "same-budget spread/joint safety candidates tied to a pinned stress target"
            ),
            "promotion_threshold": (
                "A candidate must improve spread-tail or fail rate, preserve HV mean/p90, avoid pareto or joint regressions, and hold on stable/normal slices."
            ),
            "note_only_threshold": (
                "If a candidate only helps the spread-stress slice while stable/normal non-regression stays weak, keep it as note_only_stress_slice."
            ),
            "rejection_condition": (
                "Reject if stable/normal slices regress, HV p90 tail worsens materially, pareto_ratio drops, or joint safety gets worse."
            ),
        },
        "knapsack": {
            "current_baseline_profiles": {
                "default": "greedy_local_search",
                "narrow_note": PROFILE_PATHS["knapsack_note"],
            },
            "frozen_decision": "no broad default; keep repair_only as a narrow family-conditioned note only",
            "allowed_candidate_type": (
                "family-conditioned subset-sum/tight-capacity repair note checks only"
            ),
            "promotion_threshold": "Broad default promotion is intentionally forbidden.",
            "note_only_threshold": (
                "Keep a candidate only when it improves the narrow family-conditioned slice without claiming a broad default."
            ),
            "rejection_condition": (
                "Reject if the candidate implies a broad default, loses feasible quality to greedy, or is not family-conditioned."
            ),
        },
        "onemax": {
            "current_baseline_profiles": {
                "control": "none",
            },
            "frozen_decision": "control-only baseline; no active candidate target",
            "allowed_candidate_type": "control drift checks only",
            "promotion_threshold": (
                "Only a clearly simpler-or-equal control variant with obvious eval savings would matter, and none is active."
            ),
            "note_only_threshold": "Any non-default candidate stays monitor-only unless it cleanly simplifies the control.",
            "rejection_condition": "Reject anything more complex than the frozen none-control path without a clear control gain.",
        },
    }


def _representative_metrics(project_root: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    tsp_dir = _latest_study_dir(project_root, "tsp_tail_freeze_recheck")
    if tsp_dir is not None:
        rows = _read_csv_rows(tsp_dir / "tsp_fast_tail_summary.csv")
        summary = {row["case_group"]: row for row in rows}
        metrics["tsp"] = {
            "study_dir": str(tsp_dir.relative_to(project_root)),
            "overall": summary.get("overall", {}),
            "anti_case": summary.get("anti_case", {}),
            "rescue_target": summary.get("rescue_target", {}),
        }

    zdt1_dir = _latest_study_dir(project_root, "zdt1_spread_candidate_boundary_confirm")
    if zdt1_dir is not None:
        rows = _read_csv_rows(zdt1_dir / "zdt1_spread_candidate_boundary_table.csv")
        metrics["zdt1"] = {
            "study_dir": str(zdt1_dir.relative_to(project_root)),
            "boundary_rows": rows,
        }

    knapsack_dir = _latest_study_dir(project_root, "knapsack_stress_freeze_check")
    if knapsack_dir is not None:
        metrics["knapsack"] = {
            "study_dir": str(knapsack_dir.relative_to(project_root)),
            "summary_rows": _read_csv_rows(knapsack_dir / "summary.csv"),
        }

    onemax_dir = _latest_study_dir(project_root, "onemax_control_stress_freeze_check")
    if onemax_dir is not None:
        metrics["onemax"] = {
            "study_dir": str(onemax_dir.relative_to(project_root)),
            "summary_rows": _read_csv_rows(onemax_dir / "summary.csv"),
        }

    return metrics


def freeze_target_registry_statuses(
    project_root: Path | None = None,
    *,
    write_outputs: bool = True,
) -> list[dict[str, Any]]:
    root = project_root or PROJECT_ROOT
    json_path = root / "outputs" / "local_studies" / "future_optimization_targets.json"
    csv_path = root / "outputs" / "local_studies" / "future_optimization_targets.csv"
    md_path = root / "outputs" / "local_studies" / "future_optimization_targets.md"

    rows = _load_json(json_path)
    if not isinstance(rows, list):
        raise ValueError("future_optimization_targets.json must contain a list")

    for row in rows:
        target_id = str(row.get("target_id", ""))
        overrides = TARGET_STATUS_OVERRIDES.get(target_id)
        if overrides is not None:
            row.update(overrides)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    if write_outputs:
        _write_json(json_path, rows)
        _rows_to_csv(csv_path, rows, fieldnames)
        md_columns = [
            "target_id",
            "problem",
            "latest_decision",
            "current_severity",
            "next_recommended_action",
            "keep_as_regression_case",
        ]
        md_body = [
            "# Future Optimization Targets",
            "",
            _markdown_table(rows, md_columns),
            "",
        ]
        md_path.write_text("\n".join(md_body), encoding="utf-8")
    return rows


def build_local_baseline_snapshot(
    project_root: Path | None = None,
    snapshot_path: Path | None = None,
    markdown_path: Path | None = None,
) -> dict[str, Any]:
    root = project_root or PROJECT_ROOT
    snapshot_path = snapshot_path or (root / "artifacts" / "local_baseline_snapshot.json")
    markdown_path = markdown_path or (root / "artifacts" / "local_baseline_snapshot.md")

    targets = freeze_target_registry_statuses(root)
    profile_hashes = {
        key: _sha256(root / rel_path)
        for key, rel_path in PROFILE_PATHS.items()
    }
    doc_hashes = {
        rel_path: _sha256(root / rel_path)
        for rel_path in DOC_PATHS
    }

    snapshot = {
        "generated_at": datetime.now(UTC).isoformat(),
        "profile_paths": PROFILE_PATHS,
        "profile_content_hashes": profile_hashes,
        "protocol_matrix_hash": _sha256(root / "configs" / "local_protocols" / "local_operating_protocols.json"),
        "stress_target_registry_hash": _sha256(root / "outputs" / "local_studies" / "future_optimization_targets.json"),
        "failure_hypothesis_registry_hash": _sha256(root / "outputs" / "local_studies" / "failure_hypotheses.json"),
        "candidate_schema_hash": _sha256(root / CANDIDATE_SCHEMA_PATH),
        "doc_hashes": doc_hashes,
        "problem_operating_decisions": _current_problem_decisions(),
        "candidate_admission_rules": _candidate_admission_rules(),
        "frozen_target_decisions": {
            row["target_id"]: {
                "latest_decision": row.get("latest_decision"),
                "keep_as_regression_case": row.get("keep_as_regression_case"),
                "next_recommended_action": row.get("next_recommended_action"),
            }
            for row in targets
        },
        "representative_metrics": _representative_metrics(root),
        "smoke_commands_used_to_verify": [
            "python scripts/run_local_sweep.py --study local_baseline_guard_tsp",
            "python scripts/run_local_sweep.py --study local_baseline_guard_zdt1",
            "python scripts/run_local_sweep.py --study local_baseline_guard_knapsack",
            "python scripts/run_local_sweep.py --study local_baseline_guard_onemax",
            "python scripts/run_local_candidate.py --candidate configs/local_candidates/example_zdt1_candidate.json --use-existing-output",
            "python scripts/run_local_candidate.py --candidate configs/local_candidates/example_tsp_candidate.json --no-execute",
            "python scripts/run_local_candidate.py --candidate configs/local_candidates/example_knapsack_candidate.json --use-existing-output",
            "python scripts/summarize_local_candidates.py",
            "python scripts/summarize_local_optimization_status.py",
            "python scripts/check_local_baseline.py --write-snapshot",
            "python scripts/check_local_baseline.py",
        ],
        "guard_studies": GUARD_STUDIES,
        "candidate_workflow": {
            "schema_path": CANDIDATE_SCHEMA_PATH,
            "runner": "scripts/run_local_candidate.py",
            "ledger_runner": "scripts/summarize_local_candidates.py",
            "change_request_runner": "scripts/build_local_baseline_change_request.py",
            "status_runner": "scripts/summarize_local_optimization_status.py",
            "automatic_promotion": False,
            "ledger_artifacts": [
                "artifacts/local_candidate_ledger.json",
                "artifacts/local_candidate_summary.json",
            ],
            "decision_labels": [
                "reject_regression",
                "reject_no_material_gain",
                "note_only_stress_slice",
                "monitor_only",
                "candidate_promising_needs_confirm",
                "candidate_passes_local_guard",
                "candidate_requires_new_mechanism_hypothesis",
                "baseline_drift_detected",
                "intentional_baseline_change_required",
            ],
        },
        "cycle_closeout": {
            "status_artifacts": [
                "artifacts/local_optimization_status.json",
                "artifacts/local_candidate_backlog_closeout.json",
                "artifacts/local_reopen_criteria.json",
            ],
            "reopen_criteria_doc": "docs/local_reopen_criteria.md",
            "baseline_change_pending": False,
        },
    }
    _write_json(snapshot_path, snapshot)

    target_rows = [
        {
            "target_id": target_id,
            "latest_decision": decision["latest_decision"],
            "keep_as_regression_case": decision["keep_as_regression_case"],
            "next_recommended_action": decision["next_recommended_action"],
        }
        for target_id, decision in snapshot["frozen_target_decisions"].items()
    ]
    md_lines = [
        "# Local Baseline Snapshot",
        "",
        f"- Generated at: `{snapshot['generated_at']}`",
        f"- Protocol matrix hash: `{snapshot['protocol_matrix_hash']}`",
        f"- Stress target registry hash: `{snapshot['stress_target_registry_hash']}`",
        f"- Failure hypothesis registry hash: `{snapshot['failure_hypothesis_registry_hash']}`",
        f"- Candidate schema hash: `{snapshot['candidate_schema_hash']}`",
        "",
        "## Problem Rules",
        "",
        _markdown_table(
            [
                {
                    "problem": problem,
                    "decision": json.dumps(decision, ensure_ascii=False),
                }
                for problem, decision in snapshot["problem_operating_decisions"].items()
            ],
            ["problem", "decision"],
        ),
        "",
        "## Candidate Admission Rules",
        "",
        _markdown_table(
            [
                {
                    "problem": problem,
                    "allowed_candidate_type": decision["allowed_candidate_type"],
                    "promotion_threshold": decision["promotion_threshold"],
                    "rejection_condition": decision["rejection_condition"],
                }
                for problem, decision in snapshot["candidate_admission_rules"].items()
            ],
            [
                "problem",
                "allowed_candidate_type",
                "promotion_threshold",
                "rejection_condition",
            ],
        ),
        "",
        "## Frozen Target Decisions",
        "",
        _markdown_table(
            target_rows,
            [
                "target_id",
                "latest_decision",
                "keep_as_regression_case",
                "next_recommended_action",
            ],
        ),
        "",
        "## Smoke Commands",
        "",
        *[f"- `{command}`" for command in snapshot["smoke_commands_used_to_verify"]],
        "",
        "## Cycle Closeout",
        "",
        _markdown_table(
            [
                {
                    "artifact": artifact,
                    "type": "status_artifact",
                }
                for artifact in snapshot["cycle_closeout"]["status_artifacts"]
            ]
            + [
                {
                    "artifact": snapshot["cycle_closeout"]["reopen_criteria_doc"],
                    "type": "reopen_doc",
                }
            ],
            ["artifact", "type"],
        ),
        "",
    ]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")
    return snapshot


def check_local_baseline(
    project_root: Path | None = None,
    snapshot_path: Path | None = None,
    output_json_path: Path | None = None,
    output_md_path: Path | None = None,
) -> dict[str, Any]:
    root = project_root or PROJECT_ROOT
    snapshot_path = snapshot_path or (root / "artifacts" / "local_baseline_snapshot.json")
    output_json_path = output_json_path or (root / "artifacts" / "local_baseline_check.json")
    output_md_path = output_md_path or (root / "artifacts" / "local_baseline_check.md")

    snapshot = _load_json(snapshot_path)
    current_targets = freeze_target_registry_statuses(root, write_outputs=False)

    checks: list[dict[str, str]] = []

    for key, rel_path in PROFILE_PATHS.items():
        actual_hash = _sha256(root / rel_path)
        expected_hash = snapshot["profile_content_hashes"][key]
        checks.append(
            {
                "name": f"profile:{key}",
                "status": "PASS" if actual_hash == expected_hash else "FAIL",
                "detail": rel_path,
            }
        )

    protocol_hash = _sha256(root / "configs" / "local_protocols" / "local_operating_protocols.json")
    checks.append(
        {
            "name": "protocol_matrix",
            "status": "PASS" if protocol_hash == snapshot["protocol_matrix_hash"] else "FAIL",
            "detail": "configs/local_protocols/local_operating_protocols.json",
        }
    )

    target_hash = _sha256(root / "outputs" / "local_studies" / "future_optimization_targets.json")
    checks.append(
        {
            "name": "target_registry_hash",
            "status": "PASS" if target_hash == snapshot["stress_target_registry_hash"] else "FAIL",
            "detail": "outputs/local_studies/future_optimization_targets.json",
        }
    )

    failure_hash = _sha256(root / "outputs" / "local_studies" / "failure_hypotheses.json")
    checks.append(
        {
            "name": "failure_hypothesis_registry_hash",
            "status": "PASS"
            if failure_hash == snapshot["failure_hypothesis_registry_hash"]
            else "WARN",
            "detail": "outputs/local_studies/failure_hypotheses.json",
        }
    )

    candidate_schema_hash = _sha256(root / CANDIDATE_SCHEMA_PATH)
    checks.append(
        {
            "name": "candidate_schema_hash",
            "status": (
                "PASS"
                if candidate_schema_hash == snapshot.get("candidate_schema_hash")
                else "FAIL"
            ),
            "detail": CANDIDATE_SCHEMA_PATH,
        }
    )

    current_target_map = {row["target_id"]: row for row in current_targets}
    for target_id, expected in snapshot["frozen_target_decisions"].items():
        current = current_target_map.get(target_id, {})
        checks.append(
            {
                "name": f"target_status:{target_id}",
                "status": (
                    "PASS"
                    if current.get("latest_decision") == expected["latest_decision"]
                    and current.get("keep_as_regression_case") == expected["keep_as_regression_case"]
                    and current.get("next_recommended_action") == expected["next_recommended_action"]
                    else "FAIL"
                ),
                "detail": str(expected["latest_decision"]),
            }
        )

    current_decisions = _current_problem_decisions()
    for problem, expected in snapshot["problem_operating_decisions"].items():
        checks.append(
            {
                "name": f"problem_rule:{problem}",
                "status": "PASS" if current_decisions.get(problem) == expected else "FAIL",
                "detail": json.dumps(expected, ensure_ascii=False),
            }
        )

    current_candidate_rules = _candidate_admission_rules()
    candidate_rule_snapshot = snapshot.get("candidate_admission_rules")
    if not isinstance(candidate_rule_snapshot, dict):
        checks.append(
            {
                "name": "candidate_rules_snapshot",
                "status": "FAIL",
                "detail": "candidate_admission_rules missing from snapshot",
            }
        )
    for problem, expected in (candidate_rule_snapshot or {}).items():
        checks.append(
            {
                "name": f"candidate_rule:{problem}",
                "status": "PASS" if current_candidate_rules.get(problem) == expected else "FAIL",
                "detail": json.dumps(expected, ensure_ascii=False),
            }
        )

    for rel_path, expected_hash in snapshot.get("doc_hashes", {}).items():
        actual_hash = _sha256(root / rel_path)
        checks.append(
            {
                "name": f"doc_hash:{rel_path}",
                "status": "PASS" if actual_hash == expected_hash else "WARN",
                "detail": rel_path,
            }
        )

    overall_status = "PASS"
    if any(check["status"] == "FAIL" for check in checks):
        overall_status = "FAIL"
    elif any(check["status"] == "WARN" for check in checks):
        overall_status = "WARN"

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": overall_status,
        "checks": checks,
    }
    _write_json(output_json_path, report)

    md_lines = [
        "# Local Baseline Check",
        "",
        f"- Status: **{overall_status}**",
        "",
        _markdown_table(checks, ["name", "status", "detail"]),
        "",
    ]
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return report
