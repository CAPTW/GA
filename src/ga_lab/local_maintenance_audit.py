from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ga_lab.local_baseline import PROJECT_ROOT as DEFAULT_PROJECT_ROOT


EXPECTED_CANDIDATE_ROWS = {
    "example_knapsack_repair_note": {
        "problem": "knapsack",
        "target_id": "knapsack_repair_boundary_subset_sum_tight_capacity",
        "decision_label": "note_only_stress_slice",
        "lifecycle_state": "note_only",
        "recommended_next_action": "keep_narrow_note_on_subset_sum_tight_capacity_only",
    },
    "example_tsp_pg_contour_rejected": {
        "problem": "tsp",
        "target_id": "tsp_fast_anti_case_tail",
        "decision_label": "candidate_requires_new_mechanism_hypothesis",
        "lifecycle_state": "requires_new_mechanism",
        "recommended_next_action": "reopen_only_with_new_mechanism_hypothesis",
    },
    "example_zdt1_spread_candidate_note_only": {
        "problem": "zdt1",
        "target_id": "zdt1_fast_spread_safety_fail",
        "decision_label": "note_only_stress_slice",
        "lifecycle_state": "note_only",
        "recommended_next_action": "only_validate_if_candidate_generalizes_beyond_stress_slice",
    },
}

EXPECTED_TARGET_STATUSES = {
    "tsp_fast_anti_case_tail": {
        "latest_decision": "freeze_as_protocol_limitation",
        "next_recommended_action": "reopen_only_with_new_mechanism_hypothesis",
        "keep_as_regression_case": True,
    },
    "tsp_rescue_target_ambiguity": {
        "latest_decision": "secondary_regression_slice",
        "next_recommended_action": "keep_secondary_regression_slice",
        "keep_as_regression_case": True,
    },
    "zdt1_fast_spread_safety_fail": {
        "latest_decision": "note_only_stress_slice",
        "next_recommended_action": "only_validate_if_candidate_generalizes_beyond_stress_slice",
        "keep_as_regression_case": True,
    },
    "zdt1_fast_joint_safety_fail": {
        "latest_decision": "monitor_only",
        "next_recommended_action": "use_Q_for_final_safety",
        "keep_as_regression_case": True,
    },
    "knapsack_repair_boundary_subset_sum_tight_capacity": {
        "latest_decision": "narrow_note_only",
        "next_recommended_action": "keep_narrow_note_on_subset_sum_tight_capacity_only",
        "keep_as_regression_case": True,
    },
    "onemax_no_active_target": {
        "latest_decision": "no_active_target",
        "next_recommended_action": "no_active_target",
        "keep_as_regression_case": False,
    },
}

REOPEN_EXPECTATIONS = {
    "tsp": {
        "json_contains": [
            ("reopen_only_if", "new mechanism hypothesis"),
            ("reopen_only_if", "anti-case p95/max"),
            ("do_not_reopen_condition", "micro contour"),
            ("do_not_reopen_condition", "new mechanism story"),
        ],
        "doc_phrases": [
            "new mechanism hypothesis",
            "anti-case p95/max",
            "rescue-target mean non-regression",
            "same or lower configured budget",
            "quality-sensitive finals still go straight to q 8-10",
        ],
    },
    "zdt1": {
        "json_contains": [
            ("reopen_only_if", "generalizes beyond stress rows"),
            ("reopen_only_if", "stable/normal regression"),
            ("do_not_reopen_condition", "stress-slice-only spread win"),
            ("do_not_reopen_condition", "hv tail penalty"),
        ],
        "doc_phrases": [
            "spread-stress slice",
            "stable/normal rows",
            "joint safety failures",
            "pareto-ratio non-regression",
            "final safety still belongs to q",
        ],
    },
    "knapsack": {
        "json_contains": [
            ("reopen_only_if", "family-conditioned evidence"),
            ("reopen_only_if", "subset-sum/tight-capacity-like"),
            ("do_not_reopen_condition", "weakly correlated"),
            ("do_not_reopen_condition", "current narrow note"),
        ],
        "doc_phrases": [
            "subset-sum-like",
            "tight-capacity-like",
            "broad default promotion stays forbidden",
        ],
    },
    "onemax": {
        "json_contains": [
            ("reopen_only_if", "control drift"),
            ("reopen_only_if", "instrumentation regression"),
            ("do_not_reopen_condition", "richer adaptive/control variant"),
        ],
        "doc_phrases": [
            "control drift",
            "instrumentation regression",
            "adaptive search",
        ],
    },
}

DOC_EXPECTATIONS = {
    "docs/local_candidate_workflow.md": [
        "candidate manifests live under",
        "baseline drift:",
        "candidate improvement:",
        "passing a candidate guard still does not rewrite the baseline automatically.",
        "future work must still enter as candidate manifests",
    ],
    "docs/local_change_control.md": [
        "only these states may open a normal change-request pack:",
        "same-budget contour candidates should never reach change-request without a",
        "no active target means no active change-request path",
    ],
    "docs/local_protocol_guide.md": [
        "anti-case / corridor suspicion or quality-sensitive final still goes straight to `q 8-10`",
        "final safety still belongs to `q`",
        "`spread_pg_pop41_gen88` stays note-only on the spread-stress slice",
        "`repair_only` can survive only as a narrow family-conditioned note",
        "candidate manifests, not as silent profile replacements.",
    ],
    "docs/local_experiment_guide.md": [
        "once the baseline is frozen, new local ideas should enter as candidate manifests",
        "anti-case / corridor suspicion or quality-sensitive final still goes straight to `q 8-10`",
        "final safety still belongs to `q`",
        "keep the `repair_only` note narrow on subset-sum / tight-capacity-like rows only",
        "`none` control only",
    ],
    "README.md": [
        "future work must enter through a candidate manifest, not a direct profile edit",
        "baseline drift and candidate improvement are intentionally separated",
        "anti-case / corridor suspicion or quality-sensitive final still goes straight",
        "final safety still belongs to `q`",
        "`none` control only",
    ],
    "examples/README.md": [
        "tsp: `explore -> f 3`, `compare -> paired q/f at 3 (rescue-target-only 5)`, `final -> q 8-10`",
        "zdt1: `explore -> f 3`, `final safety -> q 8-10`",
        "tsp fast is budget-first / exploratory only",
        "onemax stays control-only.",
    ],
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _resolve(root: Path, path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else root / raw


def _lower(value: Any) -> str:
    return str(value).lower()


def _contains(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _check(name: str, ok: bool, detail: str, *, false_status: str = "FAIL") -> dict[str, str]:
    return {
        "name": name,
        "status": "PASS" if ok else false_status,
        "detail": detail,
    }


def _problem_rules_match(protocol_matrix: dict[str, Any]) -> bool:
    protocols = protocol_matrix.get("protocols", {})
    tsp = protocols.get("tsp", {})
    zdt1 = protocols.get("zdt1", {})
    knapsack = protocols.get("knapsack", {})
    onemax = protocols.get("onemax", {})

    tsp_ok = (
        tsp.get("default_profiles", {}).get("quality_first")
        == "configs/local_profiles/tsp_seeded_swap_local.json"
        and tsp.get("default_profiles", {}).get("budget_first")
        == "configs/local_profiles/tsp_seeded_swap_local_fast.json"
        and tsp.get("modes", {}).get("final", {}).get("recommended_profile") == "quality_first"
        and tsp.get("modes", {}).get("final", {}).get("initial_seed_count") == 8
        and "anti-case" in _lower(tsp.get("modes", {}).get("final", {}).get("rule", ""))
    )
    zdt1_ok = (
        zdt1.get("default_profiles", {}).get("quality_first")
        == "configs/local_profiles/zdt1_diversity_injection.json"
        and zdt1.get("default_profiles", {}).get("budget_first")
        == "configs/local_profiles/zdt1_diversity_injection_fast.json"
        and zdt1.get("modes", {}).get("final", {}).get("recommended_profile") == "quality_first"
        and zdt1.get("modes", {}).get("final", {}).get("initial_seed_count") == 8
        and "safety" in _lower(zdt1.get("modes", {}).get("final", {}).get("rule", ""))
    )
    knapsack_ok = (
        knapsack.get("default_profiles", {}).get("default") == "greedy_local_search"
        and knapsack.get("default_profiles", {}).get("repair_note")
        == "configs/local_profiles/knapsack_repair_local_experimental.json"
        and knapsack.get("modes", {}).get("sanity", {}).get("initial_seed_count") == 3
    )
    onemax_ok = (
        onemax.get("default_profiles", {}).get("control") == "none"
        and onemax.get("modes", {}).get("control", {}).get("initial_seed_count") == 1
        and onemax.get("modes", {}).get("control", {}).get("recommended_profile") == "control"
    )
    return tsp_ok and zdt1_ok and knapsack_ok and onemax_ok


def build_local_reopen_criteria_check(
    *,
    project_root: Path | None = None,
    reopen_json_path: str | Path = "artifacts/local_reopen_criteria.json",
    reopen_doc_path: str | Path = "docs/local_reopen_criteria.md",
    output_json_path: str | Path = "artifacts/local_reopen_criteria_check.json",
    output_md_path: str | Path = "artifacts/local_reopen_criteria_check.md",
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT
    reopen_payload = _load_json(_resolve(root, reopen_json_path))
    reopen_doc = _resolve(root, reopen_doc_path).read_text(encoding="utf-8")

    checks: list[dict[str, str]] = []
    problems = reopen_payload.get("problems", {})
    for problem, expected in REOPEN_EXPECTATIONS.items():
        current = problems.get(problem, {})
        for field_name, phrase in expected["json_contains"]:
            value = _lower(current.get(field_name, ""))
            checks.append(
                _check(
                    f"artifact:{problem}:{field_name}:{phrase}",
                    phrase.lower() in value,
                    str(current.get(field_name, "")),
                )
            )
        for phrase in expected["doc_phrases"]:
            checks.append(
                _check(
                    f"doc:{problem}:{phrase}",
                    _contains(reopen_doc, phrase),
                    phrase,
                )
            )

    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "checks": checks,
    }

    output_json = _resolve(root, output_json_path)
    output_md = _resolve(root, output_md_path)
    _write_json(output_json, payload)
    _write_markdown(
        output_md,
        "\n".join(
            [
                "# Local Reopen Criteria Check",
                "",
                f"- status: **{status}**",
                "",
                _markdown_table(checks, ["name", "status", "detail"]),
                "",
            ]
        ),
    )
    return payload


def build_local_maintenance_audit(
    *,
    project_root: Path | None = None,
    baseline_snapshot_path: str | Path = "artifacts/local_baseline_snapshot.json",
    baseline_check_path: str | Path = "artifacts/local_baseline_check.json",
    optimization_status_path: str | Path = "artifacts/local_optimization_status.json",
    candidate_ledger_path: str | Path = "artifacts/local_candidate_ledger.json",
    candidate_summary_path: str | Path = "artifacts/local_candidate_summary.json",
    candidate_backlog_path: str | Path = "artifacts/local_candidate_backlog_closeout.json",
    reopen_criteria_path: str | Path = "artifacts/local_reopen_criteria.json",
    reopen_criteria_check_path: str | Path = "artifacts/local_reopen_criteria_check.json",
    protocol_matrix_path: str | Path = "configs/local_protocols/local_operating_protocols.json",
    target_registry_path: str | Path = "outputs/local_studies/future_optimization_targets.json",
    failure_hypotheses_path: str | Path = "outputs/local_studies/failure_hypotheses.json",
    output_json_path: str | Path = "artifacts/local_maintenance_audit.json",
    output_md_path: str | Path = "artifacts/local_maintenance_audit.md",
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT

    snapshot = _load_json(_resolve(root, baseline_snapshot_path))
    baseline_check = _load_json(_resolve(root, baseline_check_path))
    optimization_status = _load_json(_resolve(root, optimization_status_path))
    candidate_ledger = _load_json(_resolve(root, candidate_ledger_path))
    candidate_summary = _load_json(_resolve(root, candidate_summary_path))
    candidate_backlog = _load_json(_resolve(root, candidate_backlog_path))
    reopen_criteria = _load_json(_resolve(root, reopen_criteria_path))
    reopen_check = _load_json(_resolve(root, reopen_criteria_check_path))
    protocol_matrix = _load_json(_resolve(root, protocol_matrix_path))
    target_rows = _load_json(_resolve(root, target_registry_path))
    failure_hypotheses = _load_json(_resolve(root, failure_hypotheses_path))

    ledger_rows = list(candidate_ledger.get("rows", []))
    ledger_map = {str(row.get("candidate_id")): row for row in ledger_rows}
    target_map = {str(row.get("target_id")): row for row in target_rows}

    doc_checks: list[dict[str, str]] = []
    for rel_path, phrases in DOC_EXPECTATIONS.items():
        content = _resolve(root, rel_path).read_text(encoding="utf-8")
        for phrase in phrases:
            doc_checks.append(
                _check(
                    f"doc:{rel_path}:{phrase}",
                    _contains(content, phrase),
                    phrase,
                )
            )
    protocol_docs_status = "PASS" if all(item["status"] == "PASS" for item in doc_checks) else "FAIL"

    artifact_rows: list[dict[str, str]] = []

    snapshot_ok = bool(snapshot.get("profile_paths")) and bool(snapshot.get("problem_operating_decisions"))
    artifact_rows.append(
        {
            "artifact": "artifacts/local_baseline_snapshot.json",
            "current_status": "present" if snapshot_ok else "missing_fields",
            "expected_closeout_value": "frozen baseline snapshot with profile paths, protocol decisions, and target decisions",
            "drift_detected": "no" if snapshot_ok else "yes",
            "action_required": "no" if snapshot_ok else "yes",
            "note": "The snapshot stays frozen; this pass verifies it but does not rewrite it.",
        }
    )

    baseline_ok = baseline_check.get("status") == "PASS"
    artifact_rows.append(
        {
            "artifact": "artifacts/local_baseline_check.json",
            "current_status": str(baseline_check.get("status")),
            "expected_closeout_value": "PASS with no profile/protocol/target drift",
            "drift_detected": "no" if baseline_ok else "yes",
            "action_required": "no" if baseline_ok else "yes",
            "note": "Frozen baseline hash guard remains the first gate.",
        }
    )

    status_ok = (
        optimization_status.get("cycle_state") == "local_optimization_cycle_1_frozen"
        and optimization_status.get("no_profile_changes_pending") is True
        and optimization_status.get("no_candidate_ready_to_change_baseline") is True
    )
    artifact_rows.append(
        {
            "artifact": "artifacts/local_optimization_status.json",
            "current_status": (
                f"cycle={optimization_status.get('cycle_state')}, "
                f"pending={optimization_status.get('no_profile_changes_pending')}, "
                f"baseline_ready={optimization_status.get('no_candidate_ready_to_change_baseline')}"
            ),
            "expected_closeout_value": (
                "cycle_state=local_optimization_cycle_1_frozen, "
                "no_profile_changes_pending=true, "
                "no_candidate_ready_to_change_baseline=true"
            ),
            "drift_detected": "no" if status_ok else "yes",
            "action_required": "no" if status_ok else "yes",
            "note": "Semantic closeout status matters more than volatile generated_at fields.",
        }
    )

    candidate_ids_ok = set(ledger_map) == set(EXPECTED_CANDIDATE_ROWS)
    candidate_rows_ok = candidate_ids_ok and all(
        all(ledger_map[candidate_id].get(key) == value for key, value in expected.items())
        for candidate_id, expected in EXPECTED_CANDIDATE_ROWS.items()
    )
    artifact_rows.append(
        {
            "artifact": "artifacts/local_candidate_ledger.json",
            "current_status": f"total={len(ledger_rows)}, ids={','.join(sorted(ledger_map))}",
            "expected_closeout_value": (
                "3 candidates: example_zdt1_spread_candidate_note_only, "
                "example_tsp_pg_contour_rejected, example_knapsack_repair_note"
            ),
            "drift_detected": "no" if candidate_rows_ok else "yes",
            "action_required": "no" if candidate_rows_ok else "yes",
            "note": "All known candidates should stay audit-only and baseline-inert.",
        }
    )

    summary_ok = (
        candidate_summary.get("total_candidates") == 3
        and candidate_summary.get("counts_by_lifecycle_state", {}).get("note_only") == 2
        and candidate_summary.get("counts_by_lifecycle_state", {}).get("requires_new_mechanism") == 1
        and not candidate_summary.get("ready_for_change_request_candidates")
        and not candidate_summary.get("passed_local_guard_candidates")
    )
    artifact_rows.append(
        {
            "artifact": "artifacts/local_candidate_summary.json",
            "current_status": (
                f"total={candidate_summary.get('total_candidates')}, "
                f"note_only={candidate_summary.get('counts_by_lifecycle_state', {}).get('note_only', 0)}, "
                f"requires_new_mechanism={candidate_summary.get('counts_by_lifecycle_state', {}).get('requires_new_mechanism', 0)}, "
                f"ready={len(candidate_summary.get('ready_for_change_request_candidates', []))}"
            ),
            "expected_closeout_value": "total=3, note_only=2, requires_new_mechanism=1, ready_for_change_request=0",
            "drift_detected": "no" if summary_ok else "yes",
            "action_required": "no" if summary_ok else "yes",
            "note": "No candidate should be baseline-ready in cycle-1 closeout.",
        }
    )

    backlog_ok = (
        candidate_backlog.get("no_candidate_is_ready_to_change_baseline") is True
        and candidate_backlog.get("closeout_read", {}).get("zdt1_spread_candidate") == "note_only_stress_slice"
        and candidate_backlog.get("closeout_read", {}).get("tsp_pg_contour_candidate")
        == "candidate_requires_new_mechanism_hypothesis"
        and candidate_backlog.get("closeout_read", {}).get("knapsack_repair_note") == "note_only_stress_slice"
    )
    artifact_rows.append(
        {
            "artifact": "artifacts/local_candidate_backlog_closeout.json",
            "current_status": (
                f"ready={candidate_backlog.get('no_candidate_is_ready_to_change_baseline')}, "
                f"zdt1={candidate_backlog.get('closeout_read', {}).get('zdt1_spread_candidate')}, "
                f"tsp={candidate_backlog.get('closeout_read', {}).get('tsp_pg_contour_candidate')}"
            ),
            "expected_closeout_value": (
                "no candidate ready; zdt1 note_only_stress_slice; "
                "tsp requires_new_mechanism; knapsack note_only_stress_slice"
            ),
            "drift_detected": "no" if backlog_ok else "yes",
            "action_required": "no" if backlog_ok else "yes",
            "note": "Backlog closeout should mirror the ledger rather than invent a pending change path.",
        }
    )

    reopen_ok = reopen_check.get("status") == "PASS" and set(reopen_criteria.get("problems", {})) == {
        "tsp",
        "zdt1",
        "knapsack",
        "onemax",
    }
    artifact_rows.append(
        {
            "artifact": "artifacts/local_reopen_criteria.json",
            "current_status": str(reopen_check.get("status")),
            "expected_closeout_value": "PASS with problem-specific reopen-only triggers intact",
            "drift_detected": "no" if reopen_ok else "yes",
            "action_required": "no" if reopen_ok else "yes",
            "note": "Reopen is allowed only through candidate manifests plus baseline guard.",
        }
    )

    protocol_ok = _problem_rules_match(protocol_matrix)
    artifact_rows.append(
        {
            "artifact": "configs/local_protocols/local_operating_protocols.json",
            "current_status": "PASS" if protocol_ok else "FAIL",
            "expected_closeout_value": "TSP final -> Q 8-10, ZDT1 final safety -> Q, Knapsack broad default parked, Onemax control only",
            "drift_detected": "no" if protocol_ok else "yes",
            "action_required": "no" if protocol_ok else "yes",
            "note": "Protocol matrix still anchors the frozen local operating rules.",
        }
    )

    target_ok = all(
        target_map.get(target_id, {}).get("latest_decision") == expected["latest_decision"]
        and target_map.get(target_id, {}).get("next_recommended_action") == expected["next_recommended_action"]
        and target_map.get(target_id, {}).get("keep_as_regression_case") == expected["keep_as_regression_case"]
        for target_id, expected in EXPECTED_TARGET_STATUSES.items()
    )
    artifact_rows.append(
        {
            "artifact": "outputs/local_studies/future_optimization_targets.json",
            "current_status": "PASS" if target_ok else "FAIL",
            "expected_closeout_value": "freeze_as_protocol_limitation / note_only_stress_slice / monitor_only / narrow_note_only / no_active_target",
            "drift_detected": "no" if target_ok else "yes",
            "action_required": "no" if target_ok else "yes",
            "note": "Target registry should stay frozen until a future candidate truly clears the local guard.",
        }
    )

    hypotheses_ok = len(failure_hypotheses) >= 6
    artifact_rows.append(
        {
            "artifact": "outputs/local_studies/failure_hypotheses.json",
            "current_status": f"count={len(failure_hypotheses)}",
            "expected_closeout_value": "failure hypotheses present for TSP/ZDT1 plus parked knapsack/onemax state",
            "drift_detected": "no" if hypotheses_ok else "yes",
            "action_required": "no" if hypotheses_ok else "yes",
            "note": "Mechanism history remains available even though cycle-1 is frozen.",
        }
    )

    artifact_rows.append(
        {
            "artifact": "docs/* + README.md + examples/README.md",
            "current_status": protocol_docs_status,
            "expected_closeout_value": "candidate manifest gate, baseline drift separation, no pending baseline change, TSP anti-case->Q, ZDT1 final safety->Q, Knapsack narrow note, Onemax control only",
            "drift_detected": "no" if protocol_docs_status == "PASS" else "yes",
            "action_required": "no" if protocol_docs_status == "PASS" else "yes",
            "note": "Docs should describe the frozen state without re-opening optimization work.",
        }
    )

    drift_detected = any(row["drift_detected"] == "yes" for row in artifact_rows)
    action_required = any(row["action_required"] == "yes" for row in artifact_rows)

    candidate_state_drift = any(
        row["artifact"] in {
            "artifacts/local_candidate_ledger.json",
            "artifacts/local_candidate_summary.json",
            "artifacts/local_candidate_backlog_closeout.json",
        }
        and row["drift_detected"] == "yes"
        for row in artifact_rows
    )
    baseline_or_protocol_drift = any(
        row["artifact"] in {
            "artifacts/local_baseline_check.json",
            "configs/local_protocols/local_operating_protocols.json",
            "outputs/local_studies/future_optimization_targets.json",
        }
        and row["drift_detected"] == "yes"
        for row in artifact_rows
    )

    if baseline_or_protocol_drift:
        final_decision = "fail_baseline_drift"
    elif candidate_state_drift:
        final_decision = "fail_candidate_state_drift"
    elif protocol_docs_status != "PASS" or reopen_check.get("status") != "PASS":
        final_decision = "warn_docs_drift"
    else:
        final_decision = "no_op_pass"

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_check_status": baseline_check.get("status"),
        "candidate_backlog_status": "PASS" if summary_ok and backlog_ok and candidate_rows_ok else "FAIL",
        "reopen_criteria_status": reopen_check.get("status"),
        "protocol_docs_status": protocol_docs_status,
        "drift_detected": drift_detected,
        "action_required": action_required,
        "allowed_next_work": optimization_status.get("next_allowed_work", []),
        "disallowed_next_work": optimization_status.get("next_disallowed_work", []),
        "final_maintenance_decision": final_decision,
        "artifact_rows": artifact_rows,
        "doc_checks": doc_checks,
        "baseline_snapshot_hash": _sha256(_resolve(root, baseline_snapshot_path)),
    }

    output_json = _resolve(root, output_json_path)
    output_md = _resolve(root, output_md_path)
    _write_json(output_json, payload)
    _write_markdown(
        output_md,
        "\n".join(
            [
                "# Local Maintenance Audit",
                "",
                f"- generated at: `{payload['generated_at']}`",
                f"- baseline check status: `{payload['baseline_check_status']}`",
                f"- candidate backlog status: `{payload['candidate_backlog_status']}`",
                f"- reopen criteria status: `{payload['reopen_criteria_status']}`",
                f"- protocol/docs status: `{payload['protocol_docs_status']}`",
                f"- drift detected: `{payload['drift_detected']}`",
                f"- action required: `{payload['action_required']}`",
                f"- final maintenance decision: `{payload['final_maintenance_decision']}`",
                "",
                "## Artifact Audit Table",
                "",
                _markdown_table(
                    artifact_rows,
                    [
                        "artifact",
                        "current_status",
                        "expected_closeout_value",
                        "drift_detected",
                        "action_required",
                        "note",
                    ],
                ),
                "",
                "## Allowed Next Work",
                "",
                *[f"- {line}" for line in payload["allowed_next_work"]],
                "",
                "## Disallowed Next Work",
                "",
                *[f"- {line}" for line in payload["disallowed_next_work"]],
                "",
            ]
        ),
    )
    return payload
