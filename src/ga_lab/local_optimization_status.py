from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ga_lab.local_baseline import PROJECT_ROOT as DEFAULT_PROJECT_ROOT


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (root / raw)


def _targets_by_decision(target_rows: list[dict[str, Any]], decision: str) -> list[str]:
    return sorted(
        str(row["target_id"])
        for row in target_rows
        if str(row.get("latest_decision", "")) == decision
    )


def _candidate_rows_by_lifecycle(ledger_rows: list[dict[str, Any]], lifecycle: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in ledger_rows if str(row.get("lifecycle_state", "")) == lifecycle],
        key=lambda row: (str(row.get("problem", "")), str(row.get("candidate_id", ""))),
    )


def _current_problem_closeout_state() -> dict[str, dict[str, Any]]:
    return {
        "tsp": {
            "current_state": (
                "Frozen budget-first/exploratory split; anti-case/corridor suspicion and "
                "quality-sensitive finals still go straight to Q 8-10."
            ),
            "reopen_trigger": [
                "A genuinely new mechanism hypothesis exists.",
                "The candidate targets anti-case p95/max directly instead of repeating the old contour story.",
            ],
            "required_evidence": [
                "Anti-case p95/max improvement versus the frozen current fast baseline.",
                "Rescue-target mean non-regression.",
                "Same or lower configured budget.",
                "Baseline guard PASS before candidate comparison.",
            ],
            "required_candidate_label": (
                "candidate_promising_needs_confirm or stronger; baseline change still requires "
                "candidate_passes_local_guard."
            ),
            "minimum_guard_checks": [
                "python scripts/check_local_baseline.py",
                "python scripts/run_local_candidate.py --candidate <manifest>",
                "python scripts/summarize_local_candidates.py",
            ],
            "rejection_shortcut": (
                "Reject early when the idea is another same-budget PG contour rerun without a new "
                "mechanism story, or when anti-case p95/max is not improved."
            ),
            "forbidden_reopen_pattern": [
                "Re-running the already failed population/generation contour family.",
                "Using a stress-slice-only gain to argue for a baseline replacement.",
            ],
            "reopen_only_if": (
                "A new mechanism hypothesis can beat anti-case p95/max while keeping rescue-target "
                "mean and budget honest."
            ),
            "do_not_reopen_condition": (
                "Do not reopen for another micro contour or seed-fraction nudge that lacks a new "
                "mechanism story."
            ),
            "current_recommendation": (
                "Keep F for budget-first/exploratory work; go straight to Q 8-10 for anti-case, "
                "corridor-like, or quality-sensitive finals."
            ),
        },
        "zdt1": {
            "current_state": (
                "Frozen exploratory/budget-first split; spread candidate stays note-only and final "
                "safety still belongs to Q."
            ),
            "reopen_trigger": [
                "A spread candidate generalizes beyond the spread-stress slice to stable/normal rows.",
                "A new mechanism reduces joint safety failures without reopening HV tail loss.",
            ],
            "required_evidence": [
                "HV mean/tail non-regression versus the frozen fast baseline.",
                "Spread fail reduction that survives stable and normal slices.",
                "Joint safety non-regression.",
                "Pareto-ratio non-regression.",
                "Baseline guard PASS before candidate comparison.",
            ],
            "required_candidate_label": (
                "candidate_promising_needs_confirm or stronger; replacement review still requires "
                "candidate_passes_local_guard."
            ),
            "minimum_guard_checks": [
                "python scripts/check_local_baseline.py",
                "python scripts/run_local_candidate.py --candidate <manifest>",
                "python scripts/summarize_local_candidates.py",
            ],
            "rejection_shortcut": (
                "Reject when the candidate wins only on the spread-stress slice or when HV/joint/"
                "Pareto non-regression breaks on stable or normal rows."
            ),
            "forbidden_reopen_pattern": [
                "Claiming spread-stress-only gain as a fast-default replacement.",
                "Timing-only joint candidates that pay clear HV tail penalty.",
            ],
            "reopen_only_if": (
                "A same-budget candidate generalizes beyond stress rows without stable/normal "
                "regression and without hurting HV, Pareto, or joint safety."
            ),
            "do_not_reopen_condition": (
                "Do not reopen for another stress-slice-only spread win or another timing-only "
                "joint tweak with HV tail penalty."
            ),
            "current_recommendation": (
                "Keep F for exploratory/budget-first HV reads; keep Q for final safety and treat "
                "spread_pg_pop41_gen88 only as a note-level stress-slice reference."
            ),
        },
        "knapsack": {
            "current_state": "No broad default; repair_only remains a narrow family-conditioned note.",
            "reopen_trigger": [
                "New evidence is explicitly limited to subset-sum-like or tight-capacity-like families.",
                "The candidate shows consistent gain over none/greedy/repair_only on that family.",
            ],
            "required_evidence": [
                "Family-conditioned feasible-quality improvement.",
                "No broad-default claim.",
                "Baseline guard PASS before candidate comparison.",
            ],
            "required_candidate_label": "note_only_stress_slice at most; broad default promotion stays forbidden.",
            "minimum_guard_checks": [
                "python scripts/check_local_baseline.py",
                "python scripts/run_local_candidate.py --candidate <manifest>",
            ],
            "rejection_shortcut": "Reject any candidate that implies a broad default or loses to greedy on the narrow family.",
            "forbidden_reopen_pattern": [
                "Generalizing weakly correlated ties into a broad rule.",
                "Reopening broad default search for knapsack.",
            ],
            "reopen_only_if": (
                "Family-conditioned evidence stays narrow and consistently better on subset-sum/"
                "tight-capacity-like rows."
            ),
            "do_not_reopen_condition": (
                "Do not reopen when the gain is weakly correlated, broad, or indistinguishable from "
                "the current narrow note."
            ),
            "current_recommendation": (
                "Keep greedy_local_search as the practical baseline and treat repair_only as a "
                "narrow note only."
            ),
        },
        "onemax": {
            "current_state": "Frozen control-only branch on none; no active optimization target.",
            "reopen_trigger": [
                "Baseline control drift appears.",
                "Instrumentation regression needs a control recheck.",
            ],
            "required_evidence": [
                "Control drift or instrumentation regression evidence.",
                "Baseline guard PASS before candidate comparison.",
            ],
            "required_candidate_label": "monitor_only at most unless there is a clear control simplification.",
            "minimum_guard_checks": [
                "python scripts/check_local_baseline.py",
            ],
            "rejection_shortcut": "Reject anything more complex than the current none-control path without obvious control gain.",
            "forbidden_reopen_pattern": [
                "Reintroducing adaptive search into the control problem.",
                "Treating Onemax as an active optimization target without drift evidence.",
            ],
            "reopen_only_if": "A real control drift or instrumentation regression appears.",
            "do_not_reopen_condition": "Do not reopen just to try a richer adaptive/control variant.",
            "current_recommendation": "Keep none as the control-only baseline and spend no broader optimization budget here.",
        },
    }


def build_local_reopen_criteria(
    *,
    project_root: Path | None = None,
    output_json_path: str | Path = "artifacts/local_reopen_criteria.json",
    output_md_path: str | Path = "docs/local_reopen_criteria.md",
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT
    criteria = {
        "generated_at": datetime.now(UTC).isoformat(),
        "problems": _current_problem_closeout_state(),
    }

    json_path = _resolve(root, output_json_path)
    md_path = _resolve(root, output_md_path)
    _write_json(json_path, criteria)

    rows = [
        {
            "problem": problem,
            "current_state": payload["current_state"],
            "reopen_trigger": "; ".join(payload["reopen_trigger"]),
            "required_candidate_label": payload["required_candidate_label"],
            "rejection_shortcut": payload["rejection_shortcut"],
            "forbidden_reopen_pattern": "; ".join(payload["forbidden_reopen_pattern"]),
        }
        for problem, payload in criteria["problems"].items()
    ]
    md_lines = [
        "# Local Reopen Criteria",
        "",
        "Local optimization cycle 1 is frozen. Reopen work only through a candidate manifest plus a baseline check.",
        "",
        _markdown_table(
            rows,
            [
                "problem",
                "current_state",
                "reopen_trigger",
                "required_candidate_label",
                "rejection_shortcut",
                "forbidden_reopen_pattern",
            ],
        ),
        "",
        "## Problem Details",
        "",
    ]
    for problem, payload in criteria["problems"].items():
        md_lines.extend(
            [
                f"### {problem.upper()}",
                "",
                f"- current state: {payload['current_state']}",
                f"- reopen trigger: {'; '.join(payload['reopen_trigger'])}",
                f"- required evidence: {'; '.join(payload['required_evidence'])}",
                f"- required candidate label: {payload['required_candidate_label']}",
                f"- minimum guard checks: {'; '.join(payload['minimum_guard_checks'])}",
                f"- rejection shortcut: {payload['rejection_shortcut']}",
                f"- forbidden reopen pattern: {'; '.join(payload['forbidden_reopen_pattern'])}",
                "",
            ]
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return criteria


def build_local_candidate_backlog_closeout(
    *,
    project_root: Path | None = None,
    ledger_path: str | Path = "artifacts/local_candidate_ledger.json",
    summary_path: str | Path = "artifacts/local_candidate_summary.json",
    output_json_path: str | Path = "artifacts/local_candidate_backlog_closeout.json",
    output_md_path: str | Path = "artifacts/local_candidate_backlog_closeout.md",
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT
    ledger = _load_json(_resolve(root, ledger_path))
    summary = _load_json(_resolve(root, summary_path))
    ledger_rows = list(ledger.get("rows", []))

    note_only_rows = _candidate_rows_by_lifecycle(ledger_rows, "note_only")
    requires_new_mechanism_rows = _candidate_rows_by_lifecycle(ledger_rows, "requires_new_mechanism")
    ready_rows = _candidate_rows_by_lifecycle(ledger_rows, "ready_for_change_request")
    blocked_rows = _candidate_rows_by_lifecycle(ledger_rows, "blocked_by_baseline_drift")

    backlog = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_candidates": summary.get("total_candidates", len(ledger_rows)),
        "candidates_by_problem": summary.get("counts_by_problem", {}),
        "candidates_by_lifecycle_state": summary.get("counts_by_lifecycle_state", {}),
        "note_only_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "problem": row["problem"],
                "target_id": row["target_id"],
                "recommended_next_action": row["recommended_next_action"],
            }
            for row in note_only_rows
        ],
        "requires_new_mechanism_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "problem": row["problem"],
                "target_id": row["target_id"],
                "recommended_next_action": row["recommended_next_action"],
            }
            for row in requires_new_mechanism_rows
        ],
        "ready_for_change_request_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "problem": row["problem"],
                "target_id": row["target_id"],
            }
            for row in ready_rows
        ],
        "blocked_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "problem": row["problem"],
                "target_id": row["target_id"],
            }
            for row in blocked_rows
        ],
        "recommended_next_action_per_candidate": [
            {
                "candidate_id": row["candidate_id"],
                "problem": row["problem"],
                "decision_label": row["decision_label"],
                "lifecycle_state": row["lifecycle_state"],
                "recommended_next_action": row["recommended_next_action"],
                "can_affect_baseline": bool(row.get("baseline_change_candidate")),
            }
            for row in sorted(
                ledger_rows,
                key=lambda row: (str(row.get("problem", "")), str(row.get("candidate_id", ""))),
            )
        ],
        "no_candidate_is_ready_to_change_baseline": not ready_rows,
        "closeout_read": {
            "zdt1_spread_candidate": "note_only_stress_slice",
            "tsp_pg_contour_candidate": "candidate_requires_new_mechanism_hypothesis",
            "knapsack_repair_note": "note_only_stress_slice",
        },
    }

    json_path = _resolve(root, output_json_path)
    md_path = _resolve(root, output_md_path)
    _write_json(json_path, backlog)
    md_lines = [
        "# Local Candidate Backlog Closeout",
        "",
        f"- generated at: `{backlog['generated_at']}`",
        f"- total candidates: `{backlog['total_candidates']}`",
        f"- no candidate is ready to change baseline: `{backlog['no_candidate_is_ready_to_change_baseline']}`",
        "",
        "## Lifecycle Counts",
        "",
        _markdown_table(
            [
                {"lifecycle_state": key, "count": value}
                for key, value in backlog["candidates_by_lifecycle_state"].items()
            ],
            ["lifecycle_state", "count"],
        )
        if backlog["candidates_by_lifecycle_state"]
        else "_No lifecycle rows._",
        "",
        "## Candidate Closeout",
        "",
        _markdown_table(
            backlog["recommended_next_action_per_candidate"],
            [
                "candidate_id",
                "problem",
                "decision_label",
                "lifecycle_state",
                "recommended_next_action",
                "can_affect_baseline",
            ],
        )
        if backlog["recommended_next_action_per_candidate"]
        else "_No candidates found._",
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return backlog


def build_local_optimization_status(
    *,
    project_root: Path | None = None,
    baseline_snapshot_path: str | Path = "artifacts/local_baseline_snapshot.json",
    baseline_check_path: str | Path = "artifacts/local_baseline_check.json",
    candidate_ledger_path: str | Path = "artifacts/local_candidate_ledger.json",
    candidate_summary_path: str | Path = "artifacts/local_candidate_summary.json",
    target_registry_path: str | Path = "outputs/local_studies/future_optimization_targets.json",
    failure_hypotheses_path: str | Path = "outputs/local_studies/failure_hypotheses.json",
    output_json_path: str | Path = "artifacts/local_optimization_status.json",
    output_md_path: str | Path = "artifacts/local_optimization_status.md",
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT
    snapshot_path = _resolve(root, baseline_snapshot_path)
    check_path = _resolve(root, baseline_check_path)
    ledger_path = _resolve(root, candidate_ledger_path)
    summary_path = _resolve(root, candidate_summary_path)
    targets_path = _resolve(root, target_registry_path)
    hypotheses_path = _resolve(root, failure_hypotheses_path)

    snapshot = _load_json(snapshot_path)
    baseline_check = _load_json(check_path)
    ledger = _load_json(ledger_path)
    summary = _load_json(summary_path)
    target_rows = _load_json(targets_path)
    hypothesis_rows = _load_json(hypotheses_path)
    ledger_rows = list(ledger.get("rows", []))

    target_statuses = {
        str(row["target_id"]): {
            "latest_decision": row.get("latest_decision"),
            "current_severity": row.get("current_severity"),
            "next_recommended_action": row.get("next_recommended_action"),
        }
        for row in target_rows
    }

    problem_rows = [
        {
            "problem": "tsp",
            "current_baseline_default": (
                "Q=configs/local_profiles/tsp_seeded_swap_local.json; "
                "F=configs/local_profiles/tsp_seeded_swap_local_fast.json"
            ),
            "active_target_status": "freeze_as_protocol_limitation + secondary_regression_slice",
            "candidate_backlog_status": "1 requires_new_mechanism candidate; 0 baseline-ready",
            "current_recommendation": _current_problem_closeout_state()["tsp"]["current_recommendation"],
            "reopen_only_if": _current_problem_closeout_state()["tsp"]["reopen_only_if"],
            "do_not_reopen_condition": _current_problem_closeout_state()["tsp"]["do_not_reopen_condition"],
        },
        {
            "problem": "zdt1",
            "current_baseline_default": (
                "Q=configs/local_profiles/zdt1_diversity_injection.json; "
                "F=configs/local_profiles/zdt1_diversity_injection_fast.json"
            ),
            "active_target_status": "note_only_stress_slice + monitor_only",
            "candidate_backlog_status": "1 note_only spread candidate; 0 baseline-ready",
            "current_recommendation": _current_problem_closeout_state()["zdt1"]["current_recommendation"],
            "reopen_only_if": _current_problem_closeout_state()["zdt1"]["reopen_only_if"],
            "do_not_reopen_condition": _current_problem_closeout_state()["zdt1"]["do_not_reopen_condition"],
        },
        {
            "problem": "knapsack",
            "current_baseline_default": (
                "default=greedy_local_search; note=configs/local_profiles/knapsack_repair_local_experimental.json"
            ),
            "active_target_status": "narrow_note_only + monitor_only",
            "candidate_backlog_status": "1 note_only narrow repair candidate; 0 baseline-ready",
            "current_recommendation": _current_problem_closeout_state()["knapsack"]["current_recommendation"],
            "reopen_only_if": _current_problem_closeout_state()["knapsack"]["reopen_only_if"],
            "do_not_reopen_condition": _current_problem_closeout_state()["knapsack"]["do_not_reopen_condition"],
        },
        {
            "problem": "onemax",
            "current_baseline_default": "default=none",
            "active_target_status": "no_active_target",
            "candidate_backlog_status": "no candidates",
            "current_recommendation": _current_problem_closeout_state()["onemax"]["current_recommendation"],
            "reopen_only_if": _current_problem_closeout_state()["onemax"]["reopen_only_if"],
            "do_not_reopen_condition": _current_problem_closeout_state()["onemax"]["do_not_reopen_condition"],
        },
    ]

    note_only_candidates = [
        {
            "candidate_id": row["candidate_id"],
            "problem": row["problem"],
            "target_id": row["target_id"],
            "recommended_next_action": row["recommended_next_action"],
        }
        for row in _candidate_rows_by_lifecycle(ledger_rows, "note_only")
    ]

    status = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cycle_state": "local_optimization_cycle_1_frozen",
        "baseline_snapshot_hash": _sha256(snapshot_path),
        "baseline_check_status": baseline_check.get("status"),
        "candidate_ledger_hash": _sha256(ledger_path),
        "candidate_counts_by_lifecycle": summary.get("counts_by_lifecycle_state", {}),
        "current_default_profiles": snapshot.get("profile_paths", {}),
        "current_protocol_decisions": snapshot.get("problem_operating_decisions", {}),
        "target_statuses": target_statuses,
        "active_targets": [],
        "frozen_targets": _targets_by_decision(target_rows, "freeze_as_protocol_limitation"),
        "secondary_regression_slices": _targets_by_decision(target_rows, "secondary_regression_slice"),
        "narrow_note_targets": _targets_by_decision(target_rows, "narrow_note_only"),
        "note_only_candidates": note_only_candidates,
        "monitor_only_targets": _targets_by_decision(target_rows, "monitor_only"),
        "no_active_targets": _targets_by_decision(target_rows, "no_active_target"),
        "no_profile_changes_pending": True,
        "no_candidate_ready_to_change_baseline": (
            not summary.get("ready_for_change_request_candidates")
            and not summary.get("passed_local_guard_candidates")
        ),
        "next_allowed_work": [
            "Run python scripts/check_local_baseline.py before any new local candidate.",
            "Submit future ideas as candidate manifests under configs/local_candidates/.",
            "Reopen TSP only with a new mechanism hypothesis that targets anti-case p95/max.",
            "Reopen ZDT1 only if a candidate generalizes beyond stress slice without stable/normal regression.",
            "Keep knapsack work family-conditioned and narrow.",
            "Touch Onemax only for control drift or instrumentation regression.",
        ],
        "next_disallowed_work": [
            "Do not auto-replace configs/local_profiles/* from a candidate result.",
            "Do not reopen the old TSP PG contour family without a new mechanism story.",
            "Do not claim spread-stress-only ZDT1 gains as a new default.",
            "Do not reopen broad knapsack default discovery.",
            "Do not add adaptive/router/gate/multi-start rules in this closeout state.",
        ],
        "problem_rows": problem_rows,
        "artifact_inputs": {
            "baseline_snapshot": str(snapshot_path.relative_to(root)),
            "baseline_check": str(check_path.relative_to(root)),
            "candidate_ledger": str(ledger_path.relative_to(root)),
            "candidate_summary": str(summary_path.relative_to(root)),
            "target_registry": str(targets_path.relative_to(root)),
            "failure_hypotheses": str(hypotheses_path.relative_to(root)),
        },
        "candidate_backlog_read": {
            "total_candidates": summary.get("total_candidates", 0),
            "ready_for_change_request_candidates": summary.get("ready_for_change_request_candidates", []),
            "passed_local_guard_candidates": summary.get("passed_local_guard_candidates", []),
        },
        "failure_hypothesis_count": len(hypothesis_rows),
    }

    json_path = _resolve(root, output_json_path)
    md_path = _resolve(root, output_md_path)
    _write_json(json_path, status)
    md_lines = [
        "# Local Optimization Status",
        "",
        f"- generated at: `{status['generated_at']}`",
        f"- cycle state: `{status['cycle_state']}`",
        f"- baseline check status: `{status['baseline_check_status']}`",
        f"- baseline snapshot hash: `{status['baseline_snapshot_hash']}`",
        f"- candidate ledger hash: `{status['candidate_ledger_hash']}`",
        f"- no profile changes pending: `{status['no_profile_changes_pending']}`",
        f"- no candidate ready to change baseline: `{status['no_candidate_ready_to_change_baseline']}`",
        "",
        "## Problem Status",
        "",
        _markdown_table(
            problem_rows,
            [
                "problem",
                "current_baseline_default",
                "active_target_status",
                "candidate_backlog_status",
                "current_recommendation",
                "reopen_only_if",
                "do_not_reopen_condition",
            ],
        ),
        "",
        "## Target Buckets",
        "",
        f"- active targets: {status['active_targets']}",
        f"- frozen targets: {status['frozen_targets']}",
        f"- secondary regression slices: {status['secondary_regression_slices']}",
        f"- narrow note targets: {status['narrow_note_targets']}",
        f"- monitor-only targets: {status['monitor_only_targets']}",
        f"- no-active targets: {status['no_active_targets']}",
        "",
        "## Note-Only Candidates",
        "",
        _markdown_table(
            note_only_candidates,
            ["candidate_id", "problem", "target_id", "recommended_next_action"],
        )
        if note_only_candidates
        else "_No note-only candidates._",
        "",
        "## Allowed Work",
        "",
        *[f"- {line}" for line in status["next_allowed_work"]],
        "",
        "## Disallowed Work",
        "",
        *[f"- {line}" for line in status["next_disallowed_work"]],
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return status


def build_local_optimization_closeout(
    *,
    project_root: Path | None = None,
    status_json_path: str | Path = "artifacts/local_optimization_status.json",
    status_md_path: str | Path = "artifacts/local_optimization_status.md",
    reopen_json_path: str | Path = "artifacts/local_reopen_criteria.json",
    reopen_md_path: str | Path = "docs/local_reopen_criteria.md",
    backlog_json_path: str | Path = "artifacts/local_candidate_backlog_closeout.json",
    backlog_md_path: str | Path = "artifacts/local_candidate_backlog_closeout.md",
) -> dict[str, Any]:
    root = project_root or DEFAULT_PROJECT_ROOT
    status = build_local_optimization_status(
        project_root=root,
        output_json_path=status_json_path,
        output_md_path=status_md_path,
    )
    reopen = build_local_reopen_criteria(
        project_root=root,
        output_json_path=reopen_json_path,
        output_md_path=reopen_md_path,
    )
    backlog = build_local_candidate_backlog_closeout(
        project_root=root,
        output_json_path=backlog_json_path,
        output_md_path=backlog_md_path,
    )
    return {
        "status": status,
        "reopen_criteria": reopen,
        "candidate_backlog_closeout": backlog,
    }
