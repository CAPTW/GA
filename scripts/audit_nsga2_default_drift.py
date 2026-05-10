from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.config import load_config
from ga_lab.experiment.diversity_diagnostics import evaluate_diversity_diagnostics
from ga_lab.experiment.external_mo_comparators import result_to_front_row, run_internal_nsga2
from ga_lab.experiment.mo_metrics import reference_front_for_problem
from ga_lab.experiment.nsga2_candidate_suite import build_problem_config, mo_candidate_suite_specs

REFERENCE_ARTIFACTS = (
    "artifacts/nsga2_candidate_suite_validation_results_candidate_d_cr.json",
    "artifacts/nsga2_candidate_suite_validation_results_zdt_suite.json",
    "artifacts/external_mo_comparison_results_installed.json",
)

CANDIDATE_METADATA_KEYS = {
    "candidate_id",
    "default_changed",
    "promotion_status",
    "base_candidate_id",
    "source_diagnosis_report",
    "source_artifacts",
}
DIAGNOSTICS_METADATA_KEYS = {
    "diagnostics_enabled",
    "nsga2_diagnostics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit default NSGA-II drift against previous internal baseline artifacts."
    )
    parser.add_argument("--config", default="configs/smoke/zdt1_nsga2_smoke.json")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output-root", default="outputs/nsga2_default_drift_audit")
    parser.add_argument(
        "--results-base",
        default="nsga2_default_drift_audit_results",
        help="Base artifact name used for the JSON/CSV audit outputs.",
    )
    parser.add_argument(
        "--report-base",
        default="nsga2_default_drift_audit_report",
        help="Base artifact name used for the Markdown audit report.",
    )
    return parser.parse_args()


def _vector_signature(vectors: list[list[float]], *, precision: int = 12) -> tuple[tuple[float, ...], ...]:
    normalized = [tuple(round(float(value), precision) for value in vector) for vector in vectors]
    return tuple(sorted(normalized))


def _load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _problem_list(payload: dict[str, Any]) -> list[str]:
    if "selected_problems" in payload:
        return [str(item) for item in payload["selected_problems"]]
    if "problem" in payload:
        return [str(payload["problem"])]
    raise ValueError("Artifact does not contain problem selection metadata")


def _budget(payload: dict[str, Any]) -> int:
    budget = payload.get("budget") or payload.get("requested_budget")
    if not isinstance(budget, int):
        raise ValueError("Artifact does not contain integer budget")
    return budget


def _raw_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("raw_rows")
    if not isinstance(rows, list):
        raise ValueError("Artifact does not contain raw_rows")
    return rows


def _rerun_internal_rows(
    *,
    config_path: Path,
    problems: list[str],
    seeds: list[int],
    output_root: Path,
) -> list[dict[str, Any]]:
    base_config = load_config(config_path)
    rows: list[dict[str, Any]] = []
    for problem_name in problems:
        spec = mo_candidate_suite_specs()[problem_name]
        config = build_problem_config(base_config, spec)
        reference_front = reference_front_for_problem(
            problem_name,
            objective_count=spec.objectives,
        )
        for seed in seeds:
            result = run_internal_nsga2(config, seed=seed, output_root=str(output_root))
            row = result_to_front_row(
                result,
                reference_front=reference_front,
                reference_point=list(spec.hv_reference_point),
            )
            row.update(
                evaluate_diversity_diagnostics(
                    row.get("objective_vectors", []),
                    directions=[
                        bool(value)
                        for value in row.get("metadata", {}).get("objective_directions", [False] * spec.objectives)
                    ],
                    decision_vectors=row.get("front_decision_vectors") or row.get("decision_vectors"),
                )
            )
            rows.append(row)
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None
    return mean(values)


def _compare_rows(
    previous_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    previous_by_key = {
        (str(row["problem"]), int(row["seed"])): row
        for row in previous_rows
        if row.get("algorithm") == "internal_nsga2"
    }
    current_by_key = {
        (str(row["problem"]), int(row["seed"])): row
        for row in current_rows
        if row.get("algorithm") == "internal_nsga2"
    }
    metrics = [
        "hypervolume_2d",
        "reference_front_distance",
        "generational_distance",
        "inverted_generational_distance",
        "spacing",
        "nondominated_count",
        "actual_evaluations",
    ]
    row_comparisons: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    candidate_metadata_leak = False
    diagnostics_metadata_leak = False
    evaluation_mismatch = False
    objective_signature_mismatch = False

    common_keys = sorted(set(previous_by_key) & set(current_by_key))
    for key in common_keys:
        problem, seed = key
        previous = previous_by_key[key]
        current = current_by_key[key]
        previous_signature = _vector_signature(previous.get("objective_vectors", []))
        current_signature = _vector_signature(current.get("objective_vectors", []))
        objective_match = previous_signature == current_signature
        if not objective_match:
            objective_signature_mismatch = True
        leaked_keys = sorted(CANDIDATE_METADATA_KEYS & set(current.get("metadata", {}).keys()))
        leaked_diagnostics_keys = sorted(
            DIAGNOSTICS_METADATA_KEYS & set(current.get("metadata", {}).keys())
        )
        if leaked_keys:
            candidate_metadata_leak = True
        if leaked_diagnostics_keys:
            diagnostics_metadata_leak = True
        if int(previous.get("actual_evaluations", -1)) != int(current.get("actual_evaluations", -2)):
            evaluation_mismatch = True
        row_comparisons.append(
            {
                "problem": problem,
                "seed": seed,
                "previous_actual_evaluations": previous.get("actual_evaluations"),
                "current_actual_evaluations": current.get("actual_evaluations"),
                "objective_signature_match": objective_match,
                "candidate_metadata_leak": bool(leaked_keys),
                "leaked_metadata_keys": leaked_keys,
                "diagnostics_metadata_leak": bool(leaked_diagnostics_keys),
                "leaked_diagnostics_keys": leaked_diagnostics_keys,
                "previous_hv": previous.get("hypervolume_2d"),
                "current_hv": current.get("hypervolume_2d"),
                "previous_distance": previous.get("reference_front_distance"),
                "current_distance": current.get("reference_front_distance"),
                "previous_spacing": previous.get("spacing"),
                "current_spacing": current.get("spacing"),
            }
        )

    for problem in sorted({key[0] for key in common_keys}):
        previous_bucket = [previous_by_key[key] for key in common_keys if key[0] == problem]
        current_bucket = [current_by_key[key] for key in common_keys if key[0] == problem]
        for metric in metrics:
            previous_mean = _mean(previous_bucket, metric)
            current_mean = _mean(current_bucket, metric)
            delta = None
            drift = False
            if previous_mean is not None and current_mean is not None:
                delta = current_mean - previous_mean
                drift = not math.isclose(
                    previous_mean,
                    current_mean,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            interpretation = (
                "exact match"
                if drift is False and delta is not None
                else "mismatch"
                if drift
                else "n/a"
            )
            summary_rows.append(
                {
                    "problem": problem,
                    "metric": metric,
                    "previous_mean": previous_mean,
                    "current_mean": current_mean,
                    "delta": delta,
                    "drift_detected": drift,
                    "interpretation": interpretation,
                }
            )

    overall = {
        "candidate_metadata_leak": candidate_metadata_leak,
        "diagnostics_metadata_leak": diagnostics_metadata_leak,
        "actual_evaluations_mismatch": evaluation_mismatch,
        "objective_signature_mismatch": objective_signature_mismatch,
        "drift_detected": (
            candidate_metadata_leak
            or diagnostics_metadata_leak
            or evaluation_mismatch
            or objective_signature_mismatch
            or any(bool(row["drift_detected"]) for row in summary_rows)
        ),
    }
    return summary_rows, row_comparisons, overall


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return lines


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = ["# NSGA-II Default Drift Audit Report", ""]
    lines.append("## 1. Executive Summary")
    lines.append("")
    drift = bool(payload["overall"]["drift_detected"])
    lines.append(
        f"- 최종 판정: **{'DRIFT DETECTED' if drift else 'NO DRIFT'}**"
    )
    lines.append(
        f"- candidate metadata leak: `{payload['overall']['candidate_metadata_leak']}`"
    )
    lines.append(
        f"- diagnostics metadata leak: `{payload['overall']['diagnostics_metadata_leak']}`"
    )
    lines.append(
        f"- actual evaluations mismatch: `{payload['overall']['actual_evaluations_mismatch']}`"
    )
    lines.append(
        f"- objective signature mismatch: `{payload['overall']['objective_signature_mismatch']}`"
    )
    lines.append(
        "- 이번 감사에서는 기본 internal NSGA-II만 다시 실행했고, candidate variant metadata가 default 경로로 섞여 들어가는지 함께 확인했다."
    )
    lines.append("")
    lines.append("## 2. Reference Artifacts")
    lines.append("")
    lines.extend(
        _table(
            payload["reference_artifacts"],
            ["artifact", "problems", "seed_count", "budget", "comparison_status"],
        )
    )
    lines.append("")
    lines.append("## 3. Default Drift Audit")
    lines.append("")
    lines.extend(
        _table(
            payload["summary_rows"],
            [
                "artifact",
                "problem",
                "metric",
                "previous_mean",
                "current_mean",
                "delta",
                "drift_detected",
                "interpretation",
            ],
        )
    )
    lines.append("")
    lines.append("## 4. Candidate Isolation Checks")
    lines.append("")
    lines.extend(
        _table(
            payload["isolation_rows"],
            [
                "artifact",
                "problem",
                "seed",
                "candidate_metadata_leak",
                "leaked_metadata_keys",
                "diagnostics_metadata_leak",
                "leaked_diagnostics_keys",
                "objective_signature_match",
                "previous_actual_evaluations",
                "current_actual_evaluations",
            ],
        )
    )
    lines.append("")
    lines.append("## 5. Conclusion")
    lines.append("")
    if drift:
        lines.append("- drift audit failed. 새 h-lite candidate 실험은 이 상태에서 진행하면 안 된다.")
    else:
        lines.append("- default drift는 발견되지 않았다. 기본 NSGA-II 경로는 이전 artifact와 같은 품질 metric, objective signature, evaluation count를 재현했다.")
        lines.append("- candidate metadata leak도 없었다. candidate 관련 로직은 opt-in 경로에서만 활성화된다고 봐도 된다.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifact_root = PROJECT_ROOT / args.artifact_root
    output_root = PROJECT_ROOT / args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    isolation_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    overall_flags = {
        "candidate_metadata_leak": False,
        "diagnostics_metadata_leak": False,
        "actual_evaluations_mismatch": False,
        "objective_signature_mismatch": False,
        "drift_detected": False,
    }

    for artifact_rel in REFERENCE_ARTIFACTS:
        artifact_path = PROJECT_ROOT / artifact_rel
        payload = _load_artifact(artifact_path)
        config_path = PROJECT_ROOT / str(payload.get("config_path", args.config))
        problems = _problem_list(payload)
        seeds = [int(seed) for seed in payload["seeds"]]
        current_rows = _rerun_internal_rows(
            config_path=config_path,
            problems=problems,
            seeds=seeds,
            output_root=output_root / artifact_path.stem,
        )
        previous_rows = _raw_rows(payload)
        comparison_rows, row_comparisons, overall = _compare_rows(previous_rows, current_rows)
        for row in comparison_rows:
            row["artifact"] = artifact_path.name
            summary_rows.append(row)
        for row in row_comparisons:
            row["artifact"] = artifact_path.name
            isolation_rows.append(row)
        reference_rows.append(
            {
                "artifact": artifact_path.name,
                "problems": ", ".join(problems),
                "seed_count": len(seeds),
                "budget": _budget(payload),
                "comparison_status": "drift" if overall["drift_detected"] else "match",
            }
        )
        for key, value in overall.items():
            overall_flags[key] = bool(overall_flags[key] or value)

    json_path = artifact_root / f"{args.results_base}.json"
    csv_path = artifact_root / f"{args.results_base}.csv"
    report_path = artifact_root / f"{args.report_base}.md"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": "scripts/audit_nsga2_default_drift.py",
        "reference_artifacts": reference_rows,
        "summary_rows": summary_rows,
        "isolation_rows": isolation_rows,
        "overall": overall_flags,
    }
    _write_json(json_path, payload)
    _write_csv(csv_path, summary_rows)
    _write_report(report_path, payload)


if __name__ == "__main__":
    main()
