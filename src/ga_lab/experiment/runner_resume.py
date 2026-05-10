from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class RunnerResumeKey:
    problem: str
    algorithm: str
    seed: int
    budget: int
    dimension: int
    variant: str | None = None
    tolerance: float | None = None
    artifact_context: str | None = None
    config_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "budget": self.budget,
            "dimension": self.dimension,
            "variant": self.variant,
            "tolerance": self.tolerance,
            "artifact_context": self.artifact_context,
            "config_hash": self.config_hash,
        }

    def stable_id(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(slots=True)
class CompletedRunIndex:
    completed_keys: set[str] = field(default_factory=set)
    skipped_keys: set[str] = field(default_factory=set)
    failed_keys: set[str] = field(default_factory=set)
    source_artifact: str | None = None
    warnings: list[str] = field(default_factory=list)
    completed_rows_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_keys": sorted(self.completed_keys),
            "skipped_keys": sorted(self.skipped_keys),
            "failed_keys": sorted(self.failed_keys),
            "source_artifact": self.source_artifact,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RunnerResumePlan:
    total_planned: int
    completed_count: int
    missing_count: int
    failed_count: int
    to_run_keys: list[RunnerResumeKey]
    skipped_existing_keys: list[RunnerResumeKey]
    resume_mode: str
    source_artifact: str | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_planned": self.total_planned,
            "completed_count": self.completed_count,
            "missing_count": self.missing_count,
            "failed_count": self.failed_count,
            "to_run_keys": [key.to_dict() for key in self.to_run_keys],
            "skipped_existing_keys": [key.to_dict() for key in self.skipped_existing_keys],
            "resume_mode": self.resume_mode,
            "source_artifact": self.source_artifact,
            "warnings": list(self.warnings),
        }


def _required_int(row: dict[str, Any], *names: str) -> int:
    for name in names:
        value = row.get(name)
        if value is not None:
            return int(value)
    raise KeyError(names[0])


def _required_str(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return str(value)
    raise KeyError(names[0])


def build_resume_key(
    row: dict[str, Any],
    *,
    variant: str | None = None,
    artifact_context: str | None = None,
    config_hash: str | None = None,
) -> RunnerResumeKey:
    return RunnerResumeKey(
        problem=_required_str(row, "problem", "benchmark"),
        algorithm=_required_str(row, "algorithm", "strategy"),
        seed=_required_int(row, "seed"),
        budget=_required_int(row, "requested_budget", "budget"),
        dimension=_required_int(row, "dimension"),
        variant=variant if variant is not None else row.get("variant"),
        tolerance=(
            float(row["tolerance"])
            if row.get("tolerance") is not None
            else None
        ),
        artifact_context=artifact_context if artifact_context is not None else row.get("artifact_context"),
        config_hash=config_hash if config_hash is not None else row.get("config_hash"),
    )


def _is_completed(row: dict[str, Any]) -> tuple[bool, str | None]:
    status = row.get("status")
    if status != "success":
        return False, None
    requested_budget = row.get("requested_budget", row.get("budget"))
    actual_evaluations = row.get("actual_evaluations")
    if requested_budget is None or actual_evaluations is None:
        return False, "missing requested_budget or actual_evaluations"
    if int(requested_budget) != int(actual_evaluations):
        return False, "actual_evaluations mismatch"
    return True, None


def load_completed_run_index(path: str | Path) -> CompletedRunIndex:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Resume source artifact has no rows list: {source}")
    index = CompletedRunIndex(source_artifact=str(source))
    for offset, row in enumerate(rows):
        if not isinstance(row, dict):
            index.warnings.append(f"row {offset}: non-object row ignored")
            continue
        try:
            key = build_resume_key(row)
        except (KeyError, TypeError, ValueError) as exc:
            index.warnings.append(f"row {offset}: missing resume key field or invalid value: {exc}")
            continue
        key_id = key.stable_id()
        if row.get("status") == "skipped":
            index.skipped_keys.add(key_id)
            continue
        if row.get("status") == "failed":
            index.failed_keys.add(key_id)
            continue
        completed, warning = _is_completed(row)
        if completed:
            index.completed_keys.add(key_id)
            index.completed_rows_by_key[key_id] = dict(row)
        elif warning:
            index.warnings.append(f"row {offset}: {warning}; not completed")
    return index


def plan_runner_resume(
    planned_keys: Sequence[RunnerResumeKey],
    completed_index: CompletedRunIndex,
    *,
    resume_mode: str = "skip-completed",
) -> RunnerResumePlan:
    to_run: list[RunnerResumeKey] = []
    skipped_existing: list[RunnerResumeKey] = []
    for key in planned_keys:
        if key.stable_id() in completed_index.completed_keys:
            skipped_existing.append(key)
        else:
            to_run.append(key)
    return RunnerResumePlan(
        total_planned=len(planned_keys),
        completed_count=len(skipped_existing),
        missing_count=len(to_run),
        failed_count=len(completed_index.failed_keys),
        to_run_keys=to_run,
        skipped_existing_keys=skipped_existing,
        resume_mode=resume_mode,
        source_artifact=completed_index.source_artifact,
        warnings=list(completed_index.warnings),
    )


def filter_missing_runs(
    planned_items: Iterable[dict[str, Any]],
    *,
    completed_index: CompletedRunIndex,
    key_fn: Callable[[dict[str, Any]], RunnerResumeKey],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for item in planned_items:
        if key_fn(item).stable_id() not in completed_index.completed_keys:
            missing.append(item)
    return missing


def _with_resume_fields(
    row: dict[str, Any],
    *,
    row_origin: str,
    source_artifact: str | Path | None,
) -> dict[str, Any]:
    output = dict(row)
    output["row_origin"] = row_origin
    output["resume_key"] = build_resume_key(output).to_dict()
    if source_artifact is not None:
        output["resume_source_artifact"] = str(source_artifact)
    return output


def merge_resume_results(
    *,
    source_rows: Sequence[dict[str, Any]],
    new_rows: Sequence[dict[str, Any]],
    source_artifact: str | Path | None,
) -> list[dict[str, Any]]:
    return [
        *[
            _with_resume_fields(row, row_origin="source_artifact", source_artifact=source_artifact)
            for row in source_rows
        ],
        *[
            _with_resume_fields(row, row_origin="newly_executed", source_artifact=source_artifact)
            for row in new_rows
        ],
    ]


def write_resume_report(path: str | Path, plan: RunnerResumePlan, summary: dict[str, Any]) -> Path:
    report_path = Path(path)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing resume report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Runner-level Resume Report",
        "",
        "| item | value |",
        "|---|---|",
        f"| source_artifact | {plan.source_artifact} |",
        f"| resume_mode | {plan.resume_mode} |",
        f"| total_planned | {plan.total_planned} |",
        f"| completed_from_source | {summary.get('completed_from_source', plan.completed_count)} |",
        f"| newly_executed | {summary.get('newly_executed', plan.missing_count)} |",
        f"| failed_existing | {summary.get('failed_existing', plan.failed_count)} |",
        f"| warnings | {len(plan.warnings)} |",
        "",
        "This is runner-level skip-completed resume only. It is not algorithm mid-run checkpoint/resume.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
