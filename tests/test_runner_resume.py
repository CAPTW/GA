from __future__ import annotations

import json
from pathlib import Path

from ga_lab.experiment.runner_resume import (
    RunnerResumeKey,
    build_resume_key,
    filter_missing_runs,
    load_completed_run_index,
    merge_resume_results,
    plan_runner_resume,
)


def _row(
    *,
    problem: str = "constrained_zdt_box_toy",
    strategy: str = "constrained_nsga2_constraint_domination",
    seed: int = 0,
    budget: int = 20,
    actual: int = 20,
    status: str = "success",
) -> dict[str, object]:
    return {
        "problem": problem,
        "strategy": strategy,
        "seed": seed,
        "requested_budget": budget,
        "actual_evaluations": actual,
        "dimension": 6,
        "tolerance": 1e-8,
        "status": status,
    }


def test_build_resume_key_creates_stable_deterministic_key() -> None:
    left = build_resume_key(_row(seed=1))
    right = RunnerResumeKey(
        problem="constrained_zdt_box_toy",
        algorithm="constrained_nsga2_constraint_domination",
        seed=1,
        budget=20,
        dimension=6,
        tolerance=1e-8,
    )

    assert left == right
    assert left.stable_id() == right.stable_id()
    assert json.loads(left.stable_id())["algorithm"] == "constrained_nsga2_constraint_domination"


def test_completed_row_requires_success_and_exact_evaluations(tmp_path: Path) -> None:
    artifact = tmp_path / "source.json"
    artifact.write_text(json.dumps({"rows": [_row(seed=0), _row(seed=1, actual=19)]}), encoding="utf-8")

    index = load_completed_run_index(artifact)

    assert build_resume_key(_row(seed=0)).stable_id() in index.completed_keys
    assert build_resume_key(_row(seed=1, actual=19)).stable_id() not in index.completed_keys
    assert index.warnings


def test_failed_and_skipped_rows_are_not_completed(tmp_path: Path) -> None:
    artifact = tmp_path / "source.json"
    artifact.write_text(
        json.dumps({"rows": [_row(seed=0, status="failed"), _row(seed=1, status="skipped")]}),
        encoding="utf-8",
    )

    index = load_completed_run_index(artifact)

    assert not index.completed_keys
    assert len(index.failed_keys) == 1
    assert len(index.skipped_keys) == 1


def test_missing_key_fields_create_warning(tmp_path: Path) -> None:
    artifact = tmp_path / "source.json"
    artifact.write_text(json.dumps({"rows": [{"status": "success"}]}), encoding="utf-8")

    index = load_completed_run_index(artifact)

    assert not index.completed_keys
    assert any("missing resume key field" in warning for warning in index.warnings)


def test_filter_missing_runs_returns_only_missing_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "source.json"
    artifact.write_text(json.dumps({"rows": [_row(seed=0)]}), encoding="utf-8")
    index = load_completed_run_index(artifact)
    planned = [_row(seed=0), _row(seed=1)]

    missing = filter_missing_runs(planned, completed_index=index, key_fn=build_resume_key)
    plan = plan_runner_resume([build_resume_key(row) for row in planned], index)

    assert [row["seed"] for row in missing] == [1]
    assert plan.total_planned == 2
    assert plan.completed_count == 1
    assert plan.missing_count == 1
    json.dumps(plan.to_dict())


def test_merge_resume_results_marks_row_origin(tmp_path: Path) -> None:
    artifact = tmp_path / "source.json"
    source_row = _row(seed=0)
    new_row = _row(seed=1)

    merged = merge_resume_results(
        source_rows=[source_row],
        new_rows=[new_row],
        source_artifact=artifact,
    )

    origins = {row["seed"]: row["row_origin"] for row in merged}
    assert origins == {0: "source_artifact", 1: "newly_executed"}
    assert all("resume_key" in row for row in merged)
    json.dumps(merged)
