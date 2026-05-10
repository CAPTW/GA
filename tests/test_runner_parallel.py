from __future__ import annotations

import json
import pytest

from ga_lab.experiment.runner_parallel import (
    ParallelExecutionConfig,
    execute_runner_rows,
    parallel_summary_to_dict,
    sort_rows_deterministically,
)


def _row_fn(item: dict[str, int]) -> dict[str, int | str]:
    return {
        "problem": "toy",
        "strategy": "demo",
        "seed": item["seed"],
        "requested_budget": item["budget"],
        "actual_evaluations": item["budget"],
        "status": "success",
    }


def _failing_row_fn(item: dict[str, int]) -> dict[str, int | str]:
    if item["seed"] == 1:
        raise RuntimeError("synthetic row failure")
    return _row_fn(item)


def _slow_order_row_fn(item: dict[str, int]) -> dict[str, int | str]:
    return _row_fn(item)


def _row_key(item: dict[str, int]) -> str:
    return f"toy|demo|{item['seed']}|{item['budget']}"


class _UnpicklablePayload:
    def __init__(self) -> None:
        self.callback = lambda: None


def test_serial_backend_preserves_input_order() -> None:
    items = [{"seed": 2, "budget": 10}, {"seed": 0, "budget": 10}]

    summary = execute_runner_rows(
        items,
        _row_fn,
        config=ParallelExecutionConfig(backend="serial"),
        row_key_fn=_row_key,
    )

    assert [result.row["seed"] for result in summary.successful_results] == [2, 0]
    assert summary.success_count == 2
    assert summary.failure_count == 0
    json.dumps(parallel_summary_to_dict(summary))


def test_thread_backend_preserves_deterministic_order() -> None:
    items = [{"seed": 3, "budget": 10}, {"seed": 1, "budget": 10}, {"seed": 2, "budget": 10}]

    summary = execute_runner_rows(
        items,
        _row_fn,
        config=ParallelExecutionConfig(backend="thread", workers=2),
        row_key_fn=_row_key,
    )

    assert [result.row["seed"] for result in summary.successful_results] == [3, 1, 2]
    assert summary.backend == "thread"
    assert summary.deterministic_ordering is True


def test_process_backend_runs_top_level_picklable_row_function() -> None:
    items = [{"seed": 0, "budget": 10}]

    summary = execute_runner_rows(
        items,
        _row_fn,
        config=ParallelExecutionConfig(
            backend="process",
            workers=1,
            allow_process_backend=True,
        ),
        row_key_fn=_row_key,
    )

    assert summary.success_count == 1
    assert summary.failure_count == 0
    assert summary.successful_results[0].row["seed"] == 0


def test_process_backend_requires_allow_process_backend() -> None:
    with pytest.raises(ValueError, match="allow_process_backend=True"):
        execute_runner_rows(
            [{"seed": 0, "budget": 10}],
            _row_fn,
            config=ParallelExecutionConfig(backend="process", workers=1),
            row_key_fn=_row_key,
        )


def test_process_backend_reports_lambda_worker_as_pickling_failure() -> None:
    items = [{"seed": 0, "budget": 10}, {"seed": 1, "budget": 10}]

    summary = execute_runner_rows(
        items,
        lambda item: _row_fn(item),  # noqa: E731 - intentional unpicklable process worker
        config=ParallelExecutionConfig(
            backend="process",
            workers=1,
            allow_process_backend=True,
        ),
        row_key_fn=_row_key,
    )

    assert summary.success_count == 0
    assert summary.failure_count == 2
    assert all(result.failure["backend"] == "process" for result in summary.failed_results)
    assert all(
        result.failure["exception_type"] == "ProcessBackendPicklingError"
        for result in summary.failed_results
    )
    assert all("process_backend_pickling_failure" in result.failure for result in summary.failed_results)


def test_process_backend_reports_unpicklable_payload_as_pickling_failure() -> None:
    item = _UnpicklablePayload()

    summary = execute_runner_rows(
        [item],
        _row_fn,
        config=ParallelExecutionConfig(
            backend="process",
            workers=1,
            allow_process_backend=True,
        ),
    )

    assert summary.success_count == 0
    assert summary.failure_count == 1
    failure = summary.failed_results[0].failure
    assert failure["backend"] == "process"
    assert failure["exception_type"] == "ProcessBackendPicklingError"
    assert failure["process_backend_pickling_failure"] is True


def test_process_backend_captures_worker_exception_with_row_key_and_backend() -> None:
    items = [{"seed": 0, "budget": 10}, {"seed": 1, "budget": 10}, {"seed": 2, "budget": 10}]

    summary = execute_runner_rows(
        items,
        _failing_row_fn,
        config=ParallelExecutionConfig(
            backend="process",
            workers=2,
            fail_fast=False,
            allow_process_backend=True,
        ),
        row_key_fn=_row_key,
    )

    assert summary.success_count == 2
    assert summary.failure_count == 1
    failure = summary.failed_results[0].failure
    assert failure["row_key"] == "toy|demo|1|10"
    assert failure["backend"] == "process"
    assert failure["exception_type"] == "RuntimeError"
    assert "synthetic row failure" in failure["message"]


def test_process_backend_preserves_deterministic_order() -> None:
    items = [{"seed": 3, "budget": 10}, {"seed": 1, "budget": 10}, {"seed": 2, "budget": 10}]

    summary = execute_runner_rows(
        items,
        _slow_order_row_fn,
        config=ParallelExecutionConfig(
            backend="process",
            workers=2,
            allow_process_backend=True,
        ),
        row_key_fn=_row_key,
    )

    assert [result.row["seed"] for result in summary.successful_results] == [3, 1, 2]
    assert [result.order_index for result in summary.row_results] == [0, 1, 2]
    json.dumps(parallel_summary_to_dict(summary))


def test_parallel_execution_config_rejects_invalid_workers() -> None:
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        ParallelExecutionConfig(backend="process", workers=0, allow_process_backend=True)


def test_exception_includes_row_key_and_backend_when_fail_fast_false() -> None:
    items = [{"seed": 0, "budget": 10}, {"seed": 1, "budget": 10}, {"seed": 2, "budget": 10}]

    summary = execute_runner_rows(
        items,
        _failing_row_fn,
        config=ParallelExecutionConfig(backend="thread", workers=2, fail_fast=False),
        row_key_fn=_row_key,
    )

    assert summary.success_count == 2
    assert summary.failure_count == 1
    failure = summary.failed_results[0].failure
    assert failure["row_key"] == "toy|demo|1|10"
    assert failure["backend"] == "thread"
    assert failure["exception_type"] == "RuntimeError"
    assert "synthetic row failure" in failure["message"]


def test_sort_rows_deterministically_is_stable() -> None:
    rows = [
        {"benchmark": "b", "strategy": "s", "seed": 1, "requested_budget": 20},
        {"benchmark": "a", "strategy": "s", "seed": 0, "requested_budget": 20},
        {"benchmark": "a", "strategy": "r", "seed": 0, "requested_budget": 20},
    ]

    sorted_rows = sort_rows_deterministically(rows)

    assert [(row["benchmark"], row["strategy"], row["seed"]) for row in sorted_rows] == [
        ("a", "r", 0),
        ("a", "s", 0),
        ("b", "s", 1),
    ]
