from __future__ import annotations

import concurrent.futures
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence


ParallelBackend = Literal["serial", "thread", "process"]


class ProcessBackendPicklingError(Exception):
    """Configuration failure raised when process backend inputs are not pickle-safe."""


@dataclass(frozen=True, slots=True)
class ParallelExecutionConfig:
    backend: ParallelBackend = "serial"
    workers: int = 1
    preserve_order: bool = True
    fail_fast: bool = False
    timeout_seconds: float | None = None
    allow_process_backend: bool = False
    record_worker_metadata: bool = True

    def __post_init__(self) -> None:
        if self.backend not in {"serial", "thread", "process"}:
            raise ValueError(f"Unsupported row execution backend: {self.backend}")
        if not isinstance(self.workers, int) or isinstance(self.workers, bool) or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")


@dataclass(slots=True)
class ParallelRowResult:
    order_index: int
    row_key: str
    backend: str
    row: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None and self.row is not None


@dataclass(slots=True)
class ParallelExecutionSummary:
    backend: str
    workers: int
    total_submitted: int
    deterministic_ordering: bool
    row_results: list[ParallelRowResult] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def successful_results(self) -> list[ParallelRowResult]:
        return [result for result in self.row_results if result.succeeded]

    @property
    def failed_results(self) -> list[ParallelRowResult]:
        return [result for result in self.row_results if not result.succeeded]

    @property
    def success_count(self) -> int:
        return len(self.successful_results)

    @property
    def failure_count(self) -> int:
        return len(self.failed_results)


def _default_row_key(item: Any, order_index: int) -> str:
    if isinstance(item, dict):
        for name in ("resume_key", "row_key", "key"):
            value = item.get(name)
            if value is not None:
                return str(value)
        parts = [
            item.get("problem", item.get("benchmark", "")),
            item.get("strategy", item.get("algorithm", "")),
            item.get("seed", ""),
            item.get("requested_budget", item.get("budget", "")),
            item.get("dimension", ""),
        ]
        if any(part != "" for part in parts):
            return "|".join(str(part) for part in parts)
    return str(order_index)


def _failure_payload(
    *,
    exc: BaseException,
    row_key: str,
    order_index: int,
    backend: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "row_key": row_key,
        "order_index": order_index,
        "backend": backend,
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }
    if extra:
        payload.update(extra)
    return payload


def _execute_one(
    item: Any,
    row_fn: Callable[[Any], dict[str, Any]],
    order_index: int,
    row_key: str,
    backend: str,
) -> ParallelRowResult:
    try:
        row = row_fn(item)
    except Exception as exc:  # noqa: BLE001 - preserve row failure in runner summary
        return ParallelRowResult(
            order_index=order_index,
            row_key=row_key,
            backend=backend,
            failure=_failure_payload(
                exc=exc,
                row_key=row_key,
                order_index=order_index,
                backend=backend,
            ),
        )
    return ParallelRowResult(order_index=order_index, row_key=row_key, backend=backend, row=dict(row))


def execute_rows_serial(
    items: Sequence[Any],
    row_fn: Callable[[Any], dict[str, Any]],
    *,
    config: ParallelExecutionConfig | None = None,
    row_key_fn: Callable[[Any], str] | None = None,
) -> ParallelExecutionSummary:
    execution_config = config or ParallelExecutionConfig()
    results: list[ParallelRowResult] = []
    for order_index, item in enumerate(items):
        row_key = row_key_fn(item) if row_key_fn is not None else _default_row_key(item, order_index)
        result = _execute_one(item, row_fn, order_index, row_key, "serial")
        results.append(result)
        if result.failure is not None and execution_config.fail_fast:
            break
    return ParallelExecutionSummary(
        backend="serial",
        workers=1,
        total_submitted=len(items),
        deterministic_ordering=True,
        row_results=results,
    )


def _execute_rows_with_executor(
    executor_factory: Callable[[int], concurrent.futures.Executor],
    items: Sequence[Any],
    row_fn: Callable[[Any], dict[str, Any]],
    *,
    config: ParallelExecutionConfig,
    row_key_fn: Callable[[Any], str] | None,
    backend: str,
) -> ParallelExecutionSummary:
    indexed_items = [
        (
            order_index,
            row_key_fn(item) if row_key_fn is not None else _default_row_key(item, order_index),
            item,
        )
        for order_index, item in enumerate(items)
    ]
    results_by_index: dict[int, ParallelRowResult] = {}
    with executor_factory(config.workers) as executor:
        futures = {
            executor.submit(_execute_one, item, row_fn, order_index, row_key, backend): (
                order_index,
                row_key,
            )
            for order_index, row_key, item in indexed_items
        }
        for future in concurrent.futures.as_completed(futures, timeout=config.timeout_seconds):
            order_index, row_key = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - executor-level failure
                result = ParallelRowResult(
                    order_index=order_index,
                    row_key=row_key,
                    backend=backend,
                    failure=_failure_payload(
                        exc=exc,
                        row_key=row_key,
                        order_index=order_index,
                        backend=backend,
                    ),
                )
            results_by_index[order_index] = result
            if result.failure is not None and config.fail_fast:
                for pending in futures:
                    pending.cancel()
                break
    ordered_results = [results_by_index[index] for index in sorted(results_by_index)]
    return ParallelExecutionSummary(
        backend=backend,
        workers=config.workers,
        total_submitted=len(items),
        deterministic_ordering=True,
        row_results=ordered_results,
    )


def _process_pickling_failure(
    *,
    exc: BaseException,
    row_key: str,
    order_index: int,
) -> ParallelRowResult:
    wrapped = ProcessBackendPicklingError(str(exc))
    return ParallelRowResult(
        order_index=order_index,
        row_key=row_key,
        backend="process",
        failure=_failure_payload(
            exc=wrapped,
            row_key=row_key,
            order_index=order_index,
            backend="process",
            extra={
                "process_backend_pickling_failure": True,
                "failure_reason": "process_backend_pickling_failure",
            },
        ),
    )


def _process_resource_warnings(config: ParallelExecutionConfig) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    cpu_count = os.cpu_count()
    if cpu_count is not None and config.workers > cpu_count:
        warnings.append(
            {
                "type": "process_worker_oversubscription_warning",
                "message": (
                    f"process workers={config.workers} exceeds cpu_count={cpu_count}; "
                    "runtime speedup is not guaranteed"
                ),
                "workers": config.workers,
                "cpu_count": cpu_count,
            }
        )
    if config.timeout_seconds is not None:
        warnings.append(
            {
                "type": "process_timeout_policy_warning",
                "message": "process timeout_seconds is passed to future collection but is not a production timeout guarantee",
                "timeout_seconds": config.timeout_seconds,
            }
        )
    return warnings


def _execute_process_picklable_rows(
    indexed_items: Sequence[tuple[int, str, Any]],
    row_fn: Callable[[Any], dict[str, Any]],
    *,
    config: ParallelExecutionConfig,
) -> list[ParallelRowResult]:
    results_by_index: dict[int, ParallelRowResult] = {}
    executor: concurrent.futures.Executor | None = None
    try:
        try:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=config.workers)
            futures = {
                executor.submit(_execute_one, item, row_fn, order_index, row_key, "process"): (
                    order_index,
                    row_key,
                )
                for order_index, row_key, item in indexed_items
            }
        except Exception as exc:  # noqa: BLE001 - process pool startup/submission failure
            return [
                ParallelRowResult(
                    order_index=order_index,
                    row_key=row_key,
                    backend="process",
                    failure=_failure_payload(
                        exc=exc,
                        row_key=row_key,
                        order_index=order_index,
                        backend="process",
                        extra={"failure_reason": "process_pool_startup_or_submission_failure"},
                    ),
                )
                for order_index, row_key, _item in indexed_items
            ]
        try:
            completed = concurrent.futures.as_completed(
                futures,
                timeout=config.timeout_seconds,
            )
            for future in completed:
                order_index, row_key = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - executor-level failure
                    result = ParallelRowResult(
                        order_index=order_index,
                        row_key=row_key,
                        backend="process",
                        failure=_failure_payload(
                            exc=exc,
                            row_key=row_key,
                            order_index=order_index,
                            backend="process",
                        ),
                    )
                results_by_index[order_index] = result
                if result.failure is not None and config.fail_fast:
                    for pending in futures:
                        pending.cancel()
                    break
        except concurrent.futures.TimeoutError as exc:
            completed_indexes = set(results_by_index)
            for future, (order_index, row_key) in futures.items():
                if order_index in completed_indexes:
                    continue
                future.cancel()
                results_by_index[order_index] = ParallelRowResult(
                    order_index=order_index,
                    row_key=row_key,
                    backend="process",
                    failure=_failure_payload(
                        exc=exc,
                        row_key=row_key,
                        order_index=order_index,
                        backend="process",
                        extra={
                            "timeout_seconds": config.timeout_seconds,
                            "failure_reason": "process_backend_timeout",
                        },
                    ),
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    return [results_by_index[index] for index in sorted(results_by_index)]


def execute_rows_threaded(
    items: Sequence[Any],
    row_fn: Callable[[Any], dict[str, Any]],
    *,
    config: ParallelExecutionConfig | None = None,
    row_key_fn: Callable[[Any], str] | None = None,
) -> ParallelExecutionSummary:
    execution_config = config or ParallelExecutionConfig(backend="thread")
    return _execute_rows_with_executor(
        concurrent.futures.ThreadPoolExecutor,
        items,
        row_fn,
        config=execution_config,
        row_key_fn=row_key_fn,
        backend="thread",
    )


def execute_rows_process(
    items: Sequence[Any],
    row_fn: Callable[[Any], dict[str, Any]],
    *,
    config: ParallelExecutionConfig | None = None,
    row_key_fn: Callable[[Any], str] | None = None,
) -> ParallelExecutionSummary:
    execution_config = config or ParallelExecutionConfig(backend="process")
    if not execution_config.allow_process_backend:
        raise ValueError("process backend requires allow_process_backend=True")
    indexed_items = [
        (
            order_index,
            row_key_fn(item) if row_key_fn is not None else _default_row_key(item, order_index),
            item,
        )
        for order_index, item in enumerate(items)
    ]
    preflight_failures: list[ParallelRowResult] = []
    picklable_items: list[tuple[int, str, Any]] = []
    try:
        pickle.dumps(row_fn)
    except Exception as exc:  # noqa: BLE001 - expose process configuration failure
        preflight_failures = [
            _process_pickling_failure(exc=exc, row_key=row_key, order_index=order_index)
            for order_index, row_key, _item in indexed_items
        ]
        return ParallelExecutionSummary(
            backend="process",
            workers=execution_config.workers,
            total_submitted=len(items),
            deterministic_ordering=True,
            row_results=preflight_failures,
            warnings=_process_resource_warnings(execution_config),
        )

    for order_index, row_key, item in indexed_items:
        try:
            pickle.dumps(item)
        except Exception as exc:  # noqa: BLE001 - expose process configuration failure
            preflight_failures.append(
                _process_pickling_failure(exc=exc, row_key=row_key, order_index=order_index)
            )
            if execution_config.fail_fast:
                break
            continue
        picklable_items.append((order_index, row_key, item))

    process_results = (
        _execute_process_picklable_rows(
            picklable_items,
            row_fn,
            config=execution_config,
        )
        if picklable_items and not (preflight_failures and execution_config.fail_fast)
        else []
    )
    results = sorted([*preflight_failures, *process_results], key=lambda result: result.order_index)
    return ParallelExecutionSummary(
        backend="process",
        workers=execution_config.workers,
        total_submitted=len(items),
        deterministic_ordering=True,
        row_results=results,
        warnings=_process_resource_warnings(execution_config),
    )


def execute_runner_rows(
    items: Sequence[Any],
    row_fn: Callable[[Any], dict[str, Any]],
    *,
    config: ParallelExecutionConfig | None = None,
    row_key_fn: Callable[[Any], str] | None = None,
) -> ParallelExecutionSummary:
    execution_config = config or ParallelExecutionConfig()
    if execution_config.backend == "serial":
        return execute_rows_serial(items, row_fn, config=execution_config, row_key_fn=row_key_fn)
    if execution_config.backend == "thread":
        return execute_rows_threaded(items, row_fn, config=execution_config, row_key_fn=row_key_fn)
    if execution_config.backend == "process":
        return execute_rows_process(items, row_fn, config=execution_config, row_key_fn=row_key_fn)
    raise ValueError(f"Unsupported row execution backend: {execution_config.backend}")


def _sort_key(row: dict[str, Any]) -> tuple[str, str, int, int, int, int]:
    budget = row.get("requested_budget", row.get("budget", 0))
    return (
        str(row.get("benchmark", row.get("problem", ""))),
        str(row.get("strategy", row.get("algorithm", ""))),
        int(row.get("seed", 0)),
        int(budget or 0),
        int(row.get("dimension", 0) or 0),
        int(row.get("row_order_index", 0) or 0),
    )


def sort_rows_deterministically(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in sorted(rows, key=_sort_key)]


def parallel_summary_to_dict(summary: ParallelExecutionSummary) -> dict[str, Any]:
    failures = [dict(result.failure) for result in summary.failed_results if result.failure is not None]
    return {
        "parallel_enabled": summary.backend != "serial",
        "backend": summary.backend,
        "workers": int(summary.workers),
        "total_submitted": int(summary.total_submitted),
        "success_count": int(summary.success_count),
        "failure_count": int(summary.failure_count),
        "deterministic_merge_order": bool(summary.deterministic_ordering),
        "warnings": list(summary.warnings),
        "failures": failures,
    }


__all__ = [
    "ParallelExecutionConfig",
    "ParallelExecutionSummary",
    "ParallelRowResult",
    "ProcessBackendPicklingError",
    "execute_rows_process",
    "execute_rows_serial",
    "execute_rows_threaded",
    "execute_runner_rows",
    "parallel_summary_to_dict",
    "sort_rows_deterministically",
]
