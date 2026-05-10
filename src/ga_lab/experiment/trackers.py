from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _flatten_payload(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_payload(value, prefix=full_key))
        else:
            flattened[full_key] = _stringify(value)
    return flattened


def _numeric_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            metrics[str(key)] = float(value)
    return metrics


def _tracking_value(
    tracking: Mapping[str, Any],
    key: str,
    *,
    env: str | None = None,
    default: Any = None,
) -> Any:
    value = tracking.get(key)
    if value is not None:
        return value
    if env:
        env_value = os.getenv(env)
        if env_value:
            return env_value
    return default


class Tracker(Protocol):
    def log_config(self, config: Mapping[str, Any]) -> None: ...

    def log_summary(self, summary: Mapping[str, Any]) -> None: ...

    def log_artifacts(self, output_dir: Path) -> None: ...

    def finish(self) -> None: ...


class NullTracker:
    def log_config(self, config: Mapping[str, Any]) -> None:
        return None

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        return None

    def log_artifacts(self, output_dir: Path) -> None:
        return None

    def finish(self) -> None:
        return None


class MLflowTracker:
    def __init__(self, tracking: Mapping[str, Any], run_name: str) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "tracking.backend=mlflow requires the optional 'mlflow' package"
            ) from exc

        self._mlflow = mlflow
        uri = _tracking_value(tracking, "uri", env="MLFLOW_TRACKING_URI")
        if isinstance(uri, str) and uri.strip():
            mlflow.set_tracking_uri(uri.strip())

        experiment_name = (
            _tracking_value(tracking, "experiment", env="MLFLOW_EXPERIMENT_NAME")
            or tracking.get("project")
            or "ga-codex-lab"
        )
        mlflow.set_experiment(str(experiment_name))
        tags = tracking.get("tags", {})
        if tags is not None and not isinstance(tags, Mapping):
            raise ValueError("tracking.tags must be a JSON object")
        self._run = mlflow.start_run(
            run_name=str(tracking.get("run_name") or run_name),
            tags={str(key): str(value) for key, value in dict(tags or {}).items()},
        )

    def log_config(self, config: Mapping[str, Any]) -> None:
        self._mlflow.log_params(_flatten_payload(config))

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        self._mlflow.log_metrics(_numeric_metrics(summary))

    def log_artifacts(self, output_dir: Path) -> None:
        self._mlflow.log_artifacts(str(output_dir))

    def finish(self) -> None:
        stdout = getattr(sys, "stdout", None)
        encoding = str(getattr(stdout, "encoding", "") or "").lower()
        if encoding.startswith("utf"):
            self._mlflow.end_run()
            return

        # MLflow 3.x prints run URLs with emoji on end_run().
        # Suppress that output on non-UTF consoles so tracking still completes.
        with contextlib.redirect_stdout(io.StringIO()):
            self._mlflow.end_run()


class WandbTracker:
    def __init__(self, tracking: Mapping[str, Any], run_name: str) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "tracking.backend=wandb requires the optional 'wandb' package"
            ) from exc

        tags = tracking.get("tags", {})
        if tags is not None and not isinstance(tags, Mapping):
            raise ValueError("tracking.tags must be a JSON object")
        tag_values = [f"{key}:{value}" for key, value in dict(tags or {}).items()]
        self._wandb = wandb
        mode_value = str(_tracking_value(tracking, "mode", env="WANDB_MODE", default="online"))
        if mode_value not in {"online", "offline", "disabled", "shared"}:
            raise ValueError("tracking.mode must be one of: online, offline, disabled, shared")
        mode = cast(Literal["online", "offline", "disabled", "shared"], mode_value)
        base_url = _tracking_value(tracking, "base_url", env="WANDB_BASE_URL")
        if isinstance(base_url, str) and base_url.strip():
            os.environ["WANDB_BASE_URL"] = base_url.strip()

        api_key = _tracking_value(tracking, "api_key", env="WANDB_API_KEY")
        existing_key = getattr(getattr(wandb, "api", None), "api_key", None)
        if mode != "offline":
            resolved_key = api_key or existing_key
            if not isinstance(resolved_key, str) or not resolved_key.strip():
                raise RuntimeError(
                    "tracking.backend=wandb in online mode requires tracking.api_key, "
                    "WANDB_API_KEY, or a prior 'wandb login'"
                )
            wandb.login(
                key=resolved_key.strip(),
                host=base_url.strip() if isinstance(base_url, str) and base_url.strip() else None,
                relogin=bool(tracking.get("relogin", False)),
                verify=bool(tracking.get("verify_login", True)),
            )

        entity = _tracking_value(tracking, "entity", env="WANDB_ENTITY")
        project = _tracking_value(
            tracking,
            "project",
            env="WANDB_PROJECT",
            default="ga-codex-lab",
        )
        run_dir = _tracking_value(tracking, "dir", env="WANDB_DIR")
        self._run = wandb.init(
            project=str(project),
            entity=str(entity) if isinstance(entity, str) and entity.strip() else None,
            name=str(tracking.get("run_name") or run_name),
            tags=tag_values,
            dir=str(run_dir) if isinstance(run_dir, str) and run_dir.strip() else None,
            mode=mode,
        )

    def log_config(self, config: Mapping[str, Any]) -> None:
        self._wandb.config.update(dict(config), allow_val_change=True)

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        metrics = _numeric_metrics(summary)
        if metrics:
            self._wandb.log(metrics)
        if self._run is not None:
            for key, value in summary.items():
                if isinstance(value, bool | int | float | str):
                    self._run.summary[str(key)] = value

    def log_artifacts(self, output_dir: Path) -> None:
        if self._run is None:
            return
        artifact = self._wandb.Artifact(output_dir.name, type="ga-results")
        artifact.add_dir(str(output_dir))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        self._wandb.finish()


def build_tracker(tracking: Mapping[str, Any] | None, *, run_name: str) -> Tracker:
    if tracking is None or not tracking:
        return NullTracker()
    backend = tracking.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("tracking.backend must be provided when tracking is enabled")
    normalized = backend.strip().lower()
    if normalized == "mlflow":
        return MLflowTracker(tracking, run_name)
    if normalized == "wandb":
        return WandbTracker(tracking, run_name)
    raise ValueError("tracking.backend must be one of: mlflow, wandb")
