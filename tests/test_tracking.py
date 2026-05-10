from __future__ import annotations

import io
import sys

from ga_lab.config import GAConfig
from ga_lab.experiment.trackers import MLflowTracker
from ga_lab.runner import run_experiment


def _base_config(**overrides: object) -> GAConfig:
    payload = {
        "run_name": "tracked_onemax",
        "problem": "onemax",
        "population_size": 24,
        "genome_length": 16,
        "generations": 20,
        "crossover_rate": 0.9,
        "mutation_rate": 0.02,
        "elitism": 1,
        "tournament_size": 3,
        "seed": 7,
        "maximize": True,
        "target_fitness": 16,
        "log_every": 1,
    }
    payload.update(overrides)
    return GAConfig(**payload)


def test_run_experiment_logs_to_mlflow_tracker(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeMlflow:
        def set_tracking_uri(self, uri: str) -> None:
            calls.append(("set_tracking_uri", uri))

        def set_experiment(self, name: str) -> None:
            calls.append(("set_experiment", name))

        def start_run(self, *, run_name: str, tags: dict[str, str]) -> object:
            calls.append(("start_run", {"run_name": run_name, "tags": tags}))
            return object()

        def log_params(self, params: dict[str, str]) -> None:
            calls.append(("log_params", params))

        def log_metrics(self, metrics: dict[str, float]) -> None:
            calls.append(("log_metrics", metrics))

        def log_artifacts(self, output_dir: str) -> None:
            calls.append(("log_artifacts", output_dir))

        def end_run(self) -> None:
            calls.append(("end_run", None))

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())

    config = _base_config(
        tracking={
            "backend": "mlflow",
            "experiment": "ga-tests",
            "uri": "file:///tmp/mlruns",
            "tags": {"suite": "smoke"},
        }
    )
    result = run_experiment(config, output_root=tmp_path)

    assert result.summary["best_fitness"] == 16.0
    assert any(name == "set_experiment" and payload == "ga-tests" for name, payload in calls)
    assert any(name == "log_params" for name, _ in calls)
    assert any(name == "log_metrics" for name, _ in calls)
    assert any(
        name == "log_artifacts" and str(result.output_dir) == payload for name, payload in calls
    )
    assert calls[-1][0] == "end_run"


def test_run_experiment_uses_mlflow_env_tracking_uri(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeMlflow:
        def set_tracking_uri(self, uri: str) -> None:
            calls.append(("set_tracking_uri", uri))

        def set_experiment(self, name: str) -> None:
            calls.append(("set_experiment", name))

        def start_run(self, *, run_name: str, tags: dict[str, str]) -> object:
            calls.append(("start_run", {"run_name": run_name, "tags": tags}))
            return object()

        def log_params(self, params: dict[str, str]) -> None:
            calls.append(("log_params", params))

        def log_metrics(self, metrics: dict[str, float]) -> None:
            calls.append(("log_metrics", metrics))

        def log_artifacts(self, output_dir: str) -> None:
            calls.append(("log_artifacts", output_dir))

        def end_run(self) -> None:
            calls.append(("end_run", None))

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "env-experiment")

    config = _base_config(
        tracking={
            "backend": "mlflow",
            "tags": {"suite": "env"},
        }
    )

    run_experiment(config, output_root=tmp_path)

    assert ("set_tracking_uri", "http://127.0.0.1:5000") in calls
    assert ("set_experiment", "env-experiment") in calls


def test_run_experiment_logs_to_wandb_tracker(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeConfig:
        def update(
            self,
            payload: dict[str, object],
            allow_val_change: bool = False,
        ) -> None:
            calls.append(
                (
                    "config.update",
                    {
                        "payload": payload,
                        "allow_val_change": allow_val_change,
                    },
                )
            )

    class FakeRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}

        def log_artifact(self, artifact: object) -> None:
            calls.append(("log_artifact", artifact))

    class FakeArtifact:
        def __init__(self, name: str, type: str) -> None:
            self.name = name
            self.type = type
            self.paths: list[str] = []

        def add_dir(self, path: str) -> None:
            self.paths.append(path)
            calls.append(("artifact.add_dir", path))

    fake_run = FakeRun()

    class FakeWandb:
        def __init__(self) -> None:
            self.config = FakeConfig()

        def init(self, **kwargs: object) -> FakeRun:
            calls.append(("init", kwargs))
            return fake_run

        def log(self, metrics: dict[str, float]) -> None:
            calls.append(("log", metrics))

        def Artifact(self, name: str, type: str) -> FakeArtifact:
            calls.append(("Artifact", {"name": name, "type": type}))
            return FakeArtifact(name, type)

        def finish(self) -> None:
            calls.append(("finish", None))

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb())

    config = _base_config(
        tracking={
            "backend": "wandb",
            "project": "ga-tests",
            "mode": "offline",
            "tags": {"suite": "smoke"},
        }
    )
    result = run_experiment(config, output_root=tmp_path)

    assert result.summary["best_fitness"] == 16.0
    assert any(name == "init" for name, _ in calls)
    assert any(name == "config.update" for name, _ in calls)
    assert any(name == "log" for name, _ in calls)
    assert any(
        name == "artifact.add_dir" and payload == str(result.output_dir) for name, payload in calls
    )
    assert fake_run.summary["best_fitness"] == 16.0
    assert calls[-1][0] == "finish"


def test_run_experiment_logs_to_wandb_tracker_online(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeConfig:
        def update(
            self,
            payload: dict[str, object],
            allow_val_change: bool = False,
        ) -> None:
            calls.append(
                (
                    "config.update",
                    {
                        "payload": payload,
                        "allow_val_change": allow_val_change,
                    },
                )
            )

    class FakeRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}

        def log_artifact(self, artifact: object) -> None:
            calls.append(("log_artifact", artifact))

    class FakeArtifact:
        def __init__(self, name: str, type: str) -> None:
            self.name = name
            self.type = type

        def add_dir(self, path: str) -> None:
            calls.append(("artifact.add_dir", path))

    fake_run = FakeRun()

    class FakeApi:
        api_key = None

    class FakeWandb:
        def __init__(self) -> None:
            self.config = FakeConfig()
            self.api = FakeApi()

        def login(
            self,
            *,
            key: str | None = None,
            relogin: bool | None = None,
            host: str | None = None,
            verify: bool = False,
        ) -> bool:
            calls.append(
                (
                    "login",
                    {
                        "key": key,
                        "relogin": relogin,
                        "host": host,
                        "verify": verify,
                    },
                )
            )
            return True

        def init(self, **kwargs: object) -> FakeRun:
            calls.append(("init", kwargs))
            return fake_run

        def log(self, metrics: dict[str, float]) -> None:
            calls.append(("log", metrics))

        def Artifact(self, name: str, type: str) -> FakeArtifact:
            calls.append(("Artifact", {"name": name, "type": type}))
            return FakeArtifact(name, type)

        def finish(self) -> None:
            calls.append(("finish", None))

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb())
    monkeypatch.setenv("WANDB_API_KEY", "test-api-key")

    config = _base_config(
        tracking={
            "backend": "wandb",
            "project": "ga-tests",
            "entity": "openai-lab",
            "base_url": "https://api.wandb.ai",
            "mode": "online",
            "verify_login": True,
            "tags": {"suite": "smoke"},
        }
    )
    result = run_experiment(config, output_root=tmp_path)

    assert result.summary["best_fitness"] == 16.0
    assert any(name == "login" for name, _ in calls)
    assert any(name == "init" and payload.get("entity") == "openai-lab" for name, payload in calls)
    assert any(
        name == "artifact.add_dir" and payload == str(result.output_dir) for name, payload in calls
    )


def test_mlflow_tracker_finish_suppresses_non_utf_console_output(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeStdout(io.StringIO):
        encoding = "cp949"

        def write(self, text: str) -> int:
            text.encode(self.encoding)
            return super().write(text)

    class FakeMlflow:
        def set_experiment(self, name: str) -> None:
            calls.append(("set_experiment", name))

        def start_run(self, *, run_name: str, tags: dict[str, str]) -> object:
            calls.append(("start_run", {"run_name": run_name, "tags": tags}))
            return object()

        def end_run(self) -> None:
            print("🏃 View run")
            calls.append(("end_run", None))

    fake_stdout = FakeStdout()
    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    tracker = MLflowTracker({"experiment": "ga-tests"}, "tracked_onemax")
    tracker.finish()

    assert calls[-1][0] == "end_run"
