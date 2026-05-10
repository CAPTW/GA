from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ga_lab import __version__
from ga_lab.config import GAConfig, load_config
from ga_lab.demo import DEMO_SPECS, run_demo_payload
from ga_lab.public_types import DemoInfo, PresetInfo, RecommendationResult, RunResultSummary
from ga_lab.recommendations import recommend_preset as _recommend_preset
from ga_lab.recommendations import recommend_solver as _recommend_solver
from ga_lab.resources import builtin_resource_names, get_builtin_resource, read_builtin_json
from ga_lab.runner import run_experiment

PUBLIC_API_SCHEMA_VERSION = 1

__all__ = [
    "PUBLIC_API_SCHEMA_VERSION",
    "DemoInfo",
    "PresetInfo",
    "RecommendationResult",
    "RunResultSummary",
    "get_version",
    "list_presets",
    "list_demos",
    "load_builtin_preset",
    "load_builtin_demo",
    "recommend_preset",
    "recommend_solver",
    "run_preset",
    "run_config",
    "run_demo",
]


def get_version() -> str:
    """Return the installed package version for the stable public API surface."""

    return __version__


def _ensure_stable_family_arg(family: str | None) -> None:
    if family is None:
        return
    raise ValueError(
        "Family-conditioned benchmark semantics stay in the documented evidence layer for now. "
        "The stable portable API currently supports problem/size/priority only."
    )


def _format_recommendation(payload: dict[str, Any], *, kind: str) -> RecommendationResult:
    return RecommendationResult(
        kind=kind,
        problem=str(payload["problem"]),
        size=int(payload["size"]),
        tier=str(payload["tier"]),
        priority=str(payload["priority"]),
        validated_sizes=tuple(int(value) for value in payload["validated_sizes"]),
        solver_family=payload.get("solver_family"),
        solver_name=payload.get("solver_name"),
        preset_name=payload.get("preset_name"),
        config_name=payload.get("config_name"),
        resource_uri=payload.get("resource_uri"),
        portable_command_hint=payload.get("portable_command_hint"),
        note=str(payload["note"]),
        raw=dict(payload),
    )


def _format_output(
    result: RecommendationResult,
    output_format: str,
) -> RecommendationResult | dict[str, Any] | str:
    if output_format == "python":
        return result
    if output_format == "dict":
        return result.to_dict()
    if output_format == "json":
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    raise ValueError("Unsupported format. Allowed: python, dict, json")


def _preset_tier(name: str) -> str:
    suffix = name.rsplit("_", 1)[-1]
    if suffix in {"small", "medium", "large", "fast", "balanced", "hybrid"}:
        return suffix
    return "custom"


def list_presets() -> tuple[PresetInfo, ...]:
    """List builtin preset resources that are supported in installed consumer mode."""

    presets: list[PresetInfo] = []
    for name in builtin_resource_names("preset"):
        payload = read_builtin_json("preset", name)
        resource = get_builtin_resource("preset", name)
        presets.append(
            PresetInfo(
                name=name,
                problem=str(payload.get("problem", "unknown")),
                tier=_preset_tier(name),
                resource_uri=resource.resource_uri,
                portable_command_hint=f"ga-lab-run --preset {name}",
                repo_relative_path=resource.repo_relative_path,
            )
        )
    return tuple(sorted(presets, key=lambda item: item.name))


def list_demos() -> tuple[DemoInfo, ...]:
    """List packaged demo entry points supported by the public consumer path."""

    demos = [
        DemoInfo(
            name=name,
            mode=spec["mode"],
            solver_family=spec["solver_family"],
            title=spec["title"],
            focus=spec["focus"],
            resource_uri=get_builtin_resource(
                spec["resource_kind"], spec["resource_name"]
            ).resource_uri,
        )
        for name, spec in DEMO_SPECS.items()
    ]
    return tuple(sorted(demos, key=lambda item: item.name))


def load_builtin_preset(name: str) -> dict[str, Any]:
    """Load a packaged preset JSON object by builtin preset name."""

    return read_builtin_json("preset", name)


def load_builtin_demo(name: str) -> dict[str, Any]:
    """Load a packaged demo config or manifest JSON object by public demo name."""

    if name not in DEMO_SPECS:
        available = ", ".join(sorted(DEMO_SPECS))
        raise ValueError(f"Unknown demo '{name}'. Available: {available}")
    spec = DEMO_SPECS[name]
    return read_builtin_json(spec["resource_kind"], spec["resource_name"])


def recommend_preset(
    problem: str,
    size: int,
    priority: str,
    family: str | None = None,
    format: str = "python",
) -> RecommendationResult | dict[str, Any] | str:
    """
    Recommend a validated builtin preset for the portable consumer path.

    The stable API currently freezes recommendations at the problem/size/priority layer.
    Family-conditioned benchmark semantics remain documented, but not promoted to a stable
    programmable contract yet.
    """

    _ensure_stable_family_arg(family)
    payload = _recommend_preset(problem, size, priority)
    return _format_output(_format_recommendation(payload, kind="preset"), format)


def recommend_solver(
    problem: str,
    size: int,
    priority: str,
    family: str | None = None,
    format: str = "python",
) -> RecommendationResult | dict[str, Any] | str:
    """
    Recommend a solver family and, when available, a portable preset/config reference.

    The stable API keeps the scope intentionally narrow: validated problem/size/priority only.
    """

    _ensure_stable_family_arg(family)
    payload = _recommend_solver(problem, size, priority)
    return _format_output(_format_recommendation(payload, kind="solver"), format)


def _coerce_config(config: GAConfig | Mapping[str, Any] | str | Path) -> GAConfig:
    if isinstance(config, GAConfig):
        return config
    if isinstance(config, Mapping):
        return GAConfig.from_dict(dict(config))
    return load_config(config)


def _stable_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "best_fitness",
        "best_route_distance",
        "hypervolume",
        "pareto_ratio",
        "spread",
        "pareto_front_size",
        "stop_reason",
        "runtime_seconds",
        "configured_budget",
        "baseline_success_rate",
        "ga_success_rate",
        "hybrid_extra_evaluations",
    )
    return {key: summary[key] for key in keys if key in summary}


def _run_result_from_summary(
    *,
    kind: str,
    label: str,
    summary: dict[str, Any],
    output_dir: Path | None,
    summary_path: Path | None,
) -> RunResultSummary:
    return RunResultSummary(
        kind=kind,
        label=label,
        problem=summary.get("problem"),
        algorithm=summary.get("algorithm"),
        output_dir=str(output_dir.resolve()) if output_dir is not None else None,
        summary_path=str(summary_path.resolve()) if summary_path is not None else None,
        metrics=_stable_metrics(summary),
        raw_summary=dict(summary),
    )


def run_config(
    config: GAConfig | Mapping[str, Any] | str | Path,
    output_root: str | Path,
) -> RunResultSummary:
    """Run a config object or config path and return a stable summary subset."""

    runtime_config = _coerce_config(config)
    result = run_experiment(runtime_config, output_root=output_root)
    return _run_result_from_summary(
        kind="experiment",
        label=runtime_config.run_name,
        summary=result.summary,
        output_dir=result.output_dir,
        summary_path=result.output_dir / "summary.json",
    )


def run_preset(
    name: str,
    output_root: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> RunResultSummary:
    """
    Run a builtin preset by name.

    `overrides` is a shallow mapping over the packaged preset JSON object.
    """

    payload = load_builtin_preset(name)
    if overrides:
        payload.update(dict(overrides))
    return run_config(payload, output_root=output_root)


def run_demo(name: str, output_root: str | Path) -> RunResultSummary:
    """Run a packaged demo by public demo name."""

    payload = run_demo_payload(name, Path(output_root))
    output_dir = (
        Path(payload["summary_path"]).resolve().parent
        if payload.get("summary_path")
        else None
    )
    summary_path = Path(payload["summary_path"]).resolve() if payload.get("summary_path") else None
    return _run_result_from_summary(
        kind="demo",
        label=name,
        summary=payload,
        output_dir=output_dir,
        summary_path=summary_path,
    )
