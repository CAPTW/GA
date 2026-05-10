"""Packaged builtin resource helpers.

This module backs the public consumer path, but its direct import surface is internal-only.
Use ``ga_lab.api`` for stable library access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

RESOURCE_PACKAGE = "ga_lab.builtin_resources"

_PRESET_NAMES = (
    "knapsack_balanced",
    "knapsack_large",
    "knapsack_medium",
    "knapsack_small",
    "onemax_balanced",
    "onemax_fast",
    "onemax_large",
    "onemax_medium",
    "onemax_small",
    "tsp_balanced",
    "tsp_large",
    "tsp_medium",
    "tsp_medium_hybrid",
    "tsp_small",
    "zdt1_balanced",
    "zdt1_large",
    "zdt1_medium",
    "zdt1_small",
)

_DEMO_CONFIG_NAMES = (
    "onemax_pure_ga_demo",
    "tsp_medium_hybrid_demo",
    "zdt1_nsga2_demo",
)

_DEMO_MANIFEST_NAMES = ("baseline_onemax_demo_manifest",)


@dataclass(frozen=True, slots=True)
class BuiltinResourceInfo:
    kind: str
    name: str
    package_relative_path: str
    repo_relative_path: str | None

    @property
    def resource_uri(self) -> str:
        return f"{self.kind}:{self.name}"


def _build_catalog() -> dict[tuple[str, str], BuiltinResourceInfo]:
    catalog: dict[tuple[str, str], BuiltinResourceInfo] = {}
    for name in _PRESET_NAMES:
        catalog[("preset", name)] = BuiltinResourceInfo(
            kind="preset",
            name=name,
            package_relative_path=f"presets/{name}.json",
            repo_relative_path=f"configs/presets/{name}.json",
        )
    for name in _DEMO_CONFIG_NAMES:
        catalog[("demo", name)] = BuiltinResourceInfo(
            kind="demo",
            name=name,
            package_relative_path=f"demo/{name}.json",
            repo_relative_path=f"configs/demo/{name}.json",
        )
    for name in _DEMO_MANIFEST_NAMES:
        catalog[("demo-manifest", name)] = BuiltinResourceInfo(
            kind="demo-manifest",
            name=name,
            package_relative_path=f"demo/{name}.json",
            repo_relative_path=f"configs/demo/{name}.json",
        )
    return catalog


_RESOURCE_CATALOG = _build_catalog()


def _resource_root():
    return files(RESOURCE_PACKAGE)


def split_resource_uri(value: str) -> tuple[str, str]:
    kind, sep, name = value.partition(":")
    if not sep or not kind or not name:
        raise ValueError(
            "Resource references must use the form '<kind>:<name>', "
            "for example 'preset:onemax_small'."
        )
    return kind, name


def get_builtin_resource(kind: str, name: str) -> BuiltinResourceInfo:
    key = (kind, name)
    if key not in _RESOURCE_CATALOG:
        available = ", ".join(sorted(resource.name for resource in iter_builtin_resources(kind)))
        raise ValueError(f"Unknown builtin {kind} resource '{name}'. Available: {available}")
    return _RESOURCE_CATALOG[key]


def iter_builtin_resources(kind: str | None = None) -> tuple[BuiltinResourceInfo, ...]:
    resources = tuple(_RESOURCE_CATALOG.values())
    if kind is None:
        return tuple(sorted(resources, key=lambda item: (item.kind, item.name)))
    return tuple(
        sorted(
            (resource for resource in resources if resource.kind == kind),
            key=lambda item: item.name,
        )
    )


def builtin_resource_names(kind: str) -> tuple[str, ...]:
    return tuple(resource.name for resource in iter_builtin_resources(kind))


def builtin_repo_relative_path(kind: str, name: str) -> str | None:
    return get_builtin_resource(kind, name).repo_relative_path


def read_builtin_json(kind: str, name: str) -> dict[str, Any]:
    resource = get_builtin_resource(kind, name)
    payload = json.loads(
        _resource_root().joinpath(resource.package_relative_path).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Builtin resource {resource.resource_uri} must contain a JSON object")
    return payload


def materialize_builtin_resource(kind: str, name: str, destination: str | Path) -> Path:
    resource = get_builtin_resource(kind, name)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        _resource_root().joinpath(resource.package_relative_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return destination_path
