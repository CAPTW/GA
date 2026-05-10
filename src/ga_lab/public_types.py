from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PresetInfo:
    name: str
    problem: str
    tier: str
    resource_uri: str
    portable_command_hint: str
    repo_relative_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DemoInfo:
    name: str
    mode: str
    solver_family: str
    title: str
    focus: str
    resource_uri: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    kind: str
    problem: str
    size: int
    tier: str
    priority: str
    validated_sizes: tuple[int, ...]
    solver_family: str | None
    solver_name: str | None
    preset_name: str | None
    config_name: str | None
    resource_uri: str | None
    portable_command_hint: str | None
    note: str
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.raw)
        payload.update(
            {
                "kind": self.kind,
                "problem": self.problem,
                "size": self.size,
                "tier": self.tier,
                "priority": self.priority,
                "validated_sizes": list(self.validated_sizes),
                "solver_family": self.solver_family,
                "solver_name": self.solver_name,
                "preset_name": self.preset_name,
                "config_name": self.config_name,
                "resource_uri": self.resource_uri,
                "portable_command_hint": self.portable_command_hint,
                "note": self.note,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class RunResultSummary:
    kind: str
    label: str
    problem: str | None
    algorithm: str | None
    output_dir: str | None
    summary_path: str | None
    metrics: dict[str, Any]
    raw_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
