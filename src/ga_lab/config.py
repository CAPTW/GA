from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from ga_lab.algorithms.registry import algorithm_plugin_names, get_algorithm_plugin
from ga_lab.core.crossover import crossover_plugin_names, get_crossover_plugin
from ga_lab.core.mutation import get_mutation_plugin, mutation_plugin_names
from ga_lab.core.representation import (
    get_representation_plugin,
    representation_plugin_names,
)
from ga_lab.core.selection import get_selection_plugin, selection_plugin_names
from ga_lab.problems.registry import get_problem_plugin, problem_plugin_names


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


ALGORITHM_NAMES = algorithm_plugin_names()
PROBLEM_NAMES = problem_plugin_names()
REPRESENTATION_NAMES = representation_plugin_names()
SELECTION_NAMES = selection_plugin_names()
CROSSOVER_NAMES = crossover_plugin_names()
MUTATION_NAMES = mutation_plugin_names()


def _mapping_to_dict(value: object, *, label: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


_TOP_LEVEL_ALIASES: dict[str, tuple[str, ...]] = {
    "population_size": ("pop_size", "npop", "n_population"),
    "generations": ("generation_count", "max_iter", "n_generations"),
    "crossover_rate": ("pc",),
    "mutation_rate": ("pm",),
    "elitism": ("elite_size",),
    "seed": ("random_seed",),
}


def _normalize_scalar_aliases(data: dict[str, object]) -> None:
    for canonical, aliases in _TOP_LEVEL_ALIASES.items():
        values: list[tuple[str, object]] = []
        for alias in aliases:
            if alias in data:
                values.append((alias, data.pop(alias)))
        if not values:
            continue
        if canonical in data:
            canonical_value = data[canonical]
            for alias, alias_value in values:
                if canonical_value != alias_value:
                    raise ValueError(
                        f"Conflicting values for '{canonical}' and alias '{alias}': "
                        f"{canonical_value!r} != {alias_value!r}"
                    )
            continue
        if len(values) == 1:
            data[canonical] = values[0][1]
            continue
        distinct = {repr(alias_value) for _, alias_value in values}
        if len(distinct) != 1:
            aliases_label = ", ".join(alias for alias, _ in values)
            raise ValueError(f"Conflicting aliases for '{canonical}': {aliases_label}")
        data[canonical] = values[0][1]


def _normalize_representation_bounds(options: object) -> dict[str, object]:
    if not isinstance(options, Mapping):
        if options is None:
            return {}
        raise ValueError(
            "representation_options must be a JSON object when provided"
        )
    normalized = dict(options)
    alias_pairs = {"lower_bound": "low", "upper_bound": "high"}
    for alias, canonical in alias_pairs.items():
        if alias in normalized:
            if canonical in normalized and normalized[canonical] != normalized[alias]:
                raise ValueError(f"Conflicting values for '{canonical}' and '{alias}'")
            normalized[canonical] = normalized.pop(alias)
    if "bounds" in normalized:
        if "low" in normalized or "high" in normalized:
            raise ValueError(
                "representation_options must not define 'bounds' together with 'low'/'high'"
            )
        bounds = normalized["bounds"]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("representation_options.bounds must be a numeric pair [low, high]")
        low, high = bounds
        if not isinstance(low, int | float) or not isinstance(high, int | float):
            raise ValueError("representation_options.bounds must be a numeric pair [low, high]")
        normalized["low"] = float(low)
        normalized["high"] = float(high)
        normalized.pop("bounds", None)
    return normalized


@dataclass(slots=True)
class PluginSpec:
    name: str
    options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("PluginSpec.name must be a non-empty string")
        self.name = self.name.strip()
        self.options = dict(self.options)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "options": dict(self.options),
        }


@dataclass(slots=True)
class OperatorsSpec:
    selection: PluginSpec
    crossover: PluginSpec
    mutation: PluginSpec

    def to_dict(self) -> dict[str, object]:
        return {
            "selection": self.selection.to_dict(),
            "crossover": self.crossover.to_dict(),
            "mutation": self.mutation.to_dict(),
        }


def _plugin_spec_from_payload(
    value: object,
    *,
    label: str,
    options: object = None,
    default_name: str | None = None,
) -> PluginSpec:
    merged_options = _mapping_to_dict(options, label=f"{label}.options")

    if isinstance(value, str):
        name = value
    elif isinstance(value, Mapping):
        name_value = value.get("name", default_name)
        if not isinstance(name_value, str) or not name_value.strip():
            raise ValueError(f"{label}.name must be a non-empty string")
        name = name_value
        merged_options.update(_mapping_to_dict(value.get("options"), label=f"{label}.options"))
    elif value is None and default_name is not None:
        name = default_name
    elif value is None:
        raise ValueError(f"{label} must be provided")
    else:
        raise ValueError(f"{label} must be either a string or an object")

    return PluginSpec(name=name, options=merged_options)


def _operator_value(data: Mapping[str, object], key: str) -> object:
    operators = data.get("operators")
    if operators is None:
        return data.get(key)
    if not isinstance(operators, Mapping):
        raise ValueError("operators must be a JSON object")
    return operators.get(key, data.get(key))


def normalize_config_data(data: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(data)
    _normalize_scalar_aliases(normalized)
    normalized["representation_options"] = _normalize_representation_bounds(
        normalized.get("representation_options")
    )

    problem_spec = _plugin_spec_from_payload(
        normalized.get("problem"),
        label="problem",
        options=normalized.get("problem_options"),
    )
    algorithm_spec = _plugin_spec_from_payload(
        normalized.get("algorithm"),
        label="algorithm",
        options=normalized.get("algorithm_options"),
        default_name="ga",
    )
    representation_spec = _plugin_spec_from_payload(
        normalized.get("representation"),
        label="representation",
        options=normalized.get("representation_options"),
        default_name="bit",
    )
    selection_spec = _plugin_spec_from_payload(
        _operator_value(normalized, "selection"),
        label="operators.selection",
        options=normalized.get("selection_options"),
        default_name="tournament",
    )
    crossover_spec = _plugin_spec_from_payload(
        _operator_value(normalized, "crossover"),
        label="operators.crossover",
        options=normalized.get("crossover_options"),
        default_name="one_point",
    )
    mutation_spec = _plugin_spec_from_payload(
        _operator_value(normalized, "mutation"),
        label="operators.mutation",
        options=normalized.get("mutation_options"),
        default_name="bit_flip",
    )

    normalized["problem"] = problem_spec.name
    normalized["problem_options"] = problem_spec.options
    normalized["algorithm"] = algorithm_spec.name
    normalized["algorithm_options"] = algorithm_spec.options
    normalized["representation"] = representation_spec.name
    normalized["representation_options"] = representation_spec.options
    normalized["selection"] = selection_spec.name
    normalized["selection_options"] = selection_spec.options
    normalized["crossover"] = crossover_spec.name
    normalized["crossover_options"] = crossover_spec.options
    normalized["mutation"] = mutation_spec.name
    normalized["mutation_options"] = mutation_spec.options
    normalized.pop("operators", None)
    return normalized


@dataclass(slots=True)
class GAConfig:
    run_name: str
    problem: str
    population_size: int
    genome_length: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    elitism: int = 1
    tournament_size: int = 3
    algorithm: str = "ga"
    selection: str = "tournament"
    crossover: str = "one_point"
    mutation: str = "bit_flip"
    representation: str = "bit"
    algorithm_options: dict[str, object] = field(default_factory=dict)
    selection_options: dict[str, object] = field(default_factory=dict)
    crossover_options: dict[str, object] = field(default_factory=dict)
    mutation_options: dict[str, object] = field(default_factory=dict)
    representation_options: dict[str, object] = field(default_factory=dict)
    problem_options: dict[str, object] = field(default_factory=dict)
    tracking: dict[str, object] = field(default_factory=dict)
    objective_directions: list[bool] = field(default_factory=list)
    seed: int = 0
    maximize: bool = True
    target_fitness: float | None = None
    log_every: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> GAConfig:
        config = cls(**cast(dict[str, Any], normalize_config_data(data)))
        config.validate()
        return config

    @property
    def algorithm_spec(self) -> PluginSpec:
        return PluginSpec(self.algorithm, self.algorithm_options)

    @property
    def representation_spec(self) -> PluginSpec:
        return PluginSpec(self.representation, self.representation_options)

    @property
    def problem_spec(self) -> PluginSpec:
        return PluginSpec(self.problem, self.problem_options)

    @property
    def operators_spec(self) -> OperatorsSpec:
        selection_options = dict(self.selection_options)
        if (
            _normalize_name(self.selection) == "tournament"
            and "tournament_size" not in selection_options
        ):
            selection_options["tournament_size"] = self.tournament_size
        return OperatorsSpec(
            selection=PluginSpec(self.selection, selection_options),
            crossover=PluginSpec(self.crossover, self.crossover_options),
            mutation=PluginSpec(self.mutation, self.mutation_options),
        )

    def validate(self) -> None:
        for field_name in (
            "algorithm_options",
            "selection_options",
            "crossover_options",
            "mutation_options",
            "representation_options",
            "problem_options",
            "tracking",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, dict):
                raise ValueError(f"{field_name} must be a JSON object")
        if self.population_size <= 1:
            raise ValueError("population_size must be greater than 1")
        if self.genome_length <= 0:
            raise ValueError("genome_length must be positive")
        if self.generations <= 0:
            raise ValueError("generations must be positive")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.elitism < 0 or self.elitism >= self.population_size:
            raise ValueError("elitism must be in [0, population_size)")
        try:
            get_problem_plugin(self.problem)
        except ValueError as exc:
            raise ValueError("problem must be one of: " + ", ".join(PROBLEM_NAMES)) from exc
        try:
            get_algorithm_plugin(self.algorithm)
        except ValueError as exc:
            raise ValueError("algorithm must be one of: " + ", ".join(ALGORITHM_NAMES)) from exc
        try:
            get_representation_plugin(self.representation)
        except ValueError as exc:
            raise ValueError(
                "representation must be one of: " + ", ".join(REPRESENTATION_NAMES)
            ) from exc
        try:
            get_selection_plugin(self.selection)
        except ValueError as exc:
            raise ValueError("selection must be one of: " + ", ".join(SELECTION_NAMES)) from exc
        try:
            get_crossover_plugin(self.crossover)
        except ValueError as exc:
            raise ValueError("crossover must be one of: " + ", ".join(CROSSOVER_NAMES)) from exc
        try:
            get_mutation_plugin(self.mutation)
        except ValueError as exc:
            raise ValueError("mutation must be one of: " + ", ".join(MUTATION_NAMES)) from exc
        effective_tournament_size = self.selection_options.get(
            "tournament_size", self.tournament_size
        )
        if not isinstance(effective_tournament_size, int):
            raise ValueError("tournament_size must be an integer")
        if effective_tournament_size < 2 or effective_tournament_size > self.population_size:
            raise ValueError("tournament_size must be in [2, population_size]")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        for direction in self.objective_directions:
            if not isinstance(direction, bool):
                raise ValueError("objective_directions must be a list of booleans")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        hypervolume_reference_point = self.algorithm_options.get("hypervolume_reference_point")
        if hypervolume_reference_point is not None:
            if not isinstance(hypervolume_reference_point, list) or not hypervolume_reference_point:
                raise ValueError("hypervolume_reference_point must be a non-empty numeric list")
            for value in hypervolume_reference_point:
                if not isinstance(value, int | float):
                    raise ValueError("hypervolume_reference_point must contain only numbers")
        if self.tracking:
            backend = self.tracking.get("backend")
            if not isinstance(backend, str) or not backend.strip():
                raise ValueError(
                    "tracking.backend must be a non-empty string when tracking is enabled"
                )
            tags = self.tracking.get("tags")
            if tags is not None and not isinstance(tags, dict):
                raise ValueError("tracking.tags must be a JSON object")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "run_name": self.run_name,
            "problem": self.problem_spec.to_dict(),
            "population_size": self.population_size,
            "genome_length": self.genome_length,
            "generations": self.generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elitism": self.elitism,
            "tournament_size": self.tournament_size,
            "algorithm": self.algorithm_spec.to_dict(),
            "representation": self.representation_spec.to_dict(),
            "operators": self.operators_spec.to_dict(),
            "tracking": dict(self.tracking),
            "objective_directions": list(self.objective_directions),
            "seed": self.seed,
            "maximize": self.maximize,
            "target_fitness": self.target_fitness,
            "log_every": self.log_every,
        }


def load_config(path: str | Path) -> GAConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GAConfig.from_dict(data)
