from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from ga_lab.problems.constrained_box_quadratic import ConstrainedBoxQuadraticProblem
from ga_lab.problems.constrained_dtlz_box_toy import ConstrainedDTLZBoxToyProblem
from ga_lab.problems.constrained_equality_plane_quadratic import (
    ConstrainedEqualityPlaneQuadraticProblem,
)
from ga_lab.problems.constrained_sphere import ConstrainedSphereProblem
from ga_lab.problems.constrained_zdt_box_toy import ConstrainedZDTBoxToyProblem
from ga_lab.problems.dtlz2 import DTLZ2Problem
from ga_lab.problems.dtlz3 import DTLZ3Problem
from ga_lab.problems.dtlz4 import DTLZ4Problem
from ga_lab.problems.knapsack import KnapsackProblem
from ga_lab.problems.onemax import OneMaxProblem
from ga_lab.problems.pymoo_adapter import PymooBackedProblem
from ga_lab.problems.tsp import TSPProblem
from ga_lab.problems.zdt1 import ZDT1Problem
from ga_lab.problems.zdt2 import ZDT2Problem
from ga_lab.problems.zdt3 import ZDT3Problem

ProblemFactory = Callable[[dict[str, object]], object]


@dataclass(frozen=True, slots=True)
class ProblemPluginSpec:
    name: str
    build_fn: ProblemFactory
    aliases: tuple[str, ...] = ()


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


# Label-only metadata that local study manifests attach to ``problem_options`` for traceability
# but which problem constructors do not accept as parameters.
_PROBLEM_OPTION_METADATA_KEYS = frozenset({"instance_name", "instance_source"})


def _as_problem_options(options: Mapping[str, object] | None) -> dict[str, object]:
    if not options:
        return {}
    return {key: value for key, value in options.items() if key not in _PROBLEM_OPTION_METADATA_KEYS}


def _build_tsp(problem_options: dict[str, object]) -> object:
    return TSPProblem(**cast(dict[str, Any], problem_options))


def _build_knapsack(problem_options: dict[str, object]) -> object:
    return KnapsackProblem(**cast(dict[str, Any], problem_options))


def _build_zdt1(problem_options: dict[str, object]) -> object:
    return ZDT1Problem(**cast(dict[str, Any], problem_options))


def _options_builder(problem_cls: Callable[..., object]) -> ProblemFactory:
    def _build(problem_options: dict[str, object]) -> object:
        return problem_cls(**cast(dict[str, Any], problem_options))

    return _build


PROBLEM_SPECS = (
    ProblemPluginSpec(
        name="onemax",
        build_fn=lambda options: OneMaxProblem(**cast(dict[str, Any], options)),
    ),
    ProblemPluginSpec(name="tsp", build_fn=_build_tsp),
    ProblemPluginSpec(name="knapsack", build_fn=_build_knapsack),
    ProblemPluginSpec(name="zdt1", build_fn=_build_zdt1),
)

# Additional multi-objective and constrained benchmark problems are resolvable by name via
# ``build_problem_from_name`` / ``get_problem_plugin`` but are intentionally excluded from the
# documented public ``problem_plugin_names()`` surface.
_EXTENDED_PROBLEM_SPECS = (
    ProblemPluginSpec(name="zdt2", build_fn=_options_builder(ZDT2Problem)),
    ProblemPluginSpec(name="zdt3", build_fn=_options_builder(ZDT3Problem)),
    ProblemPluginSpec(name="dtlz2", build_fn=_options_builder(DTLZ2Problem)),
    ProblemPluginSpec(name="dtlz3", build_fn=_options_builder(DTLZ3Problem)),
    ProblemPluginSpec(name="dtlz4", build_fn=_options_builder(DTLZ4Problem)),
    ProblemPluginSpec(
        name="constrained_sphere", build_fn=_options_builder(ConstrainedSphereProblem)
    ),
    ProblemPluginSpec(
        name="constrained_box_quadratic",
        build_fn=_options_builder(ConstrainedBoxQuadraticProblem),
    ),
    ProblemPluginSpec(
        name="constrained_equality_plane_quadratic",
        build_fn=_options_builder(ConstrainedEqualityPlaneQuadraticProblem),
    ),
    ProblemPluginSpec(
        name="constrained_zdt_box_toy",
        build_fn=_options_builder(ConstrainedZDTBoxToyProblem),
    ),
    ProblemPluginSpec(
        name="constrained_dtlz_box_toy",
        build_fn=_options_builder(ConstrainedDTLZBoxToyProblem),
    ),
)


def _wfg_builder(wfg_name: str) -> ProblemFactory:
    def _build(problem_options: dict[str, object]) -> object:
        return PymooBackedProblem(
            pymoo_name=wfg_name,
            problem_name=wfg_name,
            **cast(dict[str, Any], problem_options),
        )

    return _build


# pymoo-backed WFG benchmark problems are resolvable by name when pymoo is installed (the
# builder raises ImportError otherwise). They are excluded from ``problem_plugin_names()``.
_WFG_PROBLEM_SPECS = tuple(
    ProblemPluginSpec(name=f"wfg{index}", build_fn=_wfg_builder(f"wfg{index}"))
    for index in range(1, 10)
)


def _build_registry() -> dict[str, ProblemPluginSpec]:
    registry: dict[str, ProblemPluginSpec] = {}
    for spec in (*PROBLEM_SPECS, *_EXTENDED_PROBLEM_SPECS, *_WFG_PROBLEM_SPECS):
        for alias in (spec.name, *spec.aliases):
            registry[_normalize_name(alias)] = spec
    return registry


PROBLEM_REGISTRY = _build_registry()


def problem_plugin_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in PROBLEM_SPECS)


def get_problem_plugin(name: str) -> ProblemPluginSpec:
    try:
        return PROBLEM_REGISTRY[_normalize_name(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported problem: {name}") from exc


def build_problem_from_name(
    name: str,
    options: Mapping[str, object] | None = None,
):
    spec = get_problem_plugin(name)
    return spec.build_fn(_as_problem_options(options))
