from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from ga_lab.problems.dtlz2 import DTLZ2Problem
from ga_lab.problems.dtlz3 import DTLZ3Problem
from ga_lab.problems.dtlz4 import DTLZ4Problem
from ga_lab.problems.knapsack import KnapsackProblem
from ga_lab.problems.onemax import OneMaxProblem
from ga_lab.problems.constrained_box_quadratic import ConstrainedBoxQuadraticProblem
from ga_lab.problems.constrained_equality_plane_quadratic import (
    ConstrainedEqualityPlaneQuadraticProblem,
)
from ga_lab.problems.constrained_sphere import ConstrainedSphereProblem
from ga_lab.problems.constrained_dtlz_box_toy import ConstrainedDTLZBoxToyProblem
from ga_lab.problems.constrained_zdt_box_toy import ConstrainedZDTBoxToyProblem
from ga_lab.problems.pymoo_adapter import PymooBackedProblem
from ga_lab.problems.tsp import TSPProblem
from ga_lab.problems.zdt1 import ZDT1Problem

ProblemFactory = Callable[[dict[str, object]], object]


@dataclass(frozen=True, slots=True)
class ProblemPluginSpec:
    name: str
    build_fn: ProblemFactory
    aliases: tuple[str, ...] = ()


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _as_problem_options(options: Mapping[str, object] | None) -> dict[str, object]:
    return dict(options) if options else {}


def _build_tsp(problem_options: dict[str, object]) -> object:
    return TSPProblem(**cast(dict[str, Any], problem_options))


def _build_knapsack(problem_options: dict[str, object]) -> object:
    return KnapsackProblem(**cast(dict[str, Any], problem_options))


def _build_constrained_sphere(problem_options: dict[str, object]) -> object:
    return ConstrainedSphereProblem(**cast(dict[str, Any], problem_options))


def _build_constrained_box_quadratic(problem_options: dict[str, object]) -> object:
    return ConstrainedBoxQuadraticProblem(**cast(dict[str, Any], problem_options))


def _build_constrained_equality_plane_quadratic(problem_options: dict[str, object]) -> object:
    return ConstrainedEqualityPlaneQuadraticProblem(**cast(dict[str, Any], problem_options))


def _build_constrained_zdt_box_toy(problem_options: dict[str, object]) -> object:
    return ConstrainedZDTBoxToyProblem(**cast(dict[str, Any], problem_options))


def _build_constrained_dtlz_box_toy(problem_options: dict[str, object]) -> object:
    return ConstrainedDTLZBoxToyProblem(**cast(dict[str, Any], problem_options))


def _build_zdt1(problem_options: dict[str, object]) -> object:
    return ZDT1Problem(**cast(dict[str, Any], problem_options))


def _build_zdt_family(family: str) -> ProblemFactory:
    def _build(problem_options: dict[str, object]) -> object:
        resolved = dict(problem_options)
        resolved.setdefault("family", family)
        return ZDT1Problem(**cast(dict[str, Any], resolved))

    return _build


def _build_dtlz2(problem_options: dict[str, object]) -> object:
    return DTLZ2Problem(**cast(dict[str, Any], problem_options))


def _build_dtlz3(problem_options: dict[str, object]) -> object:
    return DTLZ3Problem(**cast(dict[str, Any], problem_options))


def _build_dtlz4(problem_options: dict[str, object]) -> object:
    return DTLZ4Problem(**cast(dict[str, Any], problem_options))


def _build_pymoo_family(family: str) -> ProblemFactory:
    def _build(problem_options: dict[str, object]) -> object:
        resolved = dict(problem_options)
        objective_count = int(resolved.pop("objective_count", 2))
        variable_count = int(resolved.pop("variable_count", 6))
        if resolved:
            raise ValueError(f"Unsupported problem options for {family}: {sorted(resolved)}")
        return PymooBackedProblem(
            pymoo_name=family,
            problem_name=family,
            objective_count=objective_count,
            variable_count=variable_count,
        )

    return _build


PROBLEM_SPECS = (
    ProblemPluginSpec(
        name="onemax",
        build_fn=lambda options: OneMaxProblem(**cast(dict[str, Any], options)),
    ),
    ProblemPluginSpec(name="tsp", build_fn=_build_tsp),
    ProblemPluginSpec(name="knapsack", build_fn=_build_knapsack),
    ProblemPluginSpec(name="constrained_sphere", build_fn=_build_constrained_sphere),
    ProblemPluginSpec(
        name="constrained_box_quadratic",
        build_fn=_build_constrained_box_quadratic,
        aliases=("constrained-box-quadratic",),
    ),
    ProblemPluginSpec(
        name="constrained_equality_plane_quadratic",
        build_fn=_build_constrained_equality_plane_quadratic,
        aliases=("constrained-equality-plane-quadratic",),
    ),
    ProblemPluginSpec(
        name="constrained_zdt_box_toy",
        build_fn=_build_constrained_zdt_box_toy,
        aliases=("constrained-zdt-box-toy",),
    ),
    ProblemPluginSpec(
        name="constrained_dtlz_box_toy",
        build_fn=_build_constrained_dtlz_box_toy,
        aliases=("constrained-dtlz-box-toy",),
    ),
    ProblemPluginSpec(name="zdt1", build_fn=_build_zdt1),
    ProblemPluginSpec(name="zdt2", build_fn=_build_zdt_family("zdt2")),
    ProblemPluginSpec(name="zdt3", build_fn=_build_zdt_family("zdt3")),
    ProblemPluginSpec(name="dtlz2", build_fn=_build_dtlz2),
    ProblemPluginSpec(name="dtlz3", build_fn=_build_dtlz3),
    ProblemPluginSpec(name="dtlz4", build_fn=_build_dtlz4),
    ProblemPluginSpec(name="wfg1", build_fn=_build_pymoo_family("wfg1")),
    ProblemPluginSpec(name="wfg2", build_fn=_build_pymoo_family("wfg2")),
)


def _build_registry() -> dict[str, ProblemPluginSpec]:
    registry: dict[str, ProblemPluginSpec] = {}
    for spec in PROBLEM_SPECS:
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
