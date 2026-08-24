from __future__ import annotations

import math
import importlib.util

import pytest

from ga_lab.experiment.mo_metrics import reference_front_for_problem
from ga_lab.problems.registry import (
    build_problem_from_name,
    get_problem_plugin,
    problem_plugin_names,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pymoo") is None,
    reason="pymoo is not installed",
)


def test_wfg_adapters_return_finite_two_objective_vectors() -> None:
    for problem_name in ("wfg1", "wfg2"):
        problem = build_problem_from_name(
            problem_name,
            {"objective_count": 2, "variable_count": 6},
        )
        values = list(problem.fitness([0.5] * 6))

        assert problem.name == problem_name
        assert len(values) == 2
        assert all(math.isfinite(value) for value in values)
        assert problem.metadata().exact_genome_length == 6
        assert problem.metadata().default_objective_directions == (False, False)
        assert len(problem.source_bounds()) == 6


def test_wfg_reference_fronts_exist_from_pymoo_backend() -> None:
    for problem_name in ("wfg1", "wfg2"):
        front = reference_front_for_problem(
            problem_name,
            point_count=51,
            objective_count=2,
            variable_count=6,
        )
        assert len(front) >= 20
        assert all(len(point) == 2 for point in front)
        assert all(math.isfinite(value) for point in front for value in point)


def test_wfg_problems_are_registered() -> None:
    # WFG problems live in the extended pymoo-backed registry: resolvable + buildable by name,
    # but intentionally EXCLUDED from the core public plugin-name list (which stays exactly
    # onemax/tsp/knapsack/zdt1 — see test_registries). Assert the extended-registry contract
    # rather than core-list membership.
    core_names = problem_plugin_names()
    for wfg_name in ("wfg1", "wfg2"):
        assert wfg_name not in core_names
        spec = get_problem_plugin(wfg_name)
        assert spec.name == wfg_name
        problem = build_problem_from_name(
            wfg_name,
            {"objective_count": 2, "variable_count": 6},
        )
        assert problem is not None
        assert problem.name == wfg_name
