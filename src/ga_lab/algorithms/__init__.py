from __future__ import annotations

from ga_lab.algorithms.registry import (
    algorithm_plugin_names,
    build_algorithm_from_name,
    get_algorithm_plugin,
)

__all__ = [
    "algorithm_plugin_names",
    "build_algorithm_from_name",
    "get_algorithm_plugin",
    "run_nsga2",
    "run_single_objective_ga",
]


def __getattr__(name: str):
    if name == "run_nsga2":
        from ga_lab.algorithms.nsga2 import run_nsga2

        return run_nsga2
    if name == "run_single_objective_ga":
        from ga_lab.algorithms.single_objective import run_single_objective_ga

        return run_single_objective_ga
    raise AttributeError(name)
