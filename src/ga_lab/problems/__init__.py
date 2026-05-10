from __future__ import annotations

from ga_lab.problems.registry import (
    build_problem_from_name,
    get_problem_plugin,
    problem_plugin_names,
)


def build_problem(name: str, options=None):
    return build_problem_from_name(name, options)


__all__ = [
    "build_problem",
    "build_problem_from_name",
    "get_problem_plugin",
    "problem_plugin_names",
]
