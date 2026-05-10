from __future__ import annotations

import pytest

from ga_lab.algorithms._shared import resolve_objective_directions
from ga_lab.problems.zdt1 import ZDT1Problem


class _DirectionConfig:
    def __init__(self, maximize: bool, objective_directions: list[bool] | None = None) -> None:
        self.maximize = maximize
        self.objective_directions = objective_directions or []


def test_resolve_objective_directions_defaults_to_config_maximize_for_single_objective() -> None:
    config = _DirectionConfig(maximize=False)
    assert resolve_objective_directions(1, config) == [False]


def test_resolve_objective_directions_prefers_problem_metadata_for_multi_objective() -> None:
    config = _DirectionConfig(maximize=True)
    problem = ZDT1Problem()
    assert resolve_objective_directions(2, config, problem) == [False, False]


def test_resolve_objective_directions_uses_explicit_config_vector() -> None:
    config = _DirectionConfig(maximize=True, objective_directions=[False, True, False])
    assert resolve_objective_directions(3, config) == [False, True, False]


def test_resolve_objective_directions_rejects_length_mismatch() -> None:
    config = _DirectionConfig(maximize=True, objective_directions=[True, False])
    with pytest.raises(ValueError, match="objective_directions length must match"):
        resolve_objective_directions(3, config)
