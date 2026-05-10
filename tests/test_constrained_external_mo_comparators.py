from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ga_lab.problems import build_problem


def test_pymoo_missing_returns_skipped_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from ga_lab.experiment.constrained_external_mo_comparators import (
        run_pymoo_constrained_nsga2,
    )

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object):
        if name == "pymoo":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    problem = build_problem("constrained_zdt_box_toy", {"dimension": 6})

    result = run_pymoo_constrained_nsga2(problem=problem, seed=0, budget=12)

    assert result.status == "skipped"
    assert result.skip_reason == "pymoo is not installed"
    assert result.actual_evaluations == 0


def test_pymoo_imports_are_available_when_installed() -> None:
    from ga_lab.experiment.constrained_external_mo_comparators import is_pymoo_available

    status = is_pymoo_available()
    if not status["installed"]:
        assert status["recommended_status"] == "dependency_missing"
        return

    assert status["version"]
    assert status["api_symbols"]["Problem"]["available"] is True
    assert status["api_symbols"]["NSGA2"]["available"] is True
    assert status["api_symbols"]["minimize"]["available"] is True


def test_wrapper_f_g_shape_and_constraint_signs() -> None:
    from ga_lab.experiment.constrained_external_mo_comparators import (
        PymooConstrainedProblemWrapper,
    )

    problem = build_problem("constrained_zdt_box_toy", {"dimension": 6})
    wrapper = PymooConstrainedProblemWrapper(problem)

    feasible = [0.2, 0.3, 0.0, 0.2, 0.2, 0.2]
    infeasible = [0.8, 0.5, 0.6, 0.6, 0.6, 0.6]
    out: dict[str, object] = {}
    wrapper._evaluate(np.asarray([feasible, infeasible], dtype=float), out)

    assert np.asarray(out["F"]).shape == (2, 2)
    assert np.asarray(out["G"]).shape == (2, 2)
    assert np.all(np.asarray(out["G"])[0] <= 0.0)
    assert np.any(np.asarray(out["G"])[1] > 0.0)
    assert wrapper.evaluations == 2


def test_wrapper_rejects_non_finite_objectives_or_constraints() -> None:
    from ga_lab.experiment.constrained_external_mo_comparators import (
        PymooConstrainedProblemWrapper,
    )

    problem = build_problem("constrained_zdt_box_toy", {"dimension": 6})
    wrapper = PymooConstrainedProblemWrapper(problem)

    with pytest.raises(ValueError, match="Non-finite"):
        wrapper._evaluate(np.asarray([[float("nan")] * 6], dtype=float), {})


def test_pymoo_result_has_actual_evaluations_and_operator_warning() -> None:
    from ga_lab.experiment.constrained_external_mo_comparators import (
        run_pymoo_constrained_nsga2,
    )

    problem = build_problem("constrained_zdt_box_toy", {"dimension": 6})
    result = run_pymoo_constrained_nsga2(problem=problem, seed=1, budget=20, population_size=4)

    assert result.algorithm == "pymoo_constrained_nsga2"
    assert result.status in {"success", "skipped", "failed"}
    assert isinstance(result.actual_evaluations, int)
    if result.status == "success":
        assert result.actual_evaluations == 20
        assert any("operator_family_difference" in warning for warning in result.warnings)


def test_deap_remains_secondary_hold_not_implemented() -> None:
    from ga_lab.experiment.constrained_external_mo_comparators import deap_secondary_status

    status = deap_secondary_status()

    assert status["implemented"] is False
    assert status["primary_recommendation"] is False
    assert status["status"] == "secondary_hold"
