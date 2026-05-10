from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from ga_lab.algorithms._shared import validate_fitness_vector
from ga_lab.constraints import ConstraintEvaluation


@dataclass(slots=True)
class PymooConstrainedComparatorResult:
    algorithm: str
    status: str
    seed: int
    requested_budget: int
    actual_evaluations: int
    runtime_seconds: float
    objective_vectors: list[list[float]]
    decision_vectors: list[list[float]]
    constraint_evaluations: list[ConstraintEvaluation]
    dependency_status: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["constraint_evaluations"] = [
            evaluation.to_dict() for evaluation in self.constraint_evaluations
        ]
        return payload


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _symbol_status(path: str) -> tuple[object | None, dict[str, Any]]:
    module_name, _, symbol_name = path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        symbol = getattr(module, symbol_name)
    except Exception as exc:
        return None, {"path": path, "available": False, "error": str(exc)}
    return symbol, {"path": path, "available": True, "error": None}


def is_pymoo_available() -> dict[str, Any]:
    if importlib.util.find_spec("pymoo") is None:
        return {
            "installed": False,
            "version": None,
            "import_status": "dependency_missing",
            "recommended_status": "dependency_missing",
            "api_symbols": {},
        }
    symbols = {
        "Problem": "pymoo.core.problem.Problem",
        "NSGA2": "pymoo.algorithms.moo.nsga2.NSGA2",
        "minimize": "pymoo.optimize.minimize",
    }
    statuses: dict[str, Any] = {}
    for name, path in symbols.items():
        _, status = _symbol_status(path)
        statuses[name] = status
    import_status = "imported" if all(item["available"] for item in statuses.values()) else "api_import_failed"
    return {
        "installed": True,
        "version": _package_version("pymoo"),
        "import_status": import_status,
        "recommended_status": (
            "usable_with_version_guards" if import_status == "imported" else "api_import_failed"
        ),
        "api_symbols": statuses,
    }


class PymooConstrainedProblemWrapper:
    """pymoo Problem wrapper using repository constrained toy evaluators."""

    def __init__(self, problem) -> None:
        problem_symbol, status = _symbol_status("pymoo.core.problem.Problem")
        if problem_symbol is None:
            raise ImportError(status["error"] or "pymoo Problem import failed")
        self._problem_cls = problem_symbol
        self.problem = problem
        self.evaluations = 0
        self.last_decision_vectors: list[list[float]] = []
        self.last_objective_vectors: list[list[float]] = []
        self.last_constraint_evaluations: list[ConstraintEvaluation] = []
        bounds = problem.source_bounds()
        import numpy as np

        class Wrapped(problem_symbol):  # type: ignore[misc, valid-type]
            def __init__(inner_self, outer: PymooConstrainedProblemWrapper) -> None:
                inner_self.outer = outer
                super().__init__(
                    n_var=int(problem.dimension),
                    n_obj=int(problem.objective_count),
                    n_ieq_constr=int(problem.inequality_count),
                    n_eq_constr=int(problem.equality_count),
                    xl=np.asarray([float(low) for low, _ in bounds], dtype=float),
                    xu=np.asarray([float(high) for _, high in bounds], dtype=float),
                )

            def _evaluate(inner_self, x, out, *args, **kwargs) -> None:
                inner_self.outer._evaluate(x, out)

        self.pymoo_problem = Wrapped(self)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        import numpy as np

        rows = np.atleast_2d(x)
        objective_rows: list[list[float]] = []
        inequality_rows: list[list[float]] = []
        equality_rows: list[list[float]] = []
        decision_rows: list[list[float]] = []
        evaluations: list[ConstraintEvaluation] = []
        for offset, row in enumerate(rows):
            genome = [float(value) for value in row.tolist()]
            objectives = [float(value) for value in self.problem.fitness(genome)]
            validate_fitness_vector(
                objectives,
                problem_name=self.problem.name,
                genome=genome,
                location="pymoo_constrained_nsga2",
                generation=0,
                evaluation_index=self.evaluations + offset,
            )
            evaluation = self.problem.evaluate_constraints(genome)
            objective_rows.append(objectives)
            inequality_rows.append(list(evaluation.inequality_values))
            equality_rows.append(list(evaluation.equality_values))
            decision_rows.append(genome)
            evaluations.append(evaluation)
        self.evaluations += len(objective_rows)
        self.last_decision_vectors = decision_rows
        self.last_objective_vectors = objective_rows
        self.last_constraint_evaluations = evaluations
        out["F"] = np.asarray(objective_rows, dtype=float)
        out["G"] = np.asarray(inequality_rows, dtype=float)
        if int(getattr(self.problem, "equality_count", 0)) > 0:
            out["H"] = np.asarray(equality_rows, dtype=float)


def make_pymoo_constrained_problem(problem) -> PymooConstrainedProblemWrapper:
    return PymooConstrainedProblemWrapper(problem)


def _skipped_result(
    *,
    seed: int,
    budget: int,
    reason: str,
    dependency_status: dict[str, Any],
) -> PymooConstrainedComparatorResult:
    return PymooConstrainedComparatorResult(
        algorithm="pymoo_constrained_nsga2",
        status="skipped",
        seed=seed,
        requested_budget=budget,
        actual_evaluations=0,
        runtime_seconds=0.0,
        objective_vectors=[],
        decision_vectors=[],
        constraint_evaluations=[],
        dependency_status=dependency_status,
        warnings=[reason],
        skip_reason=reason,
        metadata={"operator_family_difference": "not_applicable_dependency_skipped"},
    )


def run_pymoo_constrained_nsga2(
    *,
    problem,
    seed: int,
    budget: int,
    population_size: int = 20,
    force_skip: bool = False,
) -> PymooConstrainedComparatorResult:
    dependency_status = is_pymoo_available()
    if force_skip:
        return _skipped_result(
            seed=seed,
            budget=budget,
            reason="pymoo comparator forced skip",
            dependency_status=dependency_status,
        )
    if not dependency_status["installed"]:
        return _skipped_result(
            seed=seed,
            budget=budget,
            reason="pymoo is not installed",
            dependency_status=dependency_status,
        )
    if dependency_status["import_status"] != "imported":
        return _skipped_result(
            seed=seed,
            budget=budget,
            reason="pymoo API import failed",
            dependency_status=dependency_status,
        )

    started = time.perf_counter()
    try:
        import numpy as np
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        wrapper = PymooConstrainedProblemWrapper(problem)
        algorithm = NSGA2(pop_size=int(population_size), eliminate_duplicates=False)
        result = minimize(
            wrapper.pymoo_problem,
            algorithm,
            termination=("n_eval", int(budget)),
            seed=int(seed),
            verbose=False,
            save_history=False,
        )
        runtime = time.perf_counter() - started
        pop = getattr(result, "pop", None)
        decision_vectors: list[list[float]] = []
        objective_vectors: list[list[float]] = []
        if pop is not None:
            pop_x = pop.get("X")
            pop_f = pop.get("F")
            if pop_x is not None:
                decision_vectors = [
                    [float(value) for value in row.tolist()]
                    for row in np.atleast_2d(pop_x)
                ]
            if pop_f is not None:
                objective_vectors = [
                    [float(value) for value in row.tolist()]
                    for row in np.atleast_2d(pop_f)
                ]
        if not decision_vectors:
            result_x = getattr(result, "X", None)
            if result_x is not None:
                decision_vectors = [
                    [float(value) for value in row.tolist()]
                    for row in np.atleast_2d(result_x)
                ]
        if decision_vectors:
            evaluations = [problem.evaluate_constraints(genome) for genome in decision_vectors]
            if not objective_vectors or len(objective_vectors) != len(decision_vectors):
                objective_vectors = [[float(value) for value in problem.fitness(genome)] for genome in decision_vectors]
        else:
            evaluations = []
        actual_evaluations = int(wrapper.evaluations)
        status = "success"
        failure_reason = None
        if actual_evaluations != int(budget):
            status = "failed"
            failure_reason = (
                f"actual_evaluations mismatch ({actual_evaluations}/{int(budget)})"
            )
        return PymooConstrainedComparatorResult(
            algorithm="pymoo_constrained_nsga2",
            status=status,
            seed=seed,
            requested_budget=int(budget),
            actual_evaluations=actual_evaluations,
            runtime_seconds=runtime,
            objective_vectors=objective_vectors,
            decision_vectors=decision_vectors,
            constraint_evaluations=evaluations,
            dependency_status=dependency_status,
            warnings=[
                "operator_family_difference:pymoo_standard_nsga2",
                "constraint_sign_convention:value_le_0_satisfied_verified_by_wrapper",
            ],
            failure_reason=failure_reason,
            metadata={
                "pymoo_version": dependency_status.get("version"),
                "operator_family": "pymoo_standard_nsga2",
                "operator_family_difference": "warning",
                "constraint_sign_convention": "repository inequality value <= 0 is satisfied; values are emitted as G",
                "termination": "n_eval",
                "seed_parameter": "minimize(seed=seed)",
                "actual_evaluation_source": "PymooConstrainedProblemWrapper.evaluations",
                "penalty_used": False,
                "repair_used": False,
            },
        )
    except Exception as exc:
        return PymooConstrainedComparatorResult(
            algorithm="pymoo_constrained_nsga2",
            status="failed",
            seed=seed,
            requested_budget=int(budget),
            actual_evaluations=0,
            runtime_seconds=time.perf_counter() - started,
            objective_vectors=[],
            decision_vectors=[],
            constraint_evaluations=[],
            dependency_status=dependency_status,
            warnings=[],
            failure_reason=str(exc),
            metadata={"pymoo_version": dependency_status.get("version")},
        )


def deap_secondary_status() -> dict[str, Any]:
    installed = importlib.util.find_spec("deap") is not None
    return {
        "implemented": False,
        "primary_recommendation": False,
        "status": "secondary_hold",
        "installed": installed,
        "version": _package_version("deap") if installed else None,
        "reason": "DEAP comparator is intentionally not implemented in this pass",
    }


__all__ = [
    "PymooConstrainedComparatorResult",
    "PymooConstrainedProblemWrapper",
    "deap_secondary_status",
    "is_pymoo_available",
    "make_pymoo_constrained_problem",
    "run_pymoo_constrained_nsga2",
]
