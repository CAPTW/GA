from __future__ import annotations

import copy
import importlib
import importlib.util
import math
import random
import time
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from ga_lab.api import run_config
from ga_lab.config import GAConfig
from ga_lab.experiment.budget_baseline_comparison import configured_evaluation_budget
from ga_lab.experiment.mo_baselines import MultiObjectiveBaselineResult, evaluate_objective_vector
from ga_lab.experiment.mo_metrics import coverage_indicator, evaluate_front_metrics, nondominated_vectors
from ga_lab.factory import build_runtime_context


METRIC_SPECS: dict[str, dict[str, bool]] = {
    "hypervolume_2d": {"higher_is_better": True},
    "reference_front_distance": {"higher_is_better": False},
    "generational_distance": {"higher_is_better": False},
    "inverted_generational_distance": {"higher_is_better": False},
    "spacing": {"higher_is_better": False},
    "nondominated_count": {"higher_is_better": True},
    "coverage_indicator": {"higher_is_better": True},
}


@dataclass(slots=True)
class ExternalLibraryStatus:
    library: str
    installed: bool
    version: str | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExternalMOComparatorResult:
    problem_name: str
    algorithm_name: str
    library_name: str
    seed: int
    requested_budget: int
    evaluations: int
    runtime_seconds: float
    status: str
    success: bool
    error_message: str | None
    objective_vectors: list[list[float]]
    nondominated_objective_vectors: list[list[float]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MOProblemContext:
    config: GAConfig
    problem: Any
    problem_name: str
    genome_length: int
    lower_bound: float
    upper_bound: float
    objective_directions: list[bool]
    objective_count: int


def optional_library_status(name: str) -> ExternalLibraryStatus:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return ExternalLibraryStatus(
            library=name,
            installed=False,
            version=None,
            reason=f"{name} is not installed",
        )
    module = importlib.import_module(name)
    return ExternalLibraryStatus(
        library=name,
        installed=True,
        version=str(getattr(module, "__version__", "unknown")),
        reason=None,
    )


def build_problem_context(config: GAConfig) -> MOProblemContext:
    runtime = build_runtime_context(config)
    low = config.representation_options.get("low")
    high = config.representation_options.get("high")
    if not isinstance(low, int | float) or not isinstance(high, int | float):
        raise ValueError("external MO comparison requires numeric low/high representation bounds")
    probe_genome = [float(low)] * config.genome_length
    first_vector = evaluate_objective_vector(
        runtime.problem,
        probe_genome,
        problem_name=config.problem,
        location="external_comparator_probe",
        evaluation_index=0,
    )
    return MOProblemContext(
        config=config,
        problem=runtime.problem,
        problem_name=config.problem,
        genome_length=config.genome_length,
        lower_bound=float(low),
        upper_bound=float(high),
        objective_directions=list(config.objective_directions)
        if config.objective_directions
        else [False] * len(first_vector),
        objective_count=len(first_vector),
    )


def result_to_front_row(
    result: ExternalMOComparatorResult,
    *,
    reference_front: list[list[float]],
    reference_point: list[float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "problem": result.problem_name,
        "algorithm": result.algorithm_name,
        "library": result.library_name,
        "seed": result.seed,
        "requested_budget": result.requested_budget,
        "actual_evaluations": result.evaluations,
        "runtime_seconds": result.runtime_seconds,
        "status": result.status,
        "success": result.success,
        "error_message": result.error_message,
        "objective_vectors": result.objective_vectors,
        "nondominated_objective_vectors": result.nondominated_objective_vectors,
        "decision_vectors": list(result.metadata.get("decision_vectors", [])),
        "front_decision_vectors": list(result.metadata.get("front_decision_vectors", [])),
        "metadata": dict(result.metadata),
    }
    if not result.success:
        row.update(
            {
                "archive_size": 0,
                "nondominated_count": 0,
                "hypervolume_2d": None,
                "reference_front_distance": None,
                "generational_distance": None,
                "inverted_generational_distance": None,
                "spacing": None,
                "objective_min_max": {"min": [], "max": []},
            }
        )
        return row

    directions = [
        bool(value)
        for value in result.metadata.get("objective_directions", [False] * len(reference_point))
    ]
    row.update(
        evaluate_front_metrics(
            result.objective_vectors,
            directions=directions,
            reference_front=reference_front,
            reference_point=reference_point,
        )
    )
    return row


def run_internal_nsga2(
    config: GAConfig,
    *,
    seed: int,
    output_root: str,
) -> ExternalMOComparatorResult:
    started = time.perf_counter()
    try:
        runtime_config = GAConfig.from_dict(config.to_dict())
        runtime_config.seed = seed
        runtime_config.run_name = f"{config.run_name}_internal_seed{seed}"
        runtime_config.algorithm_options = dict(runtime_config.algorithm_options)
        runtime_config.algorithm_options["_return_final_population"] = True
        result = run_config(runtime_config, output_root=output_root)
        summary = result.raw_summary
        front = [list(vector) for vector in summary["pareto_front_vectors"]]
        directions = [bool(value) for value in summary["objective_directions"]]
        elapsed = time.perf_counter() - started
        metadata = {
            "objective_directions": directions,
            "output_dir": result.output_dir,
            "summary_path": result.summary_path,
            "population_size": config.population_size,
            "configured_generations": config.generations,
            "configured_budget_formula": "population_size * (3 * generations + 2)",
            "decision_vectors": [list(genome) for genome in summary.get("final_population", [])],
            "front_decision_vectors": [
                list(genome) for genome in summary.get("pareto_front_genomes", [])
            ],
        }
        diagnostics_payload = summary.get("nsga2_diagnostics")
        if isinstance(diagnostics_payload, dict):
            metadata["diagnostics_enabled"] = True
            metadata["nsga2_diagnostics"] = diagnostics_payload
        low_g_tail_mutation_stats = summary.get("low_g_tail_mutation_stats")
        if isinstance(low_g_tail_mutation_stats, dict):
            metadata["low_g_tail_mutation_stats"] = low_g_tail_mutation_stats
        spread_preserving_variation_stats = summary.get("spread_preserving_variation_stats")
        if isinstance(spread_preserving_variation_stats, dict):
            metadata["spread_preserving_variation_stats"] = spread_preserving_variation_stats
        return ExternalMOComparatorResult(
            problem_name=config.problem,
            algorithm_name="internal_nsga2",
            library_name="internal",
            seed=seed,
            requested_budget=int(
                summary.get("configured_budget", configured_evaluation_budget(config))
            ),
            evaluations=int(summary["actual_evaluations_used"]),
            runtime_seconds=float(summary.get("runtime_seconds", elapsed)),
            status="success",
            success=True,
            error_message=None,
            objective_vectors=front,
            nondominated_objective_vectors=nondominated_vectors(front, directions),
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - defensive capture for reports
        return ExternalMOComparatorResult(
            problem_name=config.problem,
            algorithm_name="internal_nsga2",
            library_name="internal",
            seed=seed,
            requested_budget=0,
            evaluations=0,
            runtime_seconds=time.perf_counter() - started,
            status="failed",
            success=False,
            error_message=str(exc),
            objective_vectors=[],
            nondominated_objective_vectors=[],
            metadata={},
        )


def run_random_archive_anchor(
    result: MultiObjectiveBaselineResult,
) -> ExternalMOComparatorResult:
    return ExternalMOComparatorResult(
        problem_name=result.problem_name,
        algorithm_name=result.algorithm_name,
        library_name="internal_baseline",
        seed=result.seed,
        requested_budget=result.budget,
        evaluations=result.evaluations,
        runtime_seconds=result.runtime_seconds,
        status="success" if result.success else "failed",
        success=result.success,
        error_message=result.error_message,
        objective_vectors=result.objective_vectors,
        nondominated_objective_vectors=result.nondominated_objective_vectors,
        metadata=result.metadata,
    )


def run_pymoo_nsga2(
    config: GAConfig,
    *,
    seed: int,
    budget: int,
) -> ExternalMOComparatorResult:
    status = optional_library_status("pymoo")
    if not status.installed:
        return ExternalMOComparatorResult(
            problem_name=config.problem,
            algorithm_name="pymoo_nsga2",
            library_name="pymoo",
            seed=seed,
            requested_budget=budget,
            evaluations=0,
            runtime_seconds=0.0,
            status="skipped",
            success=False,
            error_message=status.reason,
            objective_vectors=[],
            nondominated_objective_vectors=[],
            metadata={"library_status": status.to_dict()},
        )

    context = build_problem_context(config)
    started = time.perf_counter()
    try:
        import numpy as np
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling

        class WrappedProblem(Problem):
            def __init__(self) -> None:
                problem_kwargs = {
                    "n_var": context.genome_length,
                    "n_obj": context.objective_count,
                    "xl": np.full(context.genome_length, context.lower_bound, dtype=float),
                    "xu": np.full(context.genome_length, context.upper_bound, dtype=float),
                }
                try:
                    super().__init__(n_ieq_constr=0, **problem_kwargs)
                except TypeError:
                    super().__init__(n_constr=0, **problem_kwargs)
                self.evaluations = 0

            def _evaluate(self, x, out, *args, **kwargs) -> None:
                rows = np.atleast_2d(x)
                front_values: list[list[float]] = []
                for row_index, row in enumerate(rows):
                    vector = evaluate_objective_vector(
                        context.problem,
                        row.tolist(),
                        problem_name=context.problem_name,
                        location="pymoo_nsga2",
                        evaluation_index=self.evaluations + row_index,
                    )
                    front_values.append(
                        [
                            -value if context.objective_directions[index] else value
                            for index, value in enumerate(vector)
                        ]
                    )
                self.evaluations += len(front_values)
                out["F"] = np.asarray(front_values, dtype=float)

        problem = WrappedProblem()
        mutation_probability = config.mutation_rate if config.mutation_rate > 0 else 1.0 / max(
            1, context.genome_length
        )
        algorithm = NSGA2(
            pop_size=config.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=config.crossover_rate, eta=15),
            mutation=PM(prob=mutation_probability, eta=20),
            eliminate_duplicates=False,
        )
        result = minimize(
            problem,
            algorithm,
            termination=("n_eval", budget),
            seed=seed,
            verbose=False,
            save_history=False,
        )
        result_front = getattr(result, "F", None)
        result_decisions = getattr(result, "X", None)
        result_population = getattr(result, "pop", None)
        if result_front is None:
            raise ValueError("pymoo returned no objective matrix")
        front = [
            [
                -float(value) if context.objective_directions[index] else float(value)
                for index, value in enumerate(row.tolist())
            ]
            for row in np.atleast_2d(result_front)
        ]
        front_decision_vectors = (
            [list(map(float, row.tolist())) for row in np.atleast_2d(result_decisions)]
            if result_decisions is not None
            else []
        )
        population_decision_vectors: list[list[float]] = []
        population_objective_vectors: list[list[float]] = []
        if result_population is not None:
            population_decisions = result_population.get("X")
            population_front = result_population.get("F")
            if population_decisions is not None:
                population_decision_vectors = [
                    list(map(float, row.tolist())) for row in np.atleast_2d(population_decisions)
                ]
            if population_front is not None:
                population_objective_vectors = [
                    [
                        -float(value) if context.objective_directions[index] else float(value)
                        for index, value in enumerate(row.tolist())
                    ]
                    for row in np.atleast_2d(population_front)
                ]
        decision_vectors = population_decision_vectors or front_decision_vectors
        nondominated = nondominated_vectors(front, context.objective_directions)
        return ExternalMOComparatorResult(
            problem_name=config.problem,
            algorithm_name="pymoo_nsga2",
            library_name="pymoo",
            seed=seed,
            requested_budget=budget,
            evaluations=int(problem.evaluations),
            runtime_seconds=time.perf_counter() - started,
            status="success",
            success=True,
            error_message=None,
            objective_vectors=front,
            nondominated_objective_vectors=nondominated,
            metadata={
                "library_status": status.to_dict(),
                "objective_directions": context.objective_directions,
                "population_size": config.population_size,
                "termination": "n_eval",
                "requested_budget": budget,
                "actual_generations_hint": getattr(getattr(result, "algorithm", None), "n_gen", None),
                "operator_family": "pymoo_standard_sbx_pm",
                "decision_vectors": decision_vectors,
                "front_decision_vectors": front_decision_vectors,
                "population_decision_vectors": decision_vectors,
                "population_objective_vectors": population_objective_vectors,
                "front_objective_vectors": front,
                "initialization": "FloatRandomSampling",
                "crossover_type": "SBX",
                "crossover_probability": float(config.crossover_rate),
                "crossover_eta": 15.0,
                "mutation_type": "PM",
                "mutation_probability": float(mutation_probability),
                "mutation_eta": 20.0,
                "duplicate_handling": "eliminate_duplicates=False",
                "repair_or_bounds_handling": "bounded_operators",
                "survival": "rank_crowding",
                "crowding": "pymoo_crowding_distance",
                "selection": "binary_tournament",
            },
        )
    except Exception as exc:  # pragma: no cover - exercised only when optional deps exist
        return ExternalMOComparatorResult(
            problem_name=config.problem,
            algorithm_name="pymoo_nsga2",
            library_name="pymoo",
            seed=seed,
            requested_budget=budget,
            evaluations=0,
            runtime_seconds=time.perf_counter() - started,
            status="failed",
            success=False,
            error_message=str(exc),
            objective_vectors=[],
            nondominated_objective_vectors=[],
            metadata={"library_status": status.to_dict()},
        )


def run_deap_nsga2(
    config: GAConfig,
    *,
    seed: int,
    budget: int,
) -> ExternalMOComparatorResult:
    status = optional_library_status("deap")
    if not status.installed:
        return ExternalMOComparatorResult(
            problem_name=config.problem,
            algorithm_name="deap_nsga2",
            library_name="deap",
            seed=seed,
            requested_budget=budget,
            evaluations=0,
            runtime_seconds=0.0,
            status="skipped",
            success=False,
            error_message=status.reason,
            objective_vectors=[],
            nondominated_objective_vectors=[],
            metadata={"library_status": status.to_dict()},
        )

    context = build_problem_context(config)
    started = time.perf_counter()
    random_state = random.getstate()
    try:
        from deap import base, creator, tools

        fitness_name = "FitnessMultiZDTMin"
        individual_name = "IndividualRealZDT"
        weights = tuple(-1.0 if not maximize else 1.0 for maximize in context.objective_directions)
        if not hasattr(creator, fitness_name):
            creator.create(fitness_name, base.Fitness, weights=weights)
        if not hasattr(creator, individual_name):
            creator.create(individual_name, list, fitness=getattr(creator, fitness_name))

        random.seed(seed)
        toolbox = base.Toolbox()
        toolbox.register(
            "attr_float",
            random.uniform,
            context.lower_bound,
            context.upper_bound,
        )
        toolbox.register(
            "individual",
            tools.initRepeat,
            getattr(creator, individual_name),
            toolbox.attr_float,
            n=context.genome_length,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            low=context.lower_bound,
            up=context.upper_bound,
            eta=15.0,
        )
        toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=context.lower_bound,
            up=context.upper_bound,
            eta=20.0,
            indpb=max(config.mutation_rate, 1.0 / max(1, context.genome_length)),
        )
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("clone", copy.deepcopy)

        evaluations = 0

        def evaluate(individual: list[float]) -> tuple[float, ...]:
            nonlocal evaluations
            vector = evaluate_objective_vector(
                context.problem,
                list(individual),
                problem_name=context.problem_name,
                location="deap_nsga2",
                evaluation_index=evaluations,
            )
            evaluations += 1
            return tuple(vector)

        population = toolbox.population(n=config.population_size)
        initial_batch = min(len(population), budget)
        for individual in population[:initial_batch]:
            individual.fitness.values = evaluate(individual)
        population = toolbox.select(population[:initial_batch], len(population[:initial_batch]))

        while evaluations + len(population) <= budget and population:
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [toolbox.clone(individual) for individual in offspring]
            for left, right in zip(offspring[::2], offspring[1::2], strict=False):
                if random.random() <= config.crossover_rate:
                    toolbox.mate(left, right)
                if random.random() <= config.mutation_rate:
                    toolbox.mutate(left)
                if random.random() <= config.mutation_rate:
                    toolbox.mutate(right)
                del left.fitness.values
                del right.fitness.values

            invalid = [individual for individual in offspring if not individual.fitness.valid]
            remaining_budget = budget - evaluations
            for individual in invalid[:remaining_budget]:
                individual.fitness.values = evaluate(individual)
            offspring = [individual for individual in offspring if individual.fitness.valid]
            if not offspring:
                break
            population = toolbox.select(population + offspring, config.population_size)

        if not population:
            raise ValueError("DEAP comparator produced an empty population")
        nondominated_front = tools.sortNondominated(
            population,
            len(population),
            first_front_only=True,
        )[0]
        population_objective_vectors = [list(individual.fitness.values) for individual in population]
        population_decision_vectors = [list(map(float, individual)) for individual in population]
        objective_vectors = [list(individual.fitness.values) for individual in nondominated_front]
        front_decision_vectors = [list(map(float, individual)) for individual in nondominated_front]
        nondominated = nondominated_vectors(objective_vectors, context.objective_directions)
        mutation_indpb = max(config.mutation_rate, 1.0 / max(1, context.genome_length))
        return ExternalMOComparatorResult(
            problem_name=config.problem,
            algorithm_name="deap_nsga2",
            library_name="deap",
            seed=seed,
            requested_budget=budget,
            evaluations=evaluations,
            runtime_seconds=time.perf_counter() - started,
            status="success",
            success=True,
            error_message=None,
            objective_vectors=objective_vectors,
            nondominated_objective_vectors=nondominated,
            metadata={
                "library_status": status.to_dict(),
                "objective_directions": context.objective_directions,
                "population_size": config.population_size,
                "operator_family": "deap_selNSGA2_sbx_poly",
                "requested_budget": budget,
                "decision_vectors": population_decision_vectors,
                "front_decision_vectors": front_decision_vectors,
                "population_decision_vectors": population_decision_vectors,
                "population_objective_vectors": population_objective_vectors,
                "front_objective_vectors": objective_vectors,
                "initialization": "tools.initRepeat(random.uniform)",
                "crossover_type": "cxSimulatedBinaryBounded",
                "crossover_probability": float(config.crossover_rate),
                "crossover_eta": 15.0,
                "mutation_type": "mutPolynomialBounded",
                "mutation_probability": float(config.mutation_rate),
                "mutation_eta": 20.0,
                "mutation_indpb": float(mutation_indpb),
                "duplicate_handling": "none",
                "repair_or_bounds_handling": "bounded_operators",
                "survival": "selNSGA2",
                "crowding": "deap_selNSGA2_crowding_distance",
                "selection": "selTournamentDCD + selNSGA2",
            },
        )
    except Exception as exc:  # pragma: no cover - exercised only when optional deps exist
        return ExternalMOComparatorResult(
            problem_name=config.problem,
            algorithm_name="deap_nsga2",
            library_name="deap",
            seed=seed,
            requested_budget=budget,
            evaluations=0,
            runtime_seconds=time.perf_counter() - started,
            status="failed",
            success=False,
            error_message=str(exc),
            objective_vectors=[],
            nondominated_objective_vectors=[],
            metadata={"library_status": status.to_dict()},
        )
    finally:
        random.setstate(random_state)


def paired_metric_summary(
    *,
    internal_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    metric_spec = METRIC_SPECS[metric_name]
    higher_is_better = metric_spec["higher_is_better"]
    comparator_by_seed = {
        int(row["seed"]): row
        for row in comparator_rows
        if row.get("success") and int(row.get("seed", -1)) >= 0
    }
    wins = ties = losses = 0
    deltas: list[float] = []
    comparable_seeds: list[int] = []

    for internal in internal_rows:
        if not internal.get("success"):
            continue
        seed = int(internal["seed"])
        comparator = comparator_by_seed.get(seed)
        if comparator is None:
            continue

        if metric_name == "coverage_indicator":
            directions = [
                bool(value)
                for value in internal.get("metadata", {}).get("objective_directions", [False, False])
            ]
            internal_metric = coverage_indicator(
                internal.get("nondominated_objective_vectors", []),
                comparator.get("nondominated_objective_vectors", []),
                directions,
            )
            comparator_metric = coverage_indicator(
                comparator.get("nondominated_objective_vectors", []),
                internal.get("nondominated_objective_vectors", []),
                directions,
            )
        else:
            internal_metric = internal.get(metric_name)
            comparator_metric = comparator.get(metric_name)

        if not (
            isinstance(internal_metric, int | float)
            and isinstance(comparator_metric, int | float)
            and math.isfinite(float(internal_metric))
            and math.isfinite(float(comparator_metric))
        ):
            continue

        comparable_seeds.append(seed)
        delta = float(internal_metric) - float(comparator_metric)
        deltas.append(delta)

        if math.isclose(float(internal_metric), float(comparator_metric), rel_tol=1e-12, abs_tol=1e-12):
            ties += 1
        elif (float(internal_metric) > float(comparator_metric)) == higher_is_better:
            wins += 1
        else:
            losses += 1

    return {
        "metric": metric_name,
        "comparable_seed_count": len(comparable_seeds),
        "internal_win": wins,
        "tie": ties,
        "external_win": losses,
        "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "median_delta": median(deltas) if deltas else None,
        "comparable_seeds": comparable_seeds,
    }
