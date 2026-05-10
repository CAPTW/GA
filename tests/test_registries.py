from __future__ import annotations

from ga_lab.algorithms.nsga2 import run_nsga2
from ga_lab.algorithms.registry import algorithm_plugin_names, build_algorithm_from_name
from ga_lab.config import GAConfig
from ga_lab.core.crossover import crossover_plugin_names
from ga_lab.core.mutation import mutation_plugin_names
from ga_lab.core.representation import representation_plugin_names
from ga_lab.core.selection import selection_plugin_names
from ga_lab.factory import build_runtime_context
from ga_lab.problems.registry import problem_plugin_names


def test_registry_lists_expose_documented_plugin_names() -> None:
    assert algorithm_plugin_names() == ("ga", "nsga2", "hybrid_ga")
    assert problem_plugin_names() == (
        "onemax",
        "tsp",
        "knapsack",
        "constrained_sphere",
        "constrained_box_quadratic",
        "constrained_equality_plane_quadratic",
        "constrained_zdt_box_toy",
        "constrained_dtlz_box_toy",
        "zdt1",
        "zdt2",
        "zdt3",
        "dtlz2",
        "dtlz3",
        "dtlz4",
        "wfg1",
        "wfg2",
    )
    assert representation_plugin_names() == ("bit", "real", "permutation")
    assert selection_plugin_names() == (
        "tournament",
        "rank",
        "roulette",
        "crowded_tournament",
        "nsga2_tournament",
    )
    assert crossover_plugin_names() == ("one_point", "uniform", "arithmetic", "order")
    assert mutation_plugin_names() == ("bit_flip", "gaussian", "swap", "inversion")


def test_algorithm_alias_nsga_ii_remains_supported() -> None:
    config = GAConfig(
        run_name="alias_nsga_ii",
        problem="zdt1",
        algorithm="nsga_ii",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=12,
        genome_length=6,
        generations=4,
        crossover_rate=0.9,
        mutation_rate=0.1,
        tournament_size=2,
        maximize=False,
        objective_directions=[False, False],
    )

    runtime = build_runtime_context(config)

    assert runtime.algorithm_fn is run_nsga2
    assert build_algorithm_from_name("nsga-ii") is run_nsga2
