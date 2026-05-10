from __future__ import annotations

import math

from ga_lab.problems.registry import build_problem_from_name
from ga_lab.experiment.mo_metrics import reference_front_for_problem


def test_zdt2_and_zdt3_return_finite_objective_vectors() -> None:
    zdt2 = build_problem_from_name("zdt2", {})
    zdt3 = build_problem_from_name("zdt3", {})
    genome = [0.25, 0.1, 0.2, 0.3, 0.4, 0.5]

    zdt2_values = list(zdt2.fitness(genome))
    zdt3_values = list(zdt3.fitness(genome))

    assert zdt2.name == "zdt2"
    assert zdt3.name == "zdt3"
    assert len(zdt2_values) == 2
    assert len(zdt3_values) == 2
    assert all(math.isfinite(value) for value in zdt2_values)
    assert all(math.isfinite(value) for value in zdt3_values)


def test_dtlz2_dtlz3_and_dtlz4_return_finite_two_objective_vectors() -> None:
    dtlz2 = build_problem_from_name("dtlz2", {"objective_count": 2})
    dtlz3 = build_problem_from_name("dtlz3", {"objective_count": 2})
    dtlz4 = build_problem_from_name("dtlz4", {"objective_count": 2})
    genome = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    dtlz2_values = list(dtlz2.fitness(genome))
    dtlz3_values = list(dtlz3.fitness(genome))
    dtlz4_values = list(dtlz4.fitness(genome))
    dtlz2_metadata = dtlz2.metadata()
    dtlz3_metadata = dtlz3.metadata()
    dtlz4_metadata = dtlz4.metadata()

    assert dtlz2.name == "dtlz2"
    assert dtlz3.name == "dtlz3"
    assert dtlz4.name == "dtlz4"
    assert len(dtlz2_values) == 2
    assert len(dtlz3_values) == 2
    assert len(dtlz4_values) == 2
    assert all(math.isfinite(value) for value in dtlz2_values)
    assert all(math.isfinite(value) for value in dtlz3_values)
    assert all(math.isfinite(value) for value in dtlz4_values)
    assert dtlz2_metadata.min_genome_length == 2
    assert dtlz3_metadata.min_genome_length == 2
    assert dtlz4_metadata.min_genome_length == 2
    assert dtlz2_metadata.default_objective_directions == (False, False)
    assert dtlz3_metadata.default_objective_directions == (False, False)
    assert dtlz4_metadata.default_objective_directions == (False, False)


def test_reference_fronts_exist_for_cross_benchmark_suite() -> None:
    for problem_name in ("zdt1", "zdt2", "zdt3", "dtlz2", "dtlz3", "dtlz4"):
        front = reference_front_for_problem(problem_name, point_count=51, objective_count=2)
        assert len(front) >= 10
        assert all(len(point) == 2 for point in front)
        assert all(math.isfinite(value) for point in front for value in point)
