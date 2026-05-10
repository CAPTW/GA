from __future__ import annotations

from dataclasses import dataclass

from ga_lab.algorithms.registry import AlgorithmFn, build_algorithm_from_name
from ga_lab.config import GAConfig
from ga_lab.core.crossover import CrossoverFn, build_crossover_fn, get_crossover_plugin
from ga_lab.core.mutation import MutationFn, build_mutation_fn, get_mutation_plugin
from ga_lab.core.representation import (
    Genome,
    InitFn,
    RepresentationAdapter,
    build_representation_adapter,
    get_representation_plugin,
)
from ga_lab.core.selection import SelectionFn, build_selection_fn, get_selection_plugin
from ga_lab.problems.base import ProblemMetadata, problem_metadata
from ga_lab.problems.registry import build_problem_from_name, get_problem_plugin


@dataclass(slots=True)
class OperatorBundle:
    selection_fn: SelectionFn
    crossover_fn: CrossoverFn
    mutation_fn: MutationFn
    init_fn: InitFn
    representation: RepresentationAdapter


@dataclass(slots=True)
class RuntimeContext:
    problem: object
    problem_metadata: ProblemMetadata
    operators: OperatorBundle
    algorithm_fn: AlgorithmFn


def normalize_plugin_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def build_problem_instance(config: GAConfig):
    return build_problem_from_name(config.problem, config.problem_options)


def _normalize_representation_name(name: str) -> str:
    return normalize_plugin_name(name)


def _repair_child_genome(
    adapter: RepresentationAdapter,
    genome_length: int,
    genome: Genome,
) -> Genome:
    return adapter.repair(genome, genome_length)


def _validate_problem_contract(
    config: GAConfig,
    metadata: ProblemMetadata,
) -> None:
    representation = _normalize_representation_name(config.representation)
    compatible = tuple(
        _normalize_representation_name(name) for name in metadata.compatible_representations
    )
    if compatible and representation not in compatible:
        raise ValueError(
            f"Problem '{metadata.name}' requires one of representations {list(compatible)}, "
            f"got '{config.representation}'"
        )
    if (
        metadata.exact_genome_length is not None
        and config.genome_length != metadata.exact_genome_length
    ):
        raise ValueError(
            f"Problem '{metadata.name}' requires genome_length={metadata.exact_genome_length}, "
            f"got {config.genome_length}"
        )
    if metadata.min_genome_length is not None and config.genome_length < metadata.min_genome_length:
        raise ValueError(
            f"Problem '{metadata.name}' requires genome_length>={metadata.min_genome_length}, "
            f"got {config.genome_length}"
        )


def _validate_operator_contract(config: GAConfig) -> None:
    representation = _normalize_representation_name(config.representation)
    algorithm = normalize_plugin_name(config.algorithm)
    selection_spec = get_selection_plugin(config.selection)
    crossover_spec = get_crossover_plugin(config.crossover)
    mutation_spec = get_mutation_plugin(config.mutation)
    get_representation_plugin(config.representation)
    get_problem_plugin(config.problem)

    if not crossover_spec.supports_representation(representation):
        raise ValueError(
            "Crossover "
            f"'{config.crossover}' is incompatible with representation "
            f"'{config.representation}'"
        )
    if not mutation_spec.supports_representation(representation):
        raise ValueError(
            "Mutation "
            f"'{config.mutation}' is incompatible with representation "
            f"'{config.representation}'"
        )
    if not selection_spec.supports_algorithm(algorithm):
        raise ValueError(
            f"Selection '{config.selection}' is incompatible with algorithm '{config.algorithm}'"
        )


def build_operator_bundle(config: GAConfig) -> OperatorBundle:
    representation = build_representation_adapter(config)
    crossover_fn = build_crossover_fn(config)
    mutation_fn = build_mutation_fn(config)

    def wrapped_crossover(parent_a, parent_b, rng):
        child_a, child_b = crossover_fn(parent_a, parent_b, rng)
        return (
            _repair_child_genome(representation, config.genome_length, child_a),
            _repair_child_genome(representation, config.genome_length, child_b),
        )

    return OperatorBundle(
        selection_fn=build_selection_fn(config),
        crossover_fn=wrapped_crossover,
        mutation_fn=lambda genome, rng: _repair_child_genome(
            representation,
            config.genome_length,
            mutation_fn(genome, rng),
        ),
        init_fn=representation.initialize,
        representation=representation,
    )


def build_algorithm_fn(config: GAConfig) -> AlgorithmFn:
    return build_algorithm_from_name(config.algorithm)


def build_runtime_context(config: GAConfig) -> RuntimeContext:
    problem = build_problem_instance(config)
    metadata = problem_metadata(problem)
    _validate_problem_contract(config, metadata)
    _validate_operator_contract(config)
    operators = build_operator_bundle(config)
    algorithm_fn = build_algorithm_fn(config)
    return RuntimeContext(
        problem=problem,
        problem_metadata=metadata,
        operators=operators,
        algorithm_fn=algorithm_fn,
    )
