from __future__ import annotations

import json
from pathlib import Path

from ga_lab.config import GAConfig, load_config
from ga_lab.runner import run_experiment
import pytest


def test_load_config_supports_nested_plugin_blocks(tmp_path) -> None:
    config_path = tmp_path / "nested_onemax.json"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "nested_onemax",
                "problem": {"name": "onemax"},
                "algorithm": {"name": "ga"},
                "representation": {"name": "bit"},
                "operators": {
                    "selection": {
                        "name": "tournament",
                        "options": {"tournament_size": 4},
                    },
                    "crossover": {"name": "one_point"},
                    "mutation": {"name": "bit_flip"},
                },
                "population_size": 24,
                "genome_length": 16,
                "generations": 20,
                "crossover_rate": 0.9,
                "mutation_rate": 0.05,
                "elitism": 1,
                "seed": 5,
                "maximize": True,
                "target_fitness": 16,
                "log_every": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.problem == "onemax"
    assert config.algorithm == "ga"
    assert config.representation == "bit"
    assert config.selection == "tournament"
    assert config.selection_options == {"tournament_size": 4}
    assert config.crossover == "one_point"
    assert config.mutation == "bit_flip"

    legacy = config.to_dict()
    assert legacy["algorithm"] == "ga"
    assert legacy["selection"] == "tournament"
    assert "operators" not in legacy

    canonical = config.to_canonical_dict()
    assert canonical["algorithm"]["name"] == "ga"
    assert canonical["problem"]["name"] == "onemax"
    assert canonical["operators"]["selection"]["name"] == "tournament"
    assert canonical["operators"]["selection"]["options"]["tournament_size"] == 4


def test_canonical_config_round_trip_runs_onemax(tmp_path) -> None:
    base_config = GAConfig(
        run_name="canonical_onemax",
        problem="onemax",
        population_size=40,
        genome_length=16,
        generations=40,
        crossover_rate=0.9,
        mutation_rate=0.02,
        elitism=2,
        tournament_size=3,
        seed=13,
        maximize=True,
        target_fitness=16,
        log_every=1,
    )
    config_path = tmp_path / "canonical_onemax.json"
    config_path.write_text(
        json.dumps(base_config.to_canonical_dict(), indent=2),
        encoding="utf-8",
    )

    config = load_config(config_path)
    result = run_experiment(config, output_root=tmp_path)

    assert result.summary["best_fitness"] == 16.0
    assert result.summary["stop_reason"] == "target_fitness_reached"


def test_legacy_onemax_config_keeps_default_plugin_structure() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "onemax_baseline.json")

    assert config.algorithm == "ga"
    assert config.representation == "bit"
    assert config.selection == "tournament"
    assert config.crossover == "one_point"
    assert config.mutation == "bit_flip"

    canonical = config.to_canonical_dict()
    assert canonical["algorithm"]["name"] == "ga"
    assert canonical["representation"]["name"] == "bit"
    assert canonical["operators"]["selection"]["name"] == "tournament"
    assert canonical["operators"]["selection"]["options"]["tournament_size"] == 3


def test_repository_nested_example_configs_load_with_expected_plugins() -> None:
    project_root = Path(__file__).resolve().parents[1]

    onemax_config = load_config(project_root / "configs" / "onemax_nested.json")
    assert onemax_config.problem == "onemax"
    assert onemax_config.algorithm == "ga"
    assert onemax_config.representation == "bit"
    assert onemax_config.selection_options == {"tournament_size": 3}

    zdt1_config = load_config(project_root / "configs" / "zdt1_nsga2_nested.json")
    assert zdt1_config.problem == "zdt1"
    assert zdt1_config.algorithm == "nsga2"
    assert zdt1_config.representation == "real"
    assert zdt1_config.representation_options == {"low": 0.0, "high": 1.0}
    assert zdt1_config.selection_options == {"tournament_size": 2}
    assert zdt1_config.algorithm_options == {"hypervolume_reference_point": [1.1, 11.0]}
    assert zdt1_config.objective_directions == [False, False]


def test_run_experiment_writes_legacy_and_canonical_config_artifacts(tmp_path) -> None:
    config = GAConfig(
        run_name="artifact_check",
        problem="onemax",
        population_size=24,
        genome_length=16,
        generations=24,
        crossover_rate=0.9,
        mutation_rate=0.02,
        elitism=1,
        tournament_size=4,
        seed=11,
        maximize=True,
        target_fitness=16,
        log_every=1,
    )

    result = run_experiment(config, output_root=tmp_path)
    legacy_artifact = json.loads((result.output_dir / "config.json").read_text(encoding="utf-8"))
    canonical_artifact = json.loads(
        (result.output_dir / "config.canonical.json").read_text(encoding="utf-8")
    )
    summary_artifact = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))

    assert legacy_artifact["algorithm"] == "ga"
    assert legacy_artifact["selection"] == "tournament"
    assert "operators" not in legacy_artifact

    assert canonical_artifact["problem"]["name"] == "onemax"
    assert canonical_artifact["algorithm"]["name"] == "ga"
    assert canonical_artifact["representation"]["name"] == "bit"
    assert canonical_artifact["operators"]["selection"]["name"] == "tournament"
    assert canonical_artifact["operators"]["selection"]["options"]["tournament_size"] == 4
    assert summary_artifact["summary_schema_version"] == 1
    assert summary_artifact["generations"] == 24


def test_config_aliases_are_normalized() -> None:
    config = GAConfig.from_dict(
        {
            "run_name": "alias_normalized",
            "problem": "onemax",
            "pop_size": 24,
            "genome_length": 12,
            "max_iter": 6,
            "pc": 0.9,
            "pm": 0.05,
            "elite_size": 1,
            "seed": 1,
        }
    )
    assert config.population_size == 24
    assert config.generations == 6
    assert config.crossover_rate == 0.9
    assert config.mutation_rate == 0.05
    assert config.elitism == 1


def test_config_supports_random_seed_alias() -> None:
    config = GAConfig.from_dict(
        {
            "run_name": "random_seed_alias",
            "problem": "onemax",
            "population_size": 12,
            "genome_length": 8,
            "generations": 2,
            "crossover_rate": 0.9,
            "mutation_rate": 0.02,
            "random_seed": 19,
        }
    )
    assert config.seed == 19


def test_config_allows_matching_seed_and_random_seed_aliases() -> None:
    config = GAConfig.from_dict(
        {
            "run_name": "seed_alias_match",
            "problem": "onemax",
            "seed": 7,
            "random_seed": 7,
            "population_size": 12,
            "genome_length": 8,
            "generations": 2,
            "crossover_rate": 0.9,
            "mutation_rate": 0.02,
        }
    )
    assert config.seed == 7


def test_config_rejects_conflicting_seed_and_random_seed_aliases() -> None:
    with pytest.raises(ValueError, match="Conflicting values for 'seed' and alias 'random_seed'"):
        GAConfig.from_dict(
            {
                "run_name": "seed_alias_conflict",
                "problem": "onemax",
                "seed": 7,
                "random_seed": 8,
                "population_size": 12,
                "genome_length": 8,
                "generations": 2,
                "crossover_rate": 0.9,
                "mutation_rate": 0.02,
            }
        )


def test_config_rejects_conflicting_aliases() -> None:
    with pytest.raises(ValueError, match="Conflicting values for 'population_size'"):
        GAConfig.from_dict(
            {
                "run_name": "alias_conflict",
                "problem": "onemax",
                "population_size": 20,
                "pop_size": 22,
                "genome_length": 8,
                "generations": 2,
                "crossover_rate": 0.9,
                "mutation_rate": 0.02,
            }
        )


def test_config_normalizes_representation_bounds_aliases() -> None:
    config = GAConfig.from_dict(
        {
            "run_name": "bounds_alias",
            "problem": "zdt1",
            "representation": "real",
            "population_size": 10,
            "genome_length": 5,
            "generations": 3,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "representation_options": {"lower_bound": -1.0, "upper_bound": 2.0},
        }
    )
    assert config.representation_options == {"low": -1.0, "high": 2.0}


def test_config_normalizes_bounds_vector_alias() -> None:
    config = GAConfig.from_dict(
        {
            "run_name": "bounds_vector_alias",
            "problem": "zdt1",
            "representation": "real",
            "population_size": 10,
            "genome_length": 5,
            "generations": 3,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "representation_options": {"bounds": [-2.0, 3.0]},
        }
    )
    assert config.representation_options == {"low": -2.0, "high": 3.0}


def test_config_invalid_rate_and_population_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="population_size must be greater than 1"):
        GAConfig.from_dict(
            {
                "run_name": "invalid_pop",
                "problem": "onemax",
                "population_size": 1,
                "genome_length": 8,
                "generations": 3,
                "crossover_rate": 0.9,
                "mutation_rate": 0.05,
            }
        )


def test_config_rejects_invalid_seed_types() -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        GAConfig.from_dict(
            {
                "run_name": "seed_bool",
                "problem": "onemax",
                "population_size": 12,
                "genome_length": 8,
                "generations": 2,
                "crossover_rate": 0.9,
                "mutation_rate": 0.02,
                "seed": True,
            }
        )
 
    with pytest.raises(ValueError, match="seed must be an integer"):
        GAConfig.from_dict(
            {
                "run_name": "seed_str",
                "problem": "onemax",
                "population_size": 12,
                "genome_length": 8,
                "generations": 2,
                "crossover_rate": 0.9,
                "mutation_rate": 0.02,
                "seed": "7",
            }
        )

    with pytest.raises(ValueError, match="seed must be an integer"):
        GAConfig.from_dict(
            {
                "run_name": "seed_none",
                "problem": "onemax",
                "population_size": 12,
                "genome_length": 8,
                "generations": 2,
                "crossover_rate": 0.9,
                "mutation_rate": 0.02,
                "seed": None,
            }
        )


def test_config_accepts_negative_seed_via_random_seed_alias() -> None:
    config = GAConfig.from_dict(
        {
            "run_name": "negative_seed",
            "problem": "onemax",
            "population_size": 12,
            "genome_length": 8,
            "generations": 2,
            "crossover_rate": 0.9,
            "mutation_rate": 0.02,
            "random_seed": -13,
        }
    )
    assert config.seed == -13


def test_config_rejects_invalid_random_seed_types() -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        GAConfig.from_dict(
            {
                "run_name": "random_seed_bool",
                "problem": "onemax",
                "population_size": 12,
                "genome_length": 8,
                "generations": 2,
                "crossover_rate": 0.9,
                "mutation_rate": 0.02,
                "random_seed": True,
            }
        )

    with pytest.raises(ValueError, match="seed must be an integer"):
        GAConfig.from_dict(
            {
                "run_name": "random_seed_str",
                "problem": "onemax",
                "population_size": 12,
                "genome_length": 8,
                "generations": 2,
                "crossover_rate": 0.9,
                "mutation_rate": 0.02,
                "random_seed": "7",
            }
        )
    with pytest.raises(ValueError, match="mutation_rate must be in \\[0, 1\\]"):
        GAConfig.from_dict(
            {
                "run_name": "invalid_mutation",
                "problem": "onemax",
                "population_size": 12,
                "genome_length": 8,
                "generations": 3,
                "crossover_rate": 0.9,
                "mutation_rate": 2.0,
            }
        )
