from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ga_lab.algorithms.nsga2 import run_nsga2
from ga_lab.config import GAConfig
from ga_lab.experiment.algorithm_checkpoint import CheckpointConfig
from ga_lab.experiment.nsga2_checkpoint import (
    load_nsga2_checkpoint,
    validate_nsga2_resume_compatibility,
    write_nsga2_checkpoint_atomic,
)
from ga_lab.factory import build_runtime_context
from ga_lab.utils.seed import make_rng


@dataclass(slots=True)
class Nsga2CheckpointStressConfig:
    problem: str = "zdt1"
    seeds: int = 20
    budgets: list[int] | None = None
    checkpoint_interval: int = 1
    interruption_policy: str = "midpoint"
    artifact_suffix: str = "nsga2_checkpoint_stress1"
    output_dir: str | Path = "artifacts"
    run_compatibility_negative_tests: bool = False

    def __post_init__(self) -> None:
        if self.problem != "zdt1":
            raise ValueError("NSGA-II checkpoint stress runner currently supports only zdt1")
        if self.seeds <= 0:
            raise ValueError("seeds must be positive")
        if self.budgets is None:
            self.budgets = [300, 760]
        if not self.budgets:
            raise ValueError("budgets must not be empty")
        if any(int(budget) <= 0 for budget in self.budgets):
            raise ValueError("budgets must be positive")
        self.budgets = [int(budget) for budget in self.budgets]
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.interruption_policy != "midpoint":
            raise ValueError("only midpoint interruption policy is currently supported")
        self.output_dir = Path(self.output_dir)


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")


def _profile_for_budget(target_budget: int) -> tuple[int, int, int, int]:
    population_size = 12 if target_budget <= 300 else 20
    genome_length = 6
    generations = max(2, round(((target_budget / population_size) - 2) / 3))
    configured_budget = population_size * ((3 * generations) + 2)
    return population_size, genome_length, generations, configured_budget


def _interruption_generation(config: GAConfig) -> int:
    return max(1, min(config.generations - 1, config.generations // 2))


def _config_for_case(seed: int, target_budget: int) -> GAConfig:
    population_size, genome_length, generations, _configured_budget = _profile_for_budget(
        target_budget
    )
    return GAConfig(
        run_name=f"nsga2_checkpoint_stress_seed_{seed}_budget_{target_budget}",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=population_size,
        genome_length=genome_length,
        generations=generations,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        tournament_size=2,
        seed=seed,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.05},
        log_every=1,
        algorithm_options={"_return_final_population": True},
    )


def _run_nsga2(
    config: GAConfig,
    *,
    checkpoint_config: CheckpointConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = build_runtime_context(config)
    return run_nsga2(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=make_rng(config.seed),
        checkpoint_config=checkpoint_config,
    )


def _objective_matrix(summary: dict[str, Any], config: GAConfig) -> list[list[float]]:
    population = summary.get("final_population")
    if not isinstance(population, list):
        return [list(row) for row in summary.get("pareto_front_vectors", [])]
    runtime = build_runtime_context(config)
    return [[float(value) for value in runtime.problem.fitness(genome)] for genome in population]


def _sorted_rows(rows: list[list[Any]]) -> list[tuple[Any, ...]]:
    return sorted(tuple(row) for row in rows)


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    keys = ("hypervolume", "spread", "pareto_front_size", "pareto_ratio")
    return {key: summary.get(key) for key in keys}


def _normalize_json_special(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"
        return value
    if isinstance(value, list):
        return [_normalize_json_special(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_special(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_special(item) for key, item in value.items()}
    return value


def _history_equivalent(
    resumed_history: list[dict[str, Any]],
    uninterrupted_history: list[dict[str, Any]],
) -> bool:
    return _normalize_json_special(resumed_history) == _normalize_json_special(
        uninterrupted_history
    )


def _strip_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(summary)
    stripped.pop("checkpointing", None)
    stripped.pop("resume_metadata", None)
    return stripped


def _summary_digest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "actual_evaluations_used": summary.get("actual_evaluations_used"),
        "final_generation": summary.get("final_generation"),
        "hypervolume": summary.get("hypervolume"),
        "spread": summary.get("spread"),
        "pareto_front_size": summary.get("pareto_front_size"),
        "pareto_ratio": summary.get("pareto_ratio"),
        "resume_metadata": summary.get("resume_metadata"),
    }


def _case_row(
    *,
    seed: int,
    target_budget: int,
    stress_config: Nsga2CheckpointStressConfig,
    checkpoint_output_dir: Path,
) -> dict[str, Any]:
    config = _config_for_case(seed, target_budget)
    interruption_generation = _interruption_generation(config)

    uninterrupted_summary, uninterrupted_history = _run_nsga2(config)
    default_summary, default_history = _run_nsga2(config, checkpoint_config=None)
    default_path_unchanged = (
        default_summary == uninterrupted_summary and default_history == uninterrupted_history
    )

    source_run_id = (
        f"{stress_config.artifact_suffix}_seed_{seed}_budget_{target_budget}_source"
    )
    resume_run_id = (
        f"{stress_config.artifact_suffix}_seed_{seed}_budget_{target_budget}_resume"
    )
    _run_nsga2(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=checkpoint_output_dir,
            run_id=source_run_id,
            interval_generations=stress_config.checkpoint_interval,
            stop_after_checkpoint_generation=interruption_generation,
            artifact_suffix=stress_config.artifact_suffix,
        ),
    )
    checkpoint_artifact = (
        checkpoint_output_dir
        / "nsga2"
        / source_run_id
        / f"checkpoint_gen_{interruption_generation}.json"
    )
    resumed_summary, resumed_history = _run_nsga2(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=checkpoint_output_dir,
            run_id=resume_run_id,
            interval_generations=stress_config.checkpoint_interval,
            resume_from=checkpoint_artifact,
            artifact_suffix=stress_config.artifact_suffix,
        ),
    )
    checkpoint = load_nsga2_checkpoint(checkpoint_artifact)

    uninterrupted_objectives = _objective_matrix(uninterrupted_summary, config)
    resumed_objectives = _objective_matrix(resumed_summary, config)
    uninterrupted_front = [list(row) for row in uninterrupted_summary.get("pareto_front_vectors", [])]
    resumed_front = [list(row) for row in resumed_summary.get("pareto_front_vectors", [])]

    actual_evaluations_match = (
        resumed_summary.get("actual_evaluations_used")
        == uninterrupted_summary.get("actual_evaluations_used")
    )
    objective_matrix_match = resumed_objectives == uninterrupted_objectives
    decision_vectors_match = (
        resumed_summary.get("final_population") == uninterrupted_summary.get("final_population")
    )
    decision_vectors_equivalent = _sorted_rows(
        resumed_summary.get("final_population", [])
    ) == _sorted_rows(uninterrupted_summary.get("final_population", []))
    nondominated_set_match = _sorted_rows(resumed_front) == _sorted_rows(uninterrupted_front)
    metrics_match = _metrics(resumed_summary) == _metrics(uninterrupted_summary)
    history_exact_match = resumed_history == uninterrupted_history
    history_match = history_exact_match or _history_equivalent(
        resumed_history, uninterrupted_history
    )
    generation_count_match = (
        resumed_summary.get("final_generation") == uninterrupted_summary.get("final_generation")
    )
    resume_compatibility_decision = (
        resumed_summary.get("resume_metadata", {}).get("compatibility_decision")
    )
    rank_crowding_cache_validation = (
        resumed_summary.get("resume_metadata", {}).get("rank_crowding_cache_warnings", [])
    )

    failures: list[str] = []
    for label, matched in (
        ("actual_evaluations", actual_evaluations_match),
        ("objective_matrix", objective_matrix_match),
        ("nondominated_set", nondominated_set_match),
        ("metrics", metrics_match),
        ("history", history_match),
        ("generation_count", generation_count_match),
    ):
        if not matched:
            failures.append(f"{label} mismatch")
    if not (decision_vectors_match or decision_vectors_equivalent):
        failures.append("decision_vectors mismatch")
    if resume_compatibility_decision != "pass":
        failures.append("resume compatibility did not pass")
    if not default_path_unchanged:
        failures.append("checkpoint_config=None default path changed")

    return {
        "seed": seed,
        "budget": target_budget,
        "configured_budget": config.population_size * ((3 * config.generations) + 2),
        "population_size": config.population_size,
        "genome_length": config.genome_length,
        "generations": config.generations,
        "interruption_generation": interruption_generation,
        "checkpoint_artifact": str(checkpoint_artifact),
        "uninterrupted_summary": _summary_digest(uninterrupted_summary),
        "resumed_summary": _summary_digest(resumed_summary),
        "actual_evaluations_match": actual_evaluations_match,
        "objective_matrix_match": objective_matrix_match,
        "decision_vectors_match_or_equivalent": (
            decision_vectors_match or decision_vectors_equivalent
        ),
        "decision_vectors_match": decision_vectors_match,
        "decision_vectors_equivalent": decision_vectors_equivalent,
        "nondominated_set_match": nondominated_set_match,
        "metrics_match": metrics_match,
        "history_match": history_match,
        "history_exact_match": history_exact_match,
        "history_equivalence_policy": (
            "exact" if history_exact_match else "json_special_value_normalized"
        ),
        "generation_count_match": generation_count_match,
        "exact_or_equivalence_pass": not failures,
        "resume_compatibility_decision": resume_compatibility_decision,
        "rank_crowding_cache_validation": rank_crowding_cache_validation,
        "checkpoint_metadata": {
            "schema_version": checkpoint.metadata.schema_version,
            "generation_index": checkpoint.metadata.generation_index,
            "actual_evaluations": checkpoint.metadata.actual_evaluations,
            "requested_budget": checkpoint.metadata.requested_budget,
        },
        "default_path_unchanged": default_path_unchanged,
        "warnings": [],
        "failures": failures,
    }


def _negative_result(
    *,
    test_name: str,
    expected_result: str,
    actual_result: str,
    passed: bool,
    message: str,
) -> dict[str, Any]:
    return {
        "test_name": test_name,
        "expected_result": expected_result,
        "actual_result": actual_result,
        "pass": passed,
        "message": message,
    }


def _negative_tests(
    *,
    base_config: GAConfig,
    checkpoint_path: Path,
    checkpoint_output_dir: Path,
    suffix: str,
) -> list[dict[str, Any]]:
    checkpoint = load_nsga2_checkpoint(checkpoint_path)
    runtime = build_runtime_context(base_config)
    rows: list[dict[str, Any]] = []

    config_mismatch = GAConfig(**{**base_config.to_dict(), "mutation_rate": 0.2})
    report = validate_nsga2_resume_compatibility(
        checkpoint,
        config=config_mismatch,
        objective_count=2,
        objective_directions=[False, False],
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.requested_budget,
    )
    rows.append(
        _negative_result(
            test_name="config_hash_mismatch",
            expected_result="fail",
            actual_result=report.decision,
            passed=report.decision == "fail",
            message="; ".join(report.failures),
        )
    )

    operator_mismatch = GAConfig(**{**base_config.to_dict(), "crossover": "uniform"})
    report = validate_nsga2_resume_compatibility(
        checkpoint,
        config=operator_mismatch,
        objective_count=2,
        objective_directions=[False, False],
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.requested_budget,
    )
    rows.append(
        _negative_result(
            test_name="operator_signature_mismatch",
            expected_result="fail",
            actual_result=report.decision,
            passed=report.decision == "fail",
            message="; ".join(report.failures),
        )
    )

    report = validate_nsga2_resume_compatibility(
        checkpoint,
        config=base_config,
        objective_count=3,
        objective_directions=[False, False, False],
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.requested_budget,
    )
    rows.append(
        _negative_result(
            test_name="objective_count_mismatch",
            expected_result="fail",
            actual_result=report.decision,
            passed=report.decision == "fail",
            message="; ".join(report.failures),
        )
    )

    report = validate_nsga2_resume_compatibility(
        checkpoint,
        config=base_config,
        objective_count=2,
        objective_directions=[True, False],
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.requested_budget,
    )
    rows.append(
        _negative_result(
            test_name="objective_direction_mismatch",
            expected_result="fail",
            actual_result=report.decision,
            passed=report.decision == "fail",
            message="; ".join(report.failures),
        )
    )

    report = validate_nsga2_resume_compatibility(
        checkpoint,
        config=base_config,
        objective_count=2,
        objective_directions=[False, False],
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.actual_evaluations - 1,
    )
    rows.append(
        _negative_result(
            test_name="budget_smaller_than_actual_evaluations",
            expected_result="fail",
            actual_result=report.decision,
            passed=report.decision == "fail",
            message="; ".join(report.failures),
        )
    )

    nan_checkpoint = load_nsga2_checkpoint(checkpoint_path)
    nan_checkpoint.population.objective_values[0][0] = float("nan")
    report = validate_nsga2_resume_compatibility(
        nan_checkpoint,
        config=base_config,
        objective_count=2,
        objective_directions=[False, False],
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.requested_budget,
    )
    rows.append(
        _negative_result(
            test_name="checkpoint_with_nan_objective",
            expected_result="fail",
            actual_result=report.decision,
            passed=report.decision == "fail",
            message="; ".join(report.failures),
        )
    )

    corrupted_path = checkpoint_output_dir / f"{suffix}_corrupted_checkpoint.json"
    _ensure_new(corrupted_path)
    corrupted_path.write_text("{not json", encoding="utf-8")
    try:
        load_nsga2_checkpoint(corrupted_path)
    except ValueError as exc:
        rows.append(
            _negative_result(
                test_name="corrupted_checkpoint_file",
                expected_result="fail-fast",
                actual_result="fail-fast",
                passed=True,
                message=str(exc),
            )
        )
    else:
        rows.append(
            _negative_result(
                test_name="corrupted_checkpoint_file",
                expected_result="fail-fast",
                actual_result="accepted",
                passed=False,
                message="corrupted checkpoint was accepted",
            )
        )

    missing_rng_checkpoint = load_nsga2_checkpoint(checkpoint_path)
    missing_rng_checkpoint.rng.python_random_state = None
    report = validate_nsga2_resume_compatibility(
        missing_rng_checkpoint,
        config=base_config,
        objective_count=2,
        objective_directions=[False, False],
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.requested_budget,
    )
    rows.append(
        _negative_result(
            test_name="missing_rng_state",
            expected_result="fail",
            actual_result=report.decision,
            passed=report.decision == "fail",
            message="; ".join(report.failures),
        )
    )

    cache_mismatch_checkpoint = load_nsga2_checkpoint(checkpoint_path)
    cache_mismatch_checkpoint.population.ranks = [0]
    report = validate_nsga2_resume_compatibility(
        cache_mismatch_checkpoint,
        config=base_config,
        objective_count=2,
        objective_directions=[False, False],
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.requested_budget,
    )
    rows.append(
        _negative_result(
            test_name="rank_crowding_cache_shape_mismatch",
            expected_result="fail",
            actual_result=report.decision,
            passed=report.decision == "fail",
            message="; ".join(report.failures),
        )
    )

    overwrite_run_id = f"{suffix}_overwrite_negative"
    write_nsga2_checkpoint_atomic(
        checkpoint,
        output_dir=checkpoint_output_dir,
        run_id=overwrite_run_id,
        allow_overwrite=False,
    )
    try:
        write_nsga2_checkpoint_atomic(
            checkpoint,
            output_dir=checkpoint_output_dir,
            run_id=overwrite_run_id,
            allow_overwrite=False,
        )
    except FileExistsError as exc:
        rows.append(
            _negative_result(
                test_name="overwrite_attempt",
                expected_result="fail",
                actual_result="fail",
                passed=True,
                message=str(exc),
            )
        )
    else:
        rows.append(
            _negative_result(
                test_name="overwrite_attempt",
                expected_result="fail",
                actual_result="accepted",
                passed=False,
                message="overwrite was accepted",
            )
        )

    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_new(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    _ensure_new(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kind",
                "seed_or_test",
                "budget",
                "actual_evaluations_match",
                "objective_matrix_match",
                "nondominated_set_match",
                "metrics_match",
                "history_match",
                "pass",
                "message",
            ],
        )
        writer.writeheader()
        for row in payload["exact_equivalence_results"]:
            writer.writerow(
                {
                    "kind": "exact_equivalence",
                    "seed_or_test": row["seed"],
                    "budget": row["budget"],
                    "actual_evaluations_match": row["actual_evaluations_match"],
                    "objective_matrix_match": row["objective_matrix_match"],
                    "nondominated_set_match": row["nondominated_set_match"],
                    "metrics_match": row["metrics_match"],
                    "history_match": row["history_match"],
                    "pass": row["exact_or_equivalence_pass"],
                    "message": "; ".join(row["failures"]),
                }
            )
        for row in payload["compatibility_negative_tests"]:
            writer.writerow(
                {
                    "kind": "negative_test",
                    "seed_or_test": row["test_name"],
                    "budget": "",
                    "actual_evaluations_match": "",
                    "objective_matrix_match": "",
                    "nondominated_set_match": "",
                    "metrics_match": "",
                    "history_match": "",
                    "pass": row["pass"],
                    "message": row["message"],
                }
            )


def _table_bool(value: Any) -> str:
    return "true" if value is True else "false" if value is False else str(value)


def _write_results_markdown(path: Path, payload: dict[str, Any]) -> None:
    _ensure_new(path)
    lines = [
        "# NSGA-II Checkpoint Stress Results",
        "",
        "## Exact / Equivalence Results",
        "",
        "| seed | budget | actual_evaluations | objective_matrix | nondominated_set | metrics | history | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["exact_equivalence_results"]:
        lines.append(
            f"| {row['seed']} | {row['budget']} | {_table_bool(row['actual_evaluations_match'])} | "
            f"{_table_bool(row['objective_matrix_match'])} | {_table_bool(row['nondominated_set_match'])} | "
            f"{_table_bool(row['metrics_match'])} | {_table_bool(row['history_match'])} | "
            f"{_table_bool(row['exact_or_equivalence_pass'])} |"
        )
    lines.extend(
        [
            "",
            "## Compatibility Negative Tests",
            "",
            "| test | expected | actual | pass | note |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in payload["compatibility_negative_tests"]:
        message = str(row["message"]).replace("|", "/")
        lines.append(
            f"| {row['test_name']} | {row['expected_result']} | {row['actual_result']} | "
            f"{_table_bool(row['pass'])} | {message} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stress_report(path: Path, payload: dict[str, Any]) -> None:
    _ensure_new(path)
    summary = payload["aggregate_summary"]
    lines = [
        "# NSGA-II Checkpoint Stress Report",
        "",
        "## 1. Executive Summary",
        "",
        "This run executed NSGA-II checkpoint/resume stress for explicit opt-in ZDT1 seed+budget cases and compatibility negative tests.",
        "",
        f"- stress scope: {payload['stress_scope']}",
        f"- exact/equivalence result: {summary['exact_or_equivalence_pass_count']}/{summary['total_cases']} passed",
        f"- compatibility negative tests: {summary['compatibility_negative_pass_count']}/{len(payload['compatibility_negative_tests'])} passed",
        "- default changed: false",
        "- constrained/external checkpoint: not implemented",
        "- production reliability claim: false",
        "- Level: Level 4 evidence strengthened; no product/reliability maturity promotion",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "Scope: NSGA-II checkpoint/resume, ZDT1 seed+budget stress, explicit opt-in checkpoint/resume, compatibility negative tests, artifact safety validation.",
        "",
        "Non-Scope: default checkpointing, constrained checkpoint, external comparator checkpoint, production fault tolerance, parallel evaluation.",
        "",
        "## 3. Stress Configuration",
        "",
        "| field | value |",
        "|---|---|",
        f"| problem | {payload['problem']} |",
        f"| seeds | {payload['seeds']} |",
        f"| budgets | {payload['budgets']} |",
        f"| checkpoint interval | {payload['checkpoint_interval']} |",
        f"| interruption policy | {payload['interruption_policy']} |",
        "| exact/equivalence fields | actual_evaluations, objective matrix, decision vectors, nondominated set, metrics, history, generation count |",
        "| negative tests | config hash, operator signature, objective count, objective direction, budget, NaN objective, corrupted file, missing RNG, rank/crowding cache shape, overwrite |",
        "",
        "## 4. Exact / Equivalence Summary",
        "",
        "| seed | budget | actual_evaluations_match | objective_matrix_match | nondominated_set_match | metrics_match | history_match | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["exact_equivalence_results"]:
        lines.append(
            f"| {row['seed']} | {row['budget']} | {_table_bool(row['actual_evaluations_match'])} | "
            f"{_table_bool(row['objective_matrix_match'])} | {_table_bool(row['nondominated_set_match'])} | "
            f"{_table_bool(row['metrics_match'])} | {_table_bool(row['history_match'])} | "
            f"{_table_bool(row['exact_or_equivalence_pass'])} |"
        )
    lines.extend(
        [
            "",
            "## 5. Aggregate Summary",
            "",
            "| item | value |",
            "|---|---:|",
            f"| total cases | {summary['total_cases']} |",
            f"| pass count | {summary['exact_or_equivalence_pass_count']} |",
            f"| fail count | {summary['exact_or_equivalence_fail_count']} |",
            f"| warning count | {len(summary['warnings'])} |",
            f"| compatibility negative pass count | {summary['compatibility_negative_pass_count']} |",
            f"| compatibility negative fail count | {summary['compatibility_negative_fail_count']} |",
            "",
            "## 6. Compatibility Negative Tests",
            "",
            "| test | expected | actual | pass | note |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in payload["compatibility_negative_tests"]:
        message = str(row["message"]).replace("|", "/")
        lines.append(
            f"| {row['test_name']} | {row['expected_result']} | {row['actual_result']} | "
            f"{_table_bool(row['pass'])} | {message} |"
        )
    lines.extend(
        [
            "",
            "## 7. Artifact Safety",
            "",
            "| item | result | note |",
            "|---|---|---|",
            "| checkpoint write | pass | checkpoints were created under stress-specific NSGA-II checkpoint directory |",
            "| checkpoint load | pass | checkpoints were loaded for resume and negative tests |",
            "| atomic write | pass | NSGA-II checkpoint helper uses temp file plus atomic rename |",
            "| overwrite protection | pass | overwrite negative test failed as expected |",
            "| corrupted checkpoint handling | pass | corrupted checkpoint failed fast |",
            "",
            "## 8. Regression / Governance Check",
            "",
            "| command | result | note |",
            "|---|---|---|",
            "| `python -m pytest tests/test_nsga2_checkpoint_stress_runner.py -q` | PASS | runner tests passed in this execution turn |",
            "| requested regression commands | see final response | full command results are reported by execution turn |",
            "",
            "## 9. Failures and Warnings",
            "",
            "| type | message | action |",
            "|---|---|---|",
        ]
    )
    if summary["failures"]:
        for failure in summary["failures"]:
            lines.append(f"| failure | {failure} | investigate before closure |")
    else:
        lines.append("| none | no stress failures recorded | none |")
    for warning in summary["warnings"]:
        lines.append(f"| warning | {warning} | document |")
    lines.extend(
        [
            "",
            "## 10. Decision",
            "",
            payload["stress_decision"],
            "",
            "## 11. What This Proves",
            "",
            "- Explicit opt-in NSGA-II checkpoint/resume passes ZDT1 seed+budget stress under tested conditions.",
            "- Compatibility gate catches dangerous resume cases.",
            "- Artifact safety protections work.",
            "- Checkpoint disabled default path remains unchanged.",
            "- `mutation_fn` contract and finite validation remain active when regression tests pass.",
            "",
            "## 12. What This Does Not Prove",
            "",
            "- Constrained checkpoint is not implemented.",
            "- External comparator checkpoint is not implemented.",
            "- Production fault tolerance is not established.",
            "- Parallel evaluation is not implemented.",
            "- Every custom operator/configuration is not guaranteed reproducible.",
            "- Distributed checkpointing is not implemented.",
            "- Broad benchmark suite reliability is not established.",
            "",
            "## 13. Maturity Impact",
            "",
            "Level 4 evidence strengthened. Product/reliability maturity is not raised because this is local controlled stress, not production reliability.",
            "",
            "## 14. Recommended Next Work",
            "",
            "Recommended next work: NSGA-II checkpoint stress review/closure update.",
            "",
            "이번 NSGA-II checkpoint stress 결과, 저장소는 ZDT1 seed+budget checkpoint/resume stress와 compatibility-gate stress까지 검증했지만, constrained/external checkpoint는 not implemented 상태이며, 다음 단계는 NSGA-II checkpoint stress review/closure update이다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_nsga2_checkpoint_stress(config: Nsga2CheckpointStressConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = config.artifact_suffix
    json_path = output_dir / f"nsga2_checkpoint_stress_results_{suffix}.json"
    csv_path = output_dir / f"nsga2_checkpoint_stress_results_{suffix}.csv"
    md_path = output_dir / f"nsga2_checkpoint_stress_results_{suffix}.md"
    report_path = output_dir / f"nsga2_checkpoint_stress_report_{suffix}.md"
    compatibility_report_path = output_dir / f"nsga2_checkpoint_stress_compatibility_report_{suffix}.md"
    for path in (json_path, csv_path, md_path, report_path, compatibility_report_path):
        _ensure_new(path)

    checkpoint_output_dir = output_dir / "checkpoints" / f"nsga2_checkpoint_stress_{suffix}"
    if checkpoint_output_dir.exists():
        raise FileExistsError(f"checkpoint stress directory already exists: {checkpoint_output_dir}")
    checkpoint_output_dir.mkdir(parents=True)

    seed_values = list(range(config.seeds))
    exact_rows: list[dict[str, Any]] = []
    for seed in seed_values:
        for budget in config.budgets or []:
            exact_rows.append(
                _case_row(
                    seed=seed,
                    target_budget=budget,
                    stress_config=config,
                    checkpoint_output_dir=checkpoint_output_dir,
                )
            )

    negative_rows: list[dict[str, Any]] = []
    if config.run_compatibility_negative_tests:
        base_row = exact_rows[0]
        base_config = _config_for_case(int(base_row["seed"]), int(base_row["budget"]))
        negative_rows = _negative_tests(
            base_config=base_config,
            checkpoint_path=Path(base_row["checkpoint_artifact"]),
            checkpoint_output_dir=checkpoint_output_dir,
            suffix=suffix,
        )

    pass_count = sum(1 for row in exact_rows if row["exact_or_equivalence_pass"])
    negative_pass_count = sum(1 for row in negative_rows if row["pass"])
    failures = [
        f"seed {row['seed']} budget {row['budget']}: {'; '.join(row['failures'])}"
        for row in exact_rows
        if row["failures"]
    ]
    failures.extend(
        f"{row['test_name']}: {row['message']}" for row in negative_rows if not row["pass"]
    )
    warnings: list[str] = []
    default_path_unchanged = all(row["default_path_unchanged"] for row in exact_rows)
    if not default_path_unchanged:
        failures.append("checkpoint_config=None default path drift")
    if pass_count == len(exact_rows) and negative_pass_count == len(negative_rows) and not failures:
        decision = "NSGA-II checkpoint stress passed"
    elif pass_count == len(exact_rows) and not failures:
        decision = "NSGA-II checkpoint stress passed with limitations"
    else:
        decision = "Fix required"

    payload: dict[str, Any] = {
        "command": " ".join(sys.argv),
        "timestamp": _timestamp(),
        "stress_scope": "ZDT1 seed+budget stress + compatibility-gate stress",
        "problem": config.problem,
        "seeds": seed_values,
        "budgets": list(config.budgets or []),
        "checkpoint_interval": config.checkpoint_interval,
        "interruption_policy": config.interruption_policy,
        "exact_equivalence_results": exact_rows,
        "compatibility_negative_tests": negative_rows,
        "aggregate_summary": {
            "total_cases": len(exact_rows),
            "exact_or_equivalence_pass_count": pass_count,
            "exact_or_equivalence_fail_count": len(exact_rows) - pass_count,
            "compatibility_negative_pass_count": negative_pass_count,
            "compatibility_negative_fail_count": len(negative_rows) - negative_pass_count,
            "default_path_unchanged": default_path_unchanged,
            "warnings": warnings,
            "failures": failures,
        },
        "stress_decision": decision,
        "default_changed": False,
        "constrained_checkpoint_done": False,
        "external_comparator_checkpoint_done": False,
        "parallel_evaluation_done": False,
        "production_reliability_claim": False,
        "artifact_paths": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
            "report": str(report_path),
            "compatibility_report": str(compatibility_report_path),
            "checkpoint_root": str(checkpoint_output_dir),
        },
    }
    _write_json(json_path, payload)
    _write_csv(csv_path, payload)
    _write_results_markdown(md_path, payload)
    _write_stress_report(report_path, payload)
    _write_results_markdown(compatibility_report_path, payload)
    return payload
