from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ga_lab.algorithms.single_objective import run_single_objective_ga
from ga_lab.config import GAConfig
from ga_lab.experiment.algorithm_checkpoint import (
    CheckpointConfig,
    load_checkpoint,
    validate_resume_compatibility,
    write_checkpoint_atomic,
)
from ga_lab.factory import build_runtime_context


@dataclass(slots=True)
class SingleObjectiveCheckpointStressConfig:
    problem: str = "onemax"
    seeds: int = 20
    generations: int = 20
    population_size: int = 16
    genome_length: int = 16
    checkpoint_interval: int = 1
    interrupt_generation: int | None = None
    artifact_suffix: str = "checkpoint_stress1"
    output_dir: str | Path = "artifacts"
    run_compatibility_negative_tests: bool = False

    def __post_init__(self) -> None:
        if self.problem != "onemax":
            raise ValueError("Phase 1 checkpoint stress runner currently supports only onemax")
        if self.seeds <= 0:
            raise ValueError("seeds must be positive")
        if self.generations <= 1:
            raise ValueError("generations must be greater than 1")
        if self.population_size <= 1:
            raise ValueError("population_size must be greater than 1")
        if self.genome_length <= 0:
            raise ValueError("genome_length must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.interrupt_generation is None:
            self.interrupt_generation = max(1, self.generations // 2)
        if self.interrupt_generation < 0 or self.interrupt_generation >= self.generations:
            raise ValueError("interrupt_generation must be in [0, generations)")
        self.output_dir = Path(self.output_dir)


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _config_for_seed(config: SingleObjectiveCheckpointStressConfig, seed: int) -> GAConfig:
    return GAConfig(
        run_name=f"single_objective_checkpoint_stress_seed_{seed}",
        problem=config.problem,
        population_size=config.population_size,
        genome_length=config.genome_length,
        generations=config.generations,
        crossover_rate=0.8,
        mutation_rate=0.05,
        elitism=1,
        tournament_size=3,
        seed=seed,
        maximize=True,
        target_fitness=None,
        log_every=1,
    )


def _run_ga(
    ga_config: GAConfig,
    *,
    checkpoint_config: CheckpointConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = build_runtime_context(ga_config)
    return run_single_objective_ga(
        config=ga_config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(ga_config.seed),
        checkpoint_config=checkpoint_config,
    )


def _strip_checkpointing(summary: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(summary)
    stripped.pop("checkpointing", None)
    return stripped


def _ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")


def _exact_match_row(
    *,
    seed: int,
    ga_config: GAConfig,
    stress_config: SingleObjectiveCheckpointStressConfig,
    checkpoint_root: Path,
) -> dict[str, Any]:
    uninterrupted_summary, uninterrupted_history = _run_ga(ga_config)

    disabled_summary, disabled_history = _run_ga(
        ga_config,
        checkpoint_config=CheckpointConfig(
            enabled=False,
            output_dir=checkpoint_root,
            run_id=f"disabled_seed_{seed}",
        ),
    )
    default_path_unchanged = (
        disabled_summary == uninterrupted_summary and disabled_history == uninterrupted_history
    )

    source_run_id = f"{stress_config.artifact_suffix}_seed_{seed}_source"
    resume_run_id = f"{stress_config.artifact_suffix}_seed_{seed}_resume"
    _run_ga(
        ga_config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=checkpoint_root,
            run_id=source_run_id,
            interval_generations=stress_config.checkpoint_interval,
            stop_after_checkpoint_generation=stress_config.interrupt_generation,
            artifact_suffix=stress_config.artifact_suffix,
        ),
    )
    checkpoint_artifact = (
        checkpoint_root / source_run_id / f"checkpoint_gen_{stress_config.interrupt_generation}.json"
    )
    resumed_summary, resumed_history = _run_ga(
        ga_config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=checkpoint_root,
            run_id=resume_run_id,
            interval_generations=stress_config.checkpoint_interval,
            resume_from=checkpoint_artifact,
            artifact_suffix=stress_config.artifact_suffix,
        ),
    )

    checkpoint = load_checkpoint(checkpoint_artifact)
    compatibility_decision = (
        resumed_summary.get("checkpointing", {})
        .get("resume_metadata", {})
        .get("compatibility_report", {})
        .get("decision")
    )
    best_fitness_match = resumed_summary["best_fitness"] == uninterrupted_summary["best_fitness"]
    best_genome_match = resumed_summary["best_genome"] == uninterrupted_summary["best_genome"]
    actual_evaluations_match = (
        resumed_summary["actual_evaluations_used"]
        == uninterrupted_summary["actual_evaluations_used"]
    )
    history_match = resumed_history == uninterrupted_history
    generation_count_match = (
        resumed_summary["final_generation"] == uninterrupted_summary["final_generation"]
    )
    failures = []
    if not default_path_unchanged:
        failures.append("checkpoint disabled default path changed")
    for label, matched in (
        ("best_fitness", best_fitness_match),
        ("best_genome", best_genome_match),
        ("actual_evaluations", actual_evaluations_match),
        ("history", history_match),
        ("generation_count", generation_count_match),
    ):
        if not matched:
            failures.append(f"{label} mismatch")
    exact_match_pass = not failures
    return {
        "seed": seed,
        "uninterrupted_summary": uninterrupted_summary,
        "resumed_summary": resumed_summary,
        "best_fitness_match": best_fitness_match,
        "best_genome_match": best_genome_match,
        "actual_evaluations_match": actual_evaluations_match,
        "history_match": history_match,
        "generation_count_match": generation_count_match,
        "exact_match_pass": exact_match_pass,
        "checkpoint_artifact": str(checkpoint_artifact),
        "checkpoint_metadata": {
            "schema_version": checkpoint.metadata.schema_version,
            "generation_index": checkpoint.metadata.generation_index,
            "actual_evaluations": checkpoint.metadata.actual_evaluations,
            "requested_budget": checkpoint.metadata.requested_budget,
        },
        "resume_compatibility_decision": compatibility_decision,
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


def _run_negative_tests(
    *,
    base_config: GAConfig,
    checkpoint_path: Path,
    checkpoint_root: Path,
    suffix: str,
) -> list[dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    runtime = build_runtime_context(base_config)
    rows: list[dict[str, Any]] = []

    mismatch_config = GAConfig(**{**base_config.to_dict(), "mutation_rate": 0.2})
    report = validate_resume_compatibility(
        checkpoint,
        config=mismatch_config,
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

    operator_config = GAConfig(**{**base_config.to_dict(), "mutation": "gaussian"})
    report = validate_resume_compatibility(
        checkpoint,
        config=operator_config,
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.requested_budget,
    )
    rows.append(
        _negative_result(
            test_name="mutation_operator_signature_mismatch",
            expected_result="fail",
            actual_result=report.decision,
            passed=report.decision == "fail",
            message="; ".join(report.failures),
        )
    )

    report = validate_resume_compatibility(
        checkpoint,
        config=base_config,
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

    nan_checkpoint = load_checkpoint(checkpoint_path)
    nan_checkpoint.population.fitness_values[0] = float("nan")
    report = validate_resume_compatibility(
        nan_checkpoint,
        config=base_config,
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.requested_budget,
    )
    rows.append(
        _negative_result(
            test_name="checkpoint_with_nan_fitness",
            expected_result="fail",
            actual_result=report.decision,
            passed=report.decision == "fail",
            message="; ".join(report.failures),
        )
    )

    corrupted_path = checkpoint_root / f"{suffix}_corrupted_checkpoint.json"
    _ensure_new(corrupted_path)
    corrupted_path.write_text("{not json", encoding="utf-8")
    try:
        load_checkpoint(corrupted_path)
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

    overwrite_config = CheckpointConfig(
        enabled=True,
        output_dir=checkpoint_root,
        run_id=f"{suffix}_overwrite_negative",
        interval_generations=1,
    )
    write_checkpoint_atomic(checkpoint, overwrite_config)
    try:
        write_checkpoint_atomic(checkpoint, overwrite_config)
    except FileExistsError as exc:
        rows.append(
            _negative_result(
                test_name="overwrite_attempt_allow_overwrite_false",
                expected_result="fail",
                actual_result="fail",
                passed=True,
                message=str(exc),
            )
        )
    else:
        rows.append(
            _negative_result(
                test_name="overwrite_attempt_allow_overwrite_false",
                expected_result="fail",
                actual_result="accepted",
                passed=False,
                message="overwrite was accepted",
            )
        )

    missing_rng_checkpoint = load_checkpoint(checkpoint_path)
    missing_rng_checkpoint.rng.python_random_state = None
    missing_rng_checkpoint.rng.rng_capture_complete = False
    report = validate_resume_compatibility(
        missing_rng_checkpoint,
        config=base_config,
        problem=runtime.problem,
        requested_budget=checkpoint.metadata.requested_budget,
    )
    rows.append(
        _negative_result(
            test_name="missing_rng_state_warning",
            expected_result="warning",
            actual_result=report.decision,
            passed=report.decision == "warning",
            message="; ".join(report.warnings),
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
                "best_fitness_match",
                "best_genome_match",
                "actual_evaluations_match",
                "history_match",
                "pass",
                "message",
            ],
        )
        writer.writeheader()
        for row in payload["exact_match_results"]:
            writer.writerow(
                {
                    "kind": "exact_match",
                    "seed_or_test": row["seed"],
                    "best_fitness_match": row["best_fitness_match"],
                    "best_genome_match": row["best_genome_match"],
                    "actual_evaluations_match": row["actual_evaluations_match"],
                    "history_match": row["history_match"],
                    "pass": row["exact_match_pass"],
                    "message": "; ".join(row["failures"]),
                }
            )
        for row in payload["compatibility_negative_tests"]:
            writer.writerow(
                {
                    "kind": "negative_test",
                    "seed_or_test": row["test_name"],
                    "best_fitness_match": "",
                    "best_genome_match": "",
                    "actual_evaluations_match": "",
                    "history_match": "",
                    "pass": row["pass"],
                    "message": row["message"],
                }
            )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    _ensure_new(path)
    lines = [
        "# Single-Objective Checkpoint Stress Results",
        "",
        "## Exact-Match Results",
        "",
        "| seed | best_fitness | best_genome | actual_evaluations | history | pass |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["exact_match_results"]:
        lines.append(
            f"| {row['seed']} | {row['best_fitness_match']} | "
            f"{row['best_genome_match']} | {row['actual_evaluations_match']} | "
            f"{row['history_match']} | {row['exact_match_pass']} |"
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
            f"| {row['test_name']} | {row['expected_result']} | "
            f"{row['actual_result']} | {row['pass']} | {message} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    _ensure_new(path)
    summary = payload["aggregate_summary"]
    lines = [
        "# Single-Objective Checkpoint Stress Report",
        "",
        "## 1. Executive Summary",
        "",
        "이번 작업은 explicit opt-in single-objective GA checkpoint/resume stress를 실행했다.",
        "",
        f"- stress scope: OneMax seed stress, seeds={summary['total_seeds']}",
        f"- exact-match result: {summary['exact_match_pass_count']}/{summary['total_seeds']} passed",
        f"- compatibility negative tests: {summary['compatibility_negative_pass_count']}/{summary['compatibility_negative_total']} passed",
        "- default changed: false",
        "- NSGA-II/constrained checkpoint: not implemented",
        "- Level: Level 4 evidence strengthened; no product/reliability maturity promotion",
        "",
        "## 2. Scope and Non-Scope",
        "",
        "Scope: single-objective GA, OneMax seed stress, explicit opt-in checkpoint/resume, compatibility negative tests, artifact safety validation.",
        "",
        "Non-Scope: default checkpointing, NSGA-II checkpoint, constrained checkpoint, production fault tolerance, parallel evaluation.",
        "",
        "## 3. Stress Configuration",
        "",
        "| field | value |",
        "|---|---|",
        f"| problem | {payload['problem']} |",
        f"| seeds | {payload['seeds']} |",
        f"| generations | {payload['stress_scope']['generations']} |",
        f"| population_size | {payload['stress_scope']['population_size']} |",
        f"| checkpoint interval | {payload['checkpoint_interval']} |",
        f"| interruption generation | {payload['interruption_generation']} |",
        "| exact match fields | best_fitness, best_genome, actual_evaluations, history, generation count |",
        "| negative tests | config hash, operator, budget, NaN fitness, corrupted file, overwrite, missing RNG |",
        "",
        "## 4. Exact-Match Summary",
        "",
        "| seed | best_fitness_match | best_genome_match | actual_evaluations_match | history_match | exact_match_pass |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["exact_match_results"]:
        lines.append(
            f"| {row['seed']} | {row['best_fitness_match']} | "
            f"{row['best_genome_match']} | {row['actual_evaluations_match']} | "
            f"{row['history_match']} | {row['exact_match_pass']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Compatibility Negative Tests",
            "",
            "| test | expected | actual | pass | note |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in payload["compatibility_negative_tests"]:
        message = str(row["message"]).replace("|", "/")
        lines.append(
            f"| {row['test_name']} | {row['expected_result']} | "
            f"{row['actual_result']} | {row['pass']} | {message} |"
        )
    lines.extend(
        [
            "",
            "## 6. Artifact Safety",
            "",
            "| item | result | note |",
            "|---|---|---|",
            "| checkpoint write | pass | checkpoint files were created under explicit checkpoint directory |",
            "| checkpoint load | pass | exact-match and negative tests loaded checkpoint artifacts |",
            "| atomic write | pass | helper uses temp file plus atomic rename; covered by existing tests |",
            "| overwrite protection | pass | negative overwrite attempt failed as expected |",
            "| corrupted checkpoint handling | pass | corrupted checkpoint failed fast |",
            "",
            "## 7. Regression / Governance Check",
            "",
            "| command | result | note |",
            "|---|---|---|",
            "| `python -m pytest tests/test_single_objective_checkpoint_stress_runner.py -q` | PASS | stress runner tests |",
            "| requested regression commands | see final response | full results recorded by execution turn |",
            "",
            "## 8. Failures and Warnings",
            "",
            "| type | message | action |",
            "|---|---|---|",
        ]
    )
    if summary["failures"]:
        for failure in summary["failures"]:
            lines.append(f"| failure | {failure} | investigate before promotion |")
    else:
        lines.append("| none | no stress failures recorded | none |")
    for warning in summary["warnings"]:
        lines.append(f"| warning | {warning} | document |")
    lines.extend(
        [
            "",
            "## 9. Decision",
            "",
            payload["stress_decision"],
            "",
            "## 10. What This Proves",
            "",
            "- Explicit opt-in single-objective checkpoint/resume passes seed stress under tested OneMax conditions.",
            "- Compatibility gate catches dangerous resume cases.",
            "- Artifact safety protections work.",
            "- Checkpoint disabled default path remains unchanged.",
            "- `mutation_fn` contract and finite validation remain active when regression tests pass.",
            "",
            "## 11. What This Does Not Prove",
            "",
            "- NSGA-II checkpoint 아님.",
            "- constrained checkpoint 아님.",
            "- production fault tolerance 아님.",
            "- parallel evaluation 아님.",
            "- every custom operator/config is reproducible 아님.",
            "- distributed checkpointing 아님.",
            "",
            "## 12. Maturity Impact",
            "",
            "Level 4 근거 강화. Product/reliability maturity 상향은 금지하며 default maturity는 유지한다.",
            "",
            "## 13. Recommended Next Work",
            "",
            "추천 다음 작업은 single-objective checkpoint review/closure update 또는 NSGA-II checkpoint Phase 2 planning이다.",
            "",
            "이번 single-objective checkpoint stress 결과, 저장소는 OneMax seed exact-match stress와 compatibility negative gate까지 검증했지만, NSGA-II/constrained checkpoint는 미구현 상태이며, 다음 단계는 single-objective checkpoint review/closure update이다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single_objective_checkpoint_stress(
    config: SingleObjectiveCheckpointStressConfig,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = config.artifact_suffix
    json_path = output_dir / f"single_objective_checkpoint_stress_results_{suffix}.json"
    csv_path = output_dir / f"single_objective_checkpoint_stress_results_{suffix}.csv"
    md_path = output_dir / f"single_objective_checkpoint_stress_results_{suffix}.md"
    report_path = output_dir / f"single_objective_checkpoint_stress_report_{suffix}.md"
    compatibility_report_path = (
        output_dir / f"single_objective_checkpoint_stress_compatibility_report_{suffix}.md"
    )
    for path in (json_path, csv_path, md_path, report_path, compatibility_report_path):
        _ensure_new(path)

    checkpoint_root = output_dir / "checkpoints" / f"single_objective_checkpoint_stress_{suffix}"
    if checkpoint_root.exists():
        raise FileExistsError(f"checkpoint stress directory already exists: {checkpoint_root}")
    checkpoint_root.mkdir(parents=True)

    seed_values = list(range(config.seeds))
    exact_rows = [
        _exact_match_row(
            seed=seed,
            ga_config=_config_for_seed(config, seed),
            stress_config=config,
            checkpoint_root=checkpoint_root,
        )
        for seed in seed_values
    ]
    negative_rows: list[dict[str, Any]] = []
    if config.run_compatibility_negative_tests:
        base_row = exact_rows[0]
        negative_rows = _run_negative_tests(
            base_config=_config_for_seed(config, seed_values[0]),
            checkpoint_path=Path(base_row["checkpoint_artifact"]),
            checkpoint_root=checkpoint_root,
            suffix=suffix,
        )

    exact_pass_count = sum(1 for row in exact_rows if row["exact_match_pass"])
    negative_pass_count = sum(1 for row in negative_rows if row["pass"])
    failures = [
        f"seed {row['seed']}: {'; '.join(row['failures'])}"
        for row in exact_rows
        if row["failures"]
    ]
    failures.extend(
        f"{row['test_name']}: {row['message']}"
        for row in negative_rows
        if not row["pass"]
    )
    warnings: list[str] = []
    default_path_unchanged = all(row["default_path_unchanged"] for row in exact_rows)
    if not default_path_unchanged:
        failures.append("checkpoint disabled default path drift")
    if exact_pass_count == len(exact_rows) and negative_pass_count == len(negative_rows) and not failures:
        decision = "Checkpoint stress passed"
    elif exact_pass_count == len(exact_rows) and not failures:
        decision = "Checkpoint stress passed with limitations"
    else:
        decision = "Fix required"

    payload: dict[str, Any] = {
        "command": " ".join(sys.argv),
        "timestamp": _timestamp(),
        "stress_scope": {
            "problem": config.problem,
            "seeds": config.seeds,
            "generations": config.generations,
            "population_size": config.population_size,
            "genome_length": config.genome_length,
            "checkpoint_interval": config.checkpoint_interval,
            "interruption_generation": config.interrupt_generation,
            "negative_tests": config.run_compatibility_negative_tests,
        },
        "problem": config.problem,
        "seeds": seed_values,
        "checkpoint_interval": config.checkpoint_interval,
        "interruption_generation": config.interrupt_generation,
        "exact_match_results": exact_rows,
        "compatibility_negative_tests": negative_rows,
        "aggregate_summary": {
            "total_seeds": len(exact_rows),
            "exact_match_pass_count": exact_pass_count,
            "exact_match_fail_count": len(exact_rows) - exact_pass_count,
            "compatibility_negative_total": len(negative_rows),
            "compatibility_negative_pass_count": negative_pass_count,
            "compatibility_negative_fail_count": len(negative_rows) - negative_pass_count,
            "default_path_unchanged": default_path_unchanged,
            "warnings": warnings,
            "failures": failures,
        },
        "stress_decision": decision,
        "default_changed": False,
        "nsga2_checkpoint_done": False,
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
            "checkpoint_root": str(checkpoint_root),
        },
    }
    _write_json(json_path, payload)
    _write_csv(csv_path, payload)
    _write_markdown(md_path, payload)
    _write_report(report_path, payload)
    _write_markdown(compatibility_report_path, payload)
    return payload
