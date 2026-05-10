from __future__ import annotations

import csv
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from ga_lab.algorithms._shared import (
    resolve_algorithm_reference_point,
    resolve_objective_directions,
)
from ga_lab.api import run_config, run_demo, run_preset
from ga_lab.config import GAConfig, load_config
from ga_lab.convergence_diagnostics import configured_evaluation_budget
from ga_lab.experiment.suite import apply_overrides
from ga_lab.factory import build_runtime_context
from ga_lab.local_failure_trace import (
    build_failure_hypothesis_rows,
    build_failure_trace_rows,
)
from ga_lab.metrics import finite_or_none, front_metrics
from ga_lab.resources import read_builtin_json
from ga_lab.runner import RunResult, run_experiment
from ga_lab.utils.seed import make_rng

LOCAL_STUDY_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_STUDY_ROOT = PROJECT_ROOT / "configs" / "local_studies"

_DEMO_PROBLEMS = {
    "baseline": "onemax",
    "pure-ga": "onemax",
    "hybrid": "tsp",
    "nsga2": "zdt1",
}

_LOWER_IS_BETTER_METRICS = {
    "best_route_distance",
    "evaluations_to_target",
    "generations_to_target",
    "mean_violation",
    "spread",
    "runtime_seconds",
}

_RESOLVED_CONFIG_KEYS = (
    "algorithm",
    "representation",
    "selection",
    "crossover",
    "mutation",
    "population_size",
    "generations",
    "mutation_rate",
    "crossover_rate",
    "elitism",
    "tournament_size",
)

_RESOLVED_ALGORITHM_OPTION_KEYS = (
    "adaptive_policy",
    "seed_fraction",
    "init_strategy",
    "repair_strategy",
    "local_search_strategy",
    "early_stop_policy",
    "early_stop_window",
    "early_stop_min_generation",
    "early_stop_epsilon",
    "decay_end_rate",
    "diversity_threshold",
    "refresh_fraction",
    "adaptation_cooldown",
    "feasible_ratio_threshold",
    "violation_plateau_window",
    "mutation_boost_multiplier",
    "mutation_cap",
    "gate_pilot_generation_fraction",
    "gate_initial_feasible_threshold",
    "gate_first_feasible_generation_limit",
    "gate_mean_violation_threshold",
    "gate_fast_profile",
    "gate_canonical_profile",
    "gate_pilot_budget_fraction",
    "gate_signal_policy",
    "gate_min_gain_ratio",
    "gate_late_improvement_floor",
    "gate_diversity_threshold",
    "gate_min_hypervolume",
    "gate_front_size_floor",
    "gate_spread_ceiling",
    "portfolio_profile",
    "portfolio_restart_count",
    "portfolio_total_budget_factor",
    "portfolio_total_budget",
)

_STUDY_METADATA_KEYS = (
    "family_label",
    "capacity_ratio",
    "correlation_note",
)


@dataclass(frozen=True, slots=True)
class LocalStudyCase:
    case_id: str
    overrides: dict[str, Any]
    note: str = ""
    group: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "overrides": dict(self.overrides),
            "note": self.note,
            "group": self.group,
        }


@dataclass(frozen=True, slots=True)
class LocalStudy:
    study_name: str
    description: str
    problem: str
    base_preset: str | None
    base_config: str | None
    cases: tuple[LocalStudyCase, ...]
    shared_overrides: dict[str, Any]
    variant_overrides: dict[str, dict[str, Any]]
    sweep: dict[str, list[Any]]
    seeds: tuple[int, ...]
    budget_ceiling: int | None
    primary_metric: str
    plotting: dict[str, Any]
    analysis: dict[str, Any]
    runtime_budget_note: str
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cases"] = [case.to_dict() for case in self.cases]
        payload["seeds"] = list(self.seeds)
        return payload


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_name(value: str) -> str:
    normalized = [
        character.lower() if character.isalnum() else "_"
        for character in value.strip().replace("-", "_")
    ]
    collapsed = "".join(normalized).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed or "run"


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _resolve_study_path(study_ref: str | Path) -> Path:
    candidate = Path(study_ref)
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()

    study_name = candidate.name
    if not study_name.endswith(".json"):
        study_name = f"{study_name}.json"
    path = (LOCAL_STUDY_ROOT / study_name).resolve()
    if path.exists():
        return path
    raise FileNotFoundError(f"Local study not found: {study_ref}")


def load_local_study(study_ref: str | Path) -> LocalStudy:
    path = _resolve_study_path(study_ref)
    payload = _load_json_dict(path)

    study_name = payload.get("study_name", path.stem)
    description = payload.get("description", "")
    problem = payload.get("problem")
    base_preset = payload.get("base_preset")
    base_config = payload.get("base_config")
    cases = payload.get("cases", [])
    shared_overrides = payload.get("shared_overrides", {})
    variant_overrides = payload.get("variant_overrides", {})
    sweep = payload.get("sweep", {})
    seeds = payload.get("seeds", [])
    budget_ceiling = payload.get("budget_ceiling")
    primary_metric = payload.get("primary_metric")
    plotting = payload.get("plotting", {})
    analysis = payload.get("analysis", {})
    runtime_budget_note = payload.get("runtime_budget_note", "")

    if not isinstance(study_name, str) or not study_name.strip():
        raise ValueError("study_name must be a non-empty string")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string")
    if bool(base_preset) == bool(base_config):
        raise ValueError("Exactly one of base_preset or base_config must be set")
    if base_preset is not None and not isinstance(base_preset, str):
        raise ValueError("base_preset must be a string")
    if base_config is not None and not isinstance(base_config, str):
        raise ValueError("base_config must be a string")
    if cases and not isinstance(cases, list):
        raise ValueError("cases must be an array when provided")
    if not isinstance(shared_overrides, dict):
        raise ValueError("shared_overrides must be a JSON object")
    if not isinstance(variant_overrides, dict):
        raise ValueError("variant_overrides must be a JSON object")
    if not isinstance(sweep, dict) or not sweep:
        raise ValueError("sweep must be a non-empty JSON object")
    if len(sweep) > 3:
        raise ValueError("sweep supports at most three parameter axes")
    for key, values in sweep.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Each sweep axis key must be a non-empty string")
        if not isinstance(values, list) or not values:
            raise ValueError("Each sweep axis must be a non-empty array")
    normalized_variants: dict[str, dict[str, Any]] = {}
    for name, overrides in variant_overrides.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each variant_overrides key must be a non-empty string")
        if not isinstance(overrides, dict):
            raise ValueError("Each variant_overrides payload must be a JSON object")
        normalized_variants[name.strip()] = dict(overrides)
    if normalized_variants and "study_variant" not in sweep:
        raise ValueError("variant_overrides requires a study_variant sweep axis")
    if "study_variant" in sweep:
        if not normalized_variants:
            raise ValueError("study_variant sweep requires variant_overrides")
        for value in sweep["study_variant"]:
            key = str(value)
            if key not in normalized_variants:
                raise ValueError(f"study_variant '{key}' missing from variant_overrides")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty array")
    normalized_seeds = []
    for seed in seeds:
        if not isinstance(seed, int):
            raise ValueError("All seeds must be integers")
        normalized_seeds.append(seed)
    if not isinstance(primary_metric, str) or not primary_metric.strip():
        raise ValueError("primary_metric must be a non-empty string")
    if budget_ceiling is not None and not isinstance(budget_ceiling, int):
        raise ValueError("budget_ceiling must be an integer when provided")
    if not isinstance(plotting, dict):
        raise ValueError("plotting must be a JSON object")
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be a JSON object")
    if not isinstance(description, str):
        raise ValueError("description must be a string")
    if not isinstance(runtime_budget_note, str):
        raise ValueError("runtime_budget_note must be a string")

    normalized_cases: list[LocalStudyCase] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be a JSON object")
        case_id = case.get("case_id")
        overrides = case.get("overrides", {})
        note = case.get("note", "")
        group = case.get("group", "")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Each case_id must be a non-empty string")
        if not isinstance(overrides, dict):
            raise ValueError("Each case overrides payload must be a JSON object")
        if not isinstance(note, str):
            raise ValueError("Each case note must be a string")
        if not isinstance(group, str):
            raise ValueError("Each case group must be a string")
        normalized_cases.append(
            LocalStudyCase(
                case_id=case_id.strip(),
                overrides=dict(overrides),
                note=note.strip(),
                group=group.strip(),
            )
        )

    return LocalStudy(
        study_name=study_name.strip(),
        description=description.strip(),
        problem=problem.strip(),
        base_preset=base_preset.strip() if isinstance(base_preset, str) else None,
        base_config=base_config.strip() if isinstance(base_config, str) else None,
        cases=tuple(normalized_cases),
        shared_overrides=dict(shared_overrides),
        variant_overrides=normalized_variants,
        sweep={key: list(values) for key, values in sweep.items()},
        seeds=tuple(normalized_seeds),
        budget_ceiling=int(budget_ceiling) if isinstance(budget_ceiling, int) else None,
        primary_metric=primary_metric.strip(),
        plotting=dict(plotting),
        analysis=dict(analysis),
        runtime_budget_note=runtime_budget_note.strip(),
        source_path=str(path),
    )


def _base_config_data(study: LocalStudy) -> dict[str, Any]:
    if study.base_preset is not None:
        return read_builtin_json("preset", study.base_preset)
    config_path = Path(study.base_config or "")
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    return load_config(config_path).to_dict()


def _repair_knapsack_bits(problem, bits: list[int]) -> list[float]:
    selected_by_low_ratio = [
        idx
        for _, idx in sorted(
            (
                (problem.values[idx] / problem.weights[idx], idx)
                for idx, selected in enumerate(bits)
                if selected and problem.weights[idx] > 0
            )
        )
    ]
    total_weight = sum(problem.weights[idx] for idx, selected in enumerate(bits) if selected == 1)
    for idx in selected_by_low_ratio:
        if total_weight <= problem.capacity:
            break
        bits[idx] = 0
        total_weight -= problem.weights[idx]
    item_order = sorted(
        range(problem.num_items),
        key=lambda idx: problem.values[idx] / problem.weights[idx],
        reverse=True,
    )
    for idx in item_order:
        if bits[idx] == 1:
            continue
        candidate_weight = total_weight + problem.weights[idx]
        if candidate_weight <= problem.capacity:
            bits[idx] = 1
            total_weight = candidate_weight
    return [float(bit) for bit in bits]


def _knapsack_greedy_local_search_result(
    config: GAConfig,
    *,
    output_root: Path,
) -> RunResult:
    runtime = build_runtime_context(config)
    problem = runtime.problem
    rng = make_rng(config.seed)
    configured_budget = config.population_size * (config.generations + 2)
    output_dir = Path(output_root) / f"{_timestamp()}_{config.run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    item_order = sorted(
        range(problem.num_items),
        key=lambda idx: problem.values[idx] / problem.weights[idx],
        reverse=True,
    )
    current_bits = [0] * problem.num_items
    current_weight = 0.0
    for idx in item_order:
        candidate_weight = current_weight + problem.weights[idx]
        if candidate_weight <= problem.capacity:
            current_bits[idx] = 1
            current_weight = candidate_weight

    def evaluate_candidate(bits: list[int]) -> tuple[float, dict[str, float]]:
        genome = [float(bit) for bit in bits]
        selection = problem.evaluate_selection(bits)
        return float(problem.fitness(genome)), selection

    start = time.perf_counter()
    evaluation_count = 0
    feasible_evaluations = 0
    total_violation = 0.0
    generations_since_last_improvement = 0

    current_fitness, current_selection = evaluate_candidate(current_bits)
    evaluation_count += 1
    total_violation += float(current_selection["constraint_violation"])
    if current_selection["constraint_violation"] == 0.0:
        feasible_evaluations += 1
    best_feasible_fitness = (
        float(current_selection["total_value"])
        if current_selection["constraint_violation"] == 0.0
        else None
    )
    initial_history_row = {
        "generation": 0,
        "best_fitness": current_fitness,
        "best_total_value": float(current_selection["total_value"]),
        "best_constraint_violation": float(current_selection["constraint_violation"]),
        "best_constraint_violation_rate": float(current_selection["constraint_violation_rate"]),
        "best_is_feasible": current_selection["constraint_violation"] == 0.0,
        "feasible_ratio": feasible_evaluations / evaluation_count,
        "mean_constraint_violation": total_violation / evaluation_count,
        "adaptive_mutation_rate": config.mutation_rate,
        "generations_since_last_improvement": 0,
    }

    while evaluation_count < configured_budget:
        candidate_bits = current_bits[:]
        idx = rng.randrange(problem.num_items)
        candidate_bits[idx] = 1 - candidate_bits[idx]
        candidate_bits = [
            int(value)
            for value in (
                0 if gene <= 0 else 1 if gene >= 1 else int(round(gene))
                for gene in _repair_knapsack_bits(problem, candidate_bits)
            )
        ]
        candidate_fitness, candidate_selection = evaluate_candidate(candidate_bits)
        evaluation_count += 1
        violation = float(candidate_selection["constraint_violation"])
        total_violation += violation
        candidate_value = float(candidate_selection["total_value"])
        if violation == 0.0:
            feasible_evaluations += 1
            if best_feasible_fitness is None or candidate_value > best_feasible_fitness:
                best_feasible_fitness = candidate_value

        if candidate_fitness > current_fitness or (
            math.isclose(candidate_fitness, current_fitness)
            and candidate_value > float(current_selection["total_value"])
        ):
            current_bits = candidate_bits
            current_fitness = candidate_fitness
            current_selection = candidate_selection
            generations_since_last_improvement = 0
        else:
            generations_since_last_improvement += 1

    elapsed = time.perf_counter() - start
    feasible_rate = feasible_evaluations / evaluation_count if evaluation_count else None
    mean_violation = total_violation / evaluation_count if evaluation_count else None
    final_history_row = {
        "generation": config.generations,
        "best_fitness": current_fitness,
        "best_total_value": float(current_selection["total_value"]),
        "best_constraint_violation": float(current_selection["constraint_violation"]),
        "best_constraint_violation_rate": float(current_selection["constraint_violation_rate"]),
        "best_is_feasible": current_selection["constraint_violation"] == 0.0,
        "feasible_ratio": feasible_rate,
        "mean_constraint_violation": mean_violation,
        "adaptive_mutation_rate": config.mutation_rate,
        "generations_since_last_improvement": generations_since_last_improvement,
    }
    history = [initial_history_row, final_history_row]
    summary = {
        "summary_schema_version": 1,
        "run_name": config.run_name,
        "problem": config.problem,
        "seed": config.seed,
        "algorithm": "greedy_local_search",
        "selection": config.selection,
        "crossover": config.crossover,
        "mutation": config.mutation,
        "representation": config.representation,
        "population_size": config.population_size,
        "genome_length": config.genome_length,
        "generations": config.generations,
        "crossover_rate": config.crossover_rate,
        "mutation_rate": config.mutation_rate,
        "elitism": config.elitism,
        "tournament_size": config.tournament_size,
        "runtime_seconds": elapsed,
        "log_every": config.log_every,
        "best_fitness": current_fitness,
        "best_genome": [float(bit) for bit in current_bits],
        "configured_evaluation_budget": configured_budget,
        "actual_evaluations_used": evaluation_count,
        "extra_evaluations_from_adaptation": 0,
        "adaptive_policy": "none",
        "best_total_value": float(current_selection["total_value"]),
        "best_constraint_violation": float(current_selection["constraint_violation"]),
        "best_constraint_violation_rate": float(current_selection["constraint_violation_rate"]),
        "best_is_feasible": current_selection["constraint_violation"] == 0.0,
        "best_feasible_fitness": best_feasible_fitness,
        "feasible_rate": feasible_rate,
        "mean_violation": mean_violation,
        "final_generation": config.generations,
        "stop_reason": "max_generations",
    }
    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(output_dir / "summary.json", summary)
    _write_rows_csv(output_dir / "history.csv", history)
    return RunResult(summary=summary, history=history, output_dir=output_dir)


def _history_row_number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _knapsack_best_feasible(summary: dict[str, Any]) -> float | None:
    best_feasible = summary.get("best_feasible_fitness")
    if isinstance(best_feasible, int | float) and not isinstance(best_feasible, bool):
        return float(best_feasible)
    best_total_value = summary.get("best_total_value")
    best_is_feasible = summary.get("best_is_feasible")
    if (
        isinstance(best_total_value, int | float)
        and not isinstance(best_total_value, bool)
        and bool(best_is_feasible)
    ):
        return float(best_total_value)
    return None


def _profile_payload_for_context(
    profile_ref: str | Path,
    *,
    context_config: GAConfig,
    run_name: str,
) -> dict[str, Any]:
    profile_path = Path(profile_ref)
    if not profile_path.is_absolute():
        profile_path = (PROJECT_ROOT / profile_path).resolve()
    payload = load_config(profile_path).to_dict()
    context_payload = context_config.to_dict()
    payload["run_name"] = run_name
    payload["seed"] = context_payload.get("seed", payload.get("seed"))
    payload["problem_options"] = deepcopy(
        context_payload.get("problem_options", payload.get("problem_options", {}))
    )
    payload["genome_length"] = context_payload.get("genome_length", payload.get("genome_length"))
    payload["log_every"] = context_payload.get("log_every", payload.get("log_every"))
    return payload


def _generations_for_target_budget(config: GAConfig, target_budget: float) -> int:
    population_size = max(1, int(config.population_size))
    if config.algorithm == "nsga2":
        raw_generations = (float(target_budget) / float(population_size) - 2.0) / 3.0
    else:
        raw_generations = float(target_budget) / float(population_size) - 2.0
    return max(1, min(config.generations, int(math.floor(raw_generations))))


def _portfolio_seed(base_seed: int, restart_index: int) -> int:
    return int(base_seed) + restart_index * 1009


def _portfolio_total_budget(config: GAConfig, options: dict[str, Any]) -> tuple[int, float]:
    explicit_budget = options.get("portfolio_total_budget")
    if isinstance(explicit_budget, int | float) and not isinstance(explicit_budget, bool):
        target_budget = max(1, int(round(float(explicit_budget))))
        factor = target_budget / max(1.0, float(configured_evaluation_budget(config)))
        return target_budget, factor
    factor = options.get("portfolio_total_budget_factor", 1.0)
    if not isinstance(factor, int | float) or isinstance(factor, bool):
        factor = 1.0
    factor = max(0.1, float(factor))
    base_budget = configured_evaluation_budget(config)
    return max(1, int(round(float(base_budget) * factor))), factor


def _portfolio_template_payload(config: GAConfig, options: dict[str, Any], *, run_name: str) -> dict[str, Any]:
    profile_ref = options.get("portfolio_profile")
    if isinstance(profile_ref, str) and profile_ref.strip():
        payload = _profile_payload_for_context(
            profile_ref,
            context_config=config,
            run_name=run_name,
        )
    else:
        payload = config.to_dict()
        payload["run_name"] = run_name
    payload.setdefault("algorithm_options", {})
    if isinstance(payload["algorithm_options"], dict):
        payload["algorithm_options"] = dict(payload["algorithm_options"])
        payload["algorithm_options"].pop("_return_final_population", None)
        payload["algorithm_options"].pop("_initial_population", None)
    return payload


def _portfolio_member_config(
    template_payload: dict[str, Any],
    *,
    restart_index: int,
    target_budget: float,
    base_seed: int,
    run_name: str,
) -> GAConfig:
    payload = deepcopy(template_payload)
    payload["run_name"] = f"{run_name}_restart_{restart_index + 1}"
    payload["seed"] = _portfolio_seed(base_seed, restart_index)
    provisional = GAConfig.from_dict(payload)
    payload["generations"] = _generations_for_target_budget(provisional, target_budget)
    return GAConfig.from_dict(payload)


def _single_objective_portfolio_metric(problem: str, summary: dict[str, Any]) -> float | None:
    if problem == "tsp":
        value = summary.get("best_route_distance")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        best_fitness = summary.get("best_fitness")
        if isinstance(best_fitness, int | float) and not isinstance(best_fitness, bool):
            return float(-best_fitness)
        return None
    if problem == "knapsack":
        return _knapsack_best_feasible(summary)
    if problem == "onemax":
        target_hit = bool(summary.get("stop_reason") == "target_fitness_reached")
        evaluations_to_target = summary.get("evaluations_to_target")
        if target_hit and isinstance(evaluations_to_target, int | float) and not isinstance(
            evaluations_to_target,
            bool,
        ):
            return float(evaluations_to_target)
        best_fitness = summary.get("best_fitness")
        if isinstance(best_fitness, int | float) and not isinstance(best_fitness, bool):
            return float(-best_fitness)
        return None
    return None


def _portfolio_is_better(problem: str, candidate: float, incumbent: float) -> bool:
    if problem in {"tsp", "onemax"}:
        return float(candidate) < float(incumbent)
    return float(candidate) > float(incumbent)


def _single_objective_restart_portfolio_result(
    config: GAConfig,
    *,
    output_root: Path,
    problem: str,
) -> RunResult:
    output_dir = Path(output_root) / f"{_timestamp()}_{config.run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    options = dict(config.algorithm_options)
    restart_count = options.get("portfolio_restart_count", 2)
    if not isinstance(restart_count, int) or isinstance(restart_count, bool):
        restart_count = 2
    restart_count = max(2, restart_count)
    target_total_budget, total_budget_factor = _portfolio_total_budget(config, options)
    member_target_budget = max(1.0, float(target_total_budget) / float(restart_count))
    template_payload = _portfolio_template_payload(
        config,
        options,
        run_name=f"{config.run_name}_portfolio_template",
    )

    base_seed = int(config.seed)
    member_rows: list[dict[str, Any]] = []
    total_runtime = 0.0
    total_actual_evaluations = 0
    total_configured_budget = 0
    best_result: RunResult | None = None
    best_metric: float | None = None
    best_restart_index = 0

    for restart_index in range(restart_count):
        member_config = _portfolio_member_config(
            template_payload,
            restart_index=restart_index,
            target_budget=member_target_budget,
            base_seed=base_seed,
            run_name=config.run_name,
        )
        result = run_experiment(member_config, output_root=output_dir / "portfolio_runs")
        metric_value = _single_objective_portfolio_metric(problem, result.summary)
        configured_budget = configured_evaluation_budget(member_config)
        actual_evaluations = int(result.summary.get("actual_evaluations_used", 0) or 0)
        total_runtime += float(result.summary.get("runtime_seconds", 0.0) or 0.0)
        total_actual_evaluations += actual_evaluations
        total_configured_budget += configured_budget
        member_rows.append(
            {
                "restart_index": restart_index + 1,
                "seed": member_config.seed,
                "configured_budget": configured_budget,
                "actual_evaluations_used": actual_evaluations,
                "metric_value": metric_value,
                "output_dir": str(result.output_dir.resolve()),
            }
        )
        if metric_value is None:
            continue
        if best_metric is None or _portfolio_is_better(problem, metric_value, best_metric):
            best_metric = float(metric_value)
            best_result = result
            best_restart_index = restart_index

    if best_result is None:
        raise ValueError("restart portfolio did not produce a comparable result")

    chosen_history = list(best_result.history)
    chosen_summary = dict(best_result.summary)
    summary = dict(chosen_summary)
    summary["run_name"] = config.run_name
    summary["algorithm"] = f"{problem}_restart_portfolio"
    summary["configured_evaluation_budget"] = total_configured_budget
    summary["target_total_budget"] = target_total_budget
    summary["actual_evaluations_used"] = total_actual_evaluations
    summary["total_actual_evaluations_used"] = total_actual_evaluations
    summary["extra_evaluations_from_adaptation"] = 0
    summary["runtime_seconds"] = total_runtime
    summary["portfolio_mode"] = "best_of_k"
    summary["portfolio_restart_count"] = restart_count
    summary["portfolio_total_budget_factor"] = total_budget_factor
    summary["portfolio_member_target_budget"] = member_target_budget
    summary["portfolio_member_configured_budget_mean"] = (
        total_configured_budget / restart_count if restart_count else None
    )
    summary["portfolio_member_actual_evaluations_mean"] = (
        total_actual_evaluations / restart_count if restart_count else None
    )
    summary["portfolio_profile"] = options.get("portfolio_profile") or "self"
    summary["selected_restart_index"] = best_restart_index + 1
    summary["selected_restart_seed"] = member_rows[best_restart_index]["seed"]
    summary["selected_restart_metric"] = best_metric
    summary["portfolio_member_metrics"] = [
        row["metric_value"] for row in member_rows if row.get("metric_value") is not None
    ]
    summary["portfolio_member_output_dirs"] = [row["output_dir"] for row in member_rows]
    summary["stop_reason"] = "restart_portfolio_completed"

    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(output_dir / "summary.json", summary)
    _write_rows_csv(output_dir / "history.csv", chosen_history)
    _write_rows_csv(output_dir / "portfolio_members.csv", member_rows)
    _write_json(output_dir / "portfolio_members.json", {"members": member_rows})
    return RunResult(summary=summary, history=chosen_history, output_dir=output_dir)


def _dominates_with_directions(
    candidate: list[float],
    incumbent: list[float],
    directions: list[bool],
) -> bool:
    better_or_equal = True
    strictly_better = False
    for index, direction in enumerate(directions):
        candidate_value = float(candidate[index])
        incumbent_value = float(incumbent[index])
        if direction:
            if candidate_value < incumbent_value:
                better_or_equal = False
                break
            if candidate_value > incumbent_value:
                strictly_better = True
        else:
            if candidate_value > incumbent_value:
                better_or_equal = False
                break
            if candidate_value < incumbent_value:
                strictly_better = True
    return better_or_equal and strictly_better


def _nondominated_front_indices(
    objective_vectors: list[list[float]],
    directions: list[bool],
) -> list[int]:
    front: list[int] = []
    for candidate_index, candidate in enumerate(objective_vectors):
        dominated = False
        for incumbent_index, incumbent in enumerate(objective_vectors):
            if candidate_index == incumbent_index:
                continue
            if _dominates_with_directions(incumbent, candidate, directions):
                dominated = True
                break
        if not dominated:
            front.append(candidate_index)
    return front


def _zdt1_restart_portfolio_result(
    config: GAConfig,
    *,
    output_root: Path,
) -> RunResult:
    output_dir = Path(output_root) / f"{_timestamp()}_{config.run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    options = dict(config.algorithm_options)
    restart_count = options.get("portfolio_restart_count", 2)
    if not isinstance(restart_count, int) or isinstance(restart_count, bool):
        restart_count = 2
    restart_count = max(2, restart_count)
    target_total_budget, total_budget_factor = _portfolio_total_budget(config, options)
    member_target_budget = max(1.0, float(target_total_budget) / float(restart_count))
    template_payload = _portfolio_template_payload(
        config,
        options,
        run_name=f"{config.run_name}_portfolio_template",
    )

    runtime = build_runtime_context(config)
    problem = runtime.problem
    base_seed = int(config.seed)
    merged_vectors: list[list[float]] = []
    member_rows: list[dict[str, Any]] = []
    total_runtime = 0.0
    total_actual_evaluations = 0
    total_configured_budget = 0
    best_result: RunResult | None = None
    best_hv: float | None = None
    best_restart_index = 0
    reference_metadata: dict[str, Any] = {}

    for restart_index in range(restart_count):
        member_config = _portfolio_member_config(
            template_payload,
            restart_index=restart_index,
            target_budget=member_target_budget,
            base_seed=base_seed,
            run_name=config.run_name,
        )
        result = run_experiment(member_config, output_root=output_dir / "portfolio_runs")
        configured_budget = configured_evaluation_budget(member_config)
        actual_evaluations = int(result.summary.get("actual_evaluations_used", 0) or 0)
        hypervolume = result.summary.get("hypervolume")
        pareto_vectors = result.summary.get("pareto_front_vectors", [])
        if isinstance(pareto_vectors, list):
            merged_vectors.extend(
                [list(vector) for vector in pareto_vectors if isinstance(vector, list)]
            )
        total_runtime += float(result.summary.get("runtime_seconds", 0.0) or 0.0)
        total_actual_evaluations += actual_evaluations
        total_configured_budget += configured_budget
        member_rows.append(
            {
                "restart_index": restart_index + 1,
                "seed": member_config.seed,
                "configured_budget": configured_budget,
                "actual_evaluations_used": actual_evaluations,
                "hypervolume": hypervolume,
                "output_dir": str(result.output_dir.resolve()),
            }
        )
        if isinstance(hypervolume, int | float) and not isinstance(hypervolume, bool):
            if best_hv is None or float(hypervolume) > best_hv:
                best_hv = float(hypervolume)
                best_result = result
                best_restart_index = restart_index

    if best_result is None:
        raise ValueError("restart portfolio did not produce a comparable ZDT1 result")
    if not merged_vectors:
        raise ValueError("restart portfolio did not expose any pareto_front_vectors")

    directions = resolve_objective_directions(len(merged_vectors[0]), config, problem)
    reference_point, reference_metadata = resolve_algorithm_reference_point(
        config,
        problem,
        merged_vectors,
        directions,
    )
    front_indices = _nondominated_front_indices(merged_vectors, directions)
    merged_metrics = front_metrics(
        front_indices,
        merged_vectors,
        directions,
        reference_point,
        len(merged_vectors),
    )

    chosen_history = list(best_result.history)
    chosen_summary = dict(best_result.summary)
    summary = dict(chosen_summary)
    summary.update(reference_metadata)
    summary["run_name"] = config.run_name
    summary["algorithm"] = "zdt1_restart_portfolio"
    summary["configured_evaluation_budget"] = total_configured_budget
    summary["target_total_budget"] = target_total_budget
    summary["actual_evaluations_used"] = total_actual_evaluations
    summary["total_actual_evaluations_used"] = total_actual_evaluations
    summary["extra_evaluations_from_adaptation"] = 0
    summary["runtime_seconds"] = total_runtime
    summary["portfolio_mode"] = "merged_archive"
    summary["portfolio_restart_count"] = restart_count
    summary["portfolio_total_budget_factor"] = total_budget_factor
    summary["portfolio_member_target_budget"] = member_target_budget
    summary["portfolio_member_configured_budget_mean"] = (
        total_configured_budget / restart_count if restart_count else None
    )
    summary["portfolio_member_actual_evaluations_mean"] = (
        total_actual_evaluations / restart_count if restart_count else None
    )
    summary["portfolio_profile"] = options.get("portfolio_profile") or "self"
    summary["selected_restart_index"] = best_restart_index + 1
    summary["selected_restart_seed"] = member_rows[best_restart_index]["seed"]
    summary["selected_restart_metric"] = best_hv
    summary["portfolio_member_metrics"] = [
        row["hypervolume"] for row in member_rows if row.get("hypervolume") is not None
    ]
    summary["portfolio_member_output_dirs"] = [row["output_dir"] for row in member_rows]
    summary["hypervolume"] = finite_or_none(merged_metrics["hypervolume"])
    summary["merged_archive_hv"] = summary["hypervolume"]
    summary["pareto_ratio"] = finite_or_none(merged_metrics["pareto_ratio"])
    summary["spread"] = finite_or_none(merged_metrics["spread"])
    summary["pareto_front_size"] = int(merged_metrics["pareto_front_size"])
    summary["pareto_front_vectors"] = [merged_vectors[index] for index in front_indices]
    summary["stop_reason"] = "restart_portfolio_completed"

    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(output_dir / "summary.json", summary)
    _write_rows_csv(output_dir / "history.csv", chosen_history)
    _write_rows_csv(output_dir / "portfolio_members.csv", member_rows)
    _write_json(output_dir / "portfolio_members.json", {"members": member_rows})
    return RunResult(summary=summary, history=chosen_history, output_dir=output_dir)


def _offset_history_rows(
    history_rows: list[dict[str, Any]],
    *,
    generation_offset: int,
    evaluation_offset: int,
    configured_budget: int,
    stage_label: str,
    skip_initial_row: bool = False,
) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    rows_to_use = history_rows[1:] if skip_initial_row and len(history_rows) > 1 else history_rows
    for row in rows_to_use:
        updated = dict(row)
        generation_value = _history_row_number(updated.get("generation"))
        if generation_value is not None:
            updated["generation"] = int(generation_value) + generation_offset
        evaluation_value = _history_row_number(updated.get("actual_evaluations_used"))
        if evaluation_value is not None:
            updated["actual_evaluations_used"] = int(evaluation_value) + evaluation_offset
        updated["configured_budget"] = configured_budget
        updated["gate_stage"] = stage_label
        adjusted.append(updated)
    return adjusted


def _two_stage_gate_decision(
    problem: str,
    options: dict[str, Any],
    *,
    pilot_summary: dict[str, Any],
    pilot_initial: dict[str, Any],
    pilot_final: dict[str, Any],
) -> list[str]:
    policy = str(options.get("gate_signal_policy", "")).strip()
    reasons: list[str] = []
    if problem == "tsp":
        late_floor = float(options.get("gate_late_improvement_floor", 0) or 0)
        gain_floor = float(options.get("gate_min_gain_ratio", 0.0) or 0.0)
        diversity_threshold = float(options.get("gate_diversity_threshold", 0.0) or 0.0)
        initial_best = pilot_initial.get("initial_best_route_distance")
        gain = pilot_initial.get("init_to_final_gain")
        gain_ratio = (
            float(gain) / float(initial_best)
            if isinstance(initial_best, int | float)
            and float(initial_best) > 0.0
            and isinstance(gain, int | float)
            else None
        )
        late_improvement = pilot_final.get("generations_to_last_improvement")
        diversity_signal = pilot_final.get("final_diversity_signal")
        if policy == "tsp_gain_or_stagnation":
            if gain_ratio is not None and gain_ratio < gain_floor:
                reasons.append(f"pilot_gain_ratio<{gain_floor:.3f}")
            if (
                isinstance(late_improvement, int | float)
                and float(late_improvement) <= late_floor
            ):
                reasons.append(f"pilot_last_improvement<={late_floor:.1f}")
        elif policy == "tsp_diversity_stagnation":
            if (
                isinstance(late_improvement, int | float)
                and float(late_improvement) <= late_floor
                and isinstance(diversity_signal, int | float)
                and float(diversity_signal) <= diversity_threshold
            ):
                reasons.append(
                    f"pilot_diversity<={diversity_threshold:.3f}&pilot_last_improvement<={late_floor:.1f}"
                )
        else:
            raise ValueError(f"Unsupported TSP gate_signal_policy: {policy}")
        return reasons

    if problem == "zdt1":
        late_floor = float(options.get("gate_late_improvement_floor", 0) or 0)
        min_hv = float(options.get("gate_min_hypervolume", 0.0) or 0.0)
        front_floor = float(options.get("gate_front_size_floor", 0.0) or 0.0)
        spread_ceiling = float(options.get("gate_spread_ceiling", 0.0) or 0.0)
        pilot_hv = pilot_summary.get("hypervolume")
        late_improvement = pilot_final.get("generations_to_last_improvement")
        front_size = pilot_final.get("final_front_size")
        spread_value = pilot_final.get("final_spread")
        if policy == "zdt1_hv_or_plateau":
            if isinstance(pilot_hv, int | float) and float(pilot_hv) < min_hv:
                reasons.append(f"pilot_hv<{min_hv:.3f}")
            if (
                isinstance(late_improvement, int | float)
                and float(late_improvement) <= late_floor
            ):
                reasons.append(f"pilot_last_improvement<={late_floor:.1f}")
        elif policy == "zdt1_front_plateau":
            if (
                isinstance(late_improvement, int | float)
                and float(late_improvement) <= late_floor
                and isinstance(front_size, int | float)
                and float(front_size) <= front_floor
            ):
                reasons.append(
                    f"pilot_front_size<={front_floor:.1f}&pilot_last_improvement<={late_floor:.1f}"
                )
            if (
                isinstance(spread_value, int | float)
                and spread_ceiling > 0.0
                and float(spread_value) >= spread_ceiling
            ):
                reasons.append(f"pilot_spread>={spread_ceiling:.3f}")
        else:
            raise ValueError(f"Unsupported ZDT1 gate_signal_policy: {policy}")
        return reasons

    raise ValueError(f"Unsupported gate problem: {problem}")


def _two_stage_escalation_result(
    config: GAConfig,
    *,
    output_root: Path,
    problem: str,
) -> RunResult:
    output_dir = Path(output_root) / f"{_timestamp()}_{config.run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    options = dict(config.algorithm_options)
    fast_profile = options.get("gate_fast_profile")
    canonical_profile = options.get("gate_canonical_profile")
    if not isinstance(fast_profile, str) or not fast_profile.strip():
        raise ValueError("two-stage gate requires algorithm_options.gate_fast_profile")
    if not isinstance(canonical_profile, str) or not canonical_profile.strip():
        raise ValueError("two-stage gate requires algorithm_options.gate_canonical_profile")

    canonical_payload = _profile_payload_for_context(
        canonical_profile,
        context_config=config,
        run_name=f"{config.run_name}_canonical_template",
    )
    fast_payload = _profile_payload_for_context(
        fast_profile,
        context_config=config,
        run_name=f"{config.run_name}_fast_template",
    )
    canonical_config = GAConfig.from_dict(canonical_payload)
    fast_config = GAConfig.from_dict(fast_payload)
    canonical_budget = configured_evaluation_budget(canonical_config)
    pilot_fraction = options.get("gate_pilot_budget_fraction", 0.25)
    if not isinstance(pilot_fraction, int | float) or isinstance(pilot_fraction, bool):
        pilot_fraction = 0.25
    pilot_fraction = min(0.5, max(0.1, float(pilot_fraction)))
    pilot_target_budget = float(canonical_budget) * pilot_fraction
    pilot_generations = _generations_for_target_budget(fast_config, pilot_target_budget)

    pilot_payload = deepcopy(fast_payload)
    pilot_payload["run_name"] = f"{config.run_name}_pilot"
    pilot_payload["generations"] = pilot_generations
    pilot_payload.setdefault("algorithm_options", {})
    pilot_payload["algorithm_options"] = dict(pilot_payload["algorithm_options"])
    pilot_payload["algorithm_options"]["_return_final_population"] = True
    pilot_config = GAConfig.from_dict(pilot_payload)
    pilot_result = run_experiment(pilot_config, output_root=output_dir / "gate_runs")
    pilot_history = list(pilot_result.history)
    pilot_initial = _extract_initial_history_metrics(problem, pilot_history)
    pilot_final = _extract_final_history_metrics(problem, pilot_history)
    escalation_reasons = _two_stage_gate_decision(
        problem,
        options,
        pilot_summary=pilot_result.summary,
        pilot_initial=pilot_initial,
        pilot_final=pilot_final,
    )
    escalation_triggered = bool(escalation_reasons)

    total_runtime = float(pilot_result.summary.get("runtime_seconds", 0.0) or 0.0)
    pilot_actual_evaluations = int(pilot_result.summary.get("actual_evaluations_used", 0) or 0)
    total_actual_evaluations = pilot_actual_evaluations
    combined_history = _offset_history_rows(
        pilot_history,
        generation_offset=0,
        evaluation_offset=0,
        configured_budget=canonical_budget,
        stage_label="pilot",
    )
    chosen_summary = dict(pilot_result.summary)
    selected_profile = "fast"
    continuation_output_dir: str | None = None
    escalation_output_dir: str | None = None
    escalation_actual_evaluations = 0

    if escalation_triggered:
        canonical_payload["run_name"] = f"{config.run_name}_canonical_rerun"
        canonical_run_config = GAConfig.from_dict(canonical_payload)
        canonical_result = run_experiment(canonical_run_config, output_root=output_dir / "gate_runs")
        escalation_actual_evaluations = int(
            canonical_result.summary.get("actual_evaluations_used", 0) or 0
        )
        total_actual_evaluations += escalation_actual_evaluations
        total_runtime += float(canonical_result.summary.get("runtime_seconds", 0.0) or 0.0)
        escalation_output_dir = str(canonical_result.output_dir.resolve())
        combined_history.extend(
            _offset_history_rows(
                canonical_result.history,
                generation_offset=pilot_generations + 1,
                evaluation_offset=pilot_actual_evaluations,
                configured_budget=canonical_budget,
                stage_label="canonical_rerun",
            )
        )
        chosen_summary = dict(canonical_result.summary)
        selected_profile = "canonical"
    else:
        final_population = pilot_result.summary.get("final_population")
        if not isinstance(final_population, list):
            raise ValueError("Pilot run did not expose final_population for fast continuation")
        remaining_generations = max(1, fast_config.generations - pilot_generations)
        continue_payload = deepcopy(fast_payload)
        continue_payload["run_name"] = f"{config.run_name}_fast_continue"
        continue_payload["generations"] = remaining_generations
        continue_payload.setdefault("algorithm_options", {})
        continue_payload["algorithm_options"] = dict(continue_payload["algorithm_options"])
        continue_payload["algorithm_options"]["_initial_population"] = deepcopy(final_population)
        continue_config = GAConfig.from_dict(continue_payload)
        continue_result = run_experiment(continue_config, output_root=output_dir / "gate_runs")
        escalation_actual_evaluations = int(
            continue_result.summary.get("actual_evaluations_used", 0) or 0
        )
        total_actual_evaluations += escalation_actual_evaluations
        total_runtime += float(continue_result.summary.get("runtime_seconds", 0.0) or 0.0)
        continuation_output_dir = str(continue_result.output_dir.resolve())
        combined_history.extend(
            _offset_history_rows(
                continue_result.history,
                generation_offset=pilot_generations,
                evaluation_offset=pilot_actual_evaluations,
                configured_budget=canonical_budget,
                stage_label="fast_continue",
                skip_initial_row=True,
            )
        )
        chosen_summary = dict(continue_result.summary)

    summary = dict(chosen_summary)
    summary["run_name"] = config.run_name
    summary["algorithm"] = f"{problem}_two_stage_gate"
    summary["configured_evaluation_budget"] = canonical_budget
    summary["actual_evaluations_used"] = total_actual_evaluations
    summary["total_actual_evaluations_used"] = total_actual_evaluations
    summary["pilot_actual_evaluations_used"] = pilot_actual_evaluations
    summary["escalation_actual_evaluations_used"] = escalation_actual_evaluations
    summary["extra_evaluations_from_adaptation"] = 0
    summary["runtime_seconds"] = total_runtime
    summary["stop_reason"] = (
        "canonical_escalated_after_pilot"
        if escalation_triggered
        else "fast_completed_after_pilot"
    )
    summary["final_generation"] = (
        int(combined_history[-1]["generation"]) if combined_history else chosen_summary.get("final_generation")
    )
    summary["selected_profile"] = selected_profile
    summary["escalation_triggered"] = 1.0 if escalation_triggered else 0.0
    summary["escalation_reason"] = (
        "; ".join(escalation_reasons) if escalation_reasons else "kept_fast"
    )
    summary["pilot_budget_fraction"] = pilot_fraction
    summary["pilot_configured_budget"] = configured_evaluation_budget(pilot_config)
    summary["pilot_generations"] = pilot_generations
    summary["pilot_initial_best_route_distance"] = pilot_initial.get("initial_best_route_distance")
    summary["pilot_init_to_final_gain"] = pilot_initial.get("init_to_final_gain")
    summary["pilot_generations_to_last_improvement"] = pilot_final.get(
        "generations_to_last_improvement"
    )
    summary["pilot_final_diversity_signal"] = pilot_final.get("final_diversity_signal")
    summary["pilot_final_hypervolume"] = pilot_result.summary.get("hypervolume")
    summary["pilot_final_front_size"] = pilot_final.get("final_front_size")
    summary["pilot_final_spread"] = pilot_final.get("final_spread")
    summary["pilot_output_dir"] = str(pilot_result.output_dir.resolve())
    summary["continuation_output_dir"] = continuation_output_dir
    summary["escalation_output_dir"] = escalation_output_dir

    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(output_dir / "summary.json", summary)
    _write_rows_csv(output_dir / "history.csv", combined_history)
    return RunResult(summary=summary, history=combined_history, output_dir=output_dir)


def _knapsack_rerun_gate_result(
    config: GAConfig,
    *,
    output_root: Path,
) -> RunResult:
    output_dir = Path(output_root) / f"{_timestamp()}_{config.run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    configured_budget = config.population_size * (config.generations + 2)
    base_config = config.to_dict()
    options = dict(config.algorithm_options)
    pilot_fraction = options.get("gate_pilot_generation_fraction", 0.25)
    if not isinstance(pilot_fraction, int | float) or isinstance(pilot_fraction, bool):
        pilot_fraction = 0.25
    pilot_fraction = min(0.75, max(0.1, float(pilot_fraction)))
    pilot_generations = max(1, min(config.generations - 3, int(round(config.generations * pilot_fraction))))
    rerun_generations = max(1, config.generations - pilot_generations - 2)
    initial_feasible_threshold = options.get("gate_initial_feasible_threshold", 0.05)
    if not isinstance(initial_feasible_threshold, int | float) or isinstance(
        initial_feasible_threshold, bool
    ):
        initial_feasible_threshold = 0.05
    first_feasible_limit = options.get(
        "gate_first_feasible_generation_limit",
        max(4, int(round(pilot_generations * 0.5))),
    )
    if not isinstance(first_feasible_limit, int) or isinstance(first_feasible_limit, bool):
        first_feasible_limit = max(4, int(round(pilot_generations * 0.5)))
    mean_violation_threshold = options.get("gate_mean_violation_threshold")
    if not isinstance(mean_violation_threshold, int | float) or isinstance(
        mean_violation_threshold, bool
    ):
        mean_violation_threshold = None

    pilot_payload = apply_overrides(
        base_config,
        {
            "run_name": f"{config.run_name}_pilot",
            "algorithm": "ga",
            "generations": pilot_generations,
            "algorithm_options": {"adaptive_policy": "none"},
        },
    )
    pilot_config = GAConfig.from_dict(pilot_payload)
    pilot_result = run_experiment(pilot_config, output_root=output_dir / "gate_runs")
    pilot_history = list(pilot_result.history)
    pilot_initial = _extract_initial_history_metrics("knapsack", pilot_history)
    pilot_final = _extract_final_history_metrics("knapsack", pilot_history)
    initial_feasible_fraction = pilot_initial.get("initial_feasible_fraction")
    generations_to_first_feasible = pilot_initial.get("generations_to_first_feasible")
    pilot_mean_violation = pilot_final.get("final_mean_constraint_violation")

    rerun_reasons: list[str] = []
    if isinstance(initial_feasible_fraction, int | float) and float(initial_feasible_fraction) <= float(
        initial_feasible_threshold
    ):
        rerun_reasons.append(
            f"initial_feasible_fraction<={float(initial_feasible_threshold):.3f}"
        )
    if generations_to_first_feasible is None or (
        isinstance(generations_to_first_feasible, int | float)
        and float(generations_to_first_feasible) >= float(first_feasible_limit)
    ):
        rerun_reasons.append(f"generations_to_first_feasible>={int(first_feasible_limit)}")
    if (
        isinstance(mean_violation_threshold, int | float)
        and
        isinstance(pilot_mean_violation, int | float)
        and float(pilot_mean_violation) >= float(mean_violation_threshold)
    ):
        rerun_reasons.append(
            f"pilot_mean_violation>={float(mean_violation_threshold):.3f}"
        )

    rerun_triggered = bool(rerun_reasons)
    combined_history = _offset_history_rows(
        pilot_history,
        generation_offset=0,
        evaluation_offset=0,
        configured_budget=configured_budget,
        stage_label="pilot",
    )
    total_runtime = float(pilot_result.summary.get("runtime_seconds", 0.0) or 0.0)
    total_actual_evaluations = int(pilot_result.summary.get("actual_evaluations_used", 0) or 0)
    chosen_summary = dict(pilot_result.summary)
    rerun_summary: dict[str, Any] | None = None
    rerun_output_dir: str | None = None
    rerun_actual_evaluations = 0

    if rerun_triggered:
        repair_payload = apply_overrides(
            base_config,
            {
                "run_name": f"{config.run_name}_repair_rerun",
                "algorithm": "hybrid_ga",
                "generations": rerun_generations,
                "algorithm_options": {
                    "adaptive_policy": "none",
                    "init_strategy": "none",
                    "seed_fraction": 0.0,
                    "repair_strategy": "knapsack_greedy_fill",
                    "local_search_strategy": "none",
                },
            },
        )
        repair_config = GAConfig.from_dict(repair_payload)
        repair_result = run_experiment(repair_config, output_root=output_dir / "gate_runs")
        rerun_summary = dict(repair_result.summary)
        rerun_output_dir = str(repair_result.output_dir.resolve())
        rerun_actual_evaluations = int(repair_result.summary.get("actual_evaluations_used", 0) or 0)
        total_runtime += float(repair_result.summary.get("runtime_seconds", 0.0) or 0.0)
        total_actual_evaluations += rerun_actual_evaluations
        repair_history = _offset_history_rows(
            repair_result.history,
            generation_offset=pilot_generations + 1,
            evaluation_offset=int(pilot_result.summary.get("actual_evaluations_used", 0) or 0),
            configured_budget=configured_budget,
            stage_label="rerun",
        )
        combined_history.extend(repair_history)
        pilot_best = _knapsack_best_feasible(pilot_result.summary)
        repair_best = _knapsack_best_feasible(repair_result.summary)
        if repair_best is not None and (
            pilot_best is None or float(repair_best) >= float(pilot_best)
        ):
            chosen_summary = dict(repair_result.summary)

    summary = dict(chosen_summary)
    summary["run_name"] = config.run_name
    summary["algorithm"] = "knapsack_repair_rerun_gate"
    summary["runtime_seconds"] = total_runtime
    summary["configured_evaluation_budget"] = configured_budget
    summary["actual_evaluations_used"] = total_actual_evaluations
    summary["extra_evaluations_from_adaptation"] = 0
    summary["adaptive_policy"] = "none"
    summary["stop_reason"] = "rerun_completed" if rerun_triggered else "pilot_only_completed"
    summary["final_generation"] = (
        int(combined_history[-1]["generation"]) if combined_history else config.generations
    )
    summary["rerun_triggered"] = 1.0 if rerun_triggered else 0.0
    summary["rerun_trigger_reason"] = "; ".join(rerun_reasons) if rerun_reasons else "pilot_only"
    summary["pilot_actual_evaluations_used"] = int(
        pilot_result.summary.get("actual_evaluations_used", 0) or 0
    )
    summary["rerun_actual_evaluations_used"] = rerun_actual_evaluations
    summary["total_actual_evaluations_used"] = total_actual_evaluations
    summary["pilot_generations"] = pilot_generations
    summary["rerun_generations"] = rerun_generations if rerun_triggered else 0
    summary["pilot_initial_feasible_fraction"] = initial_feasible_fraction
    summary["pilot_generations_to_first_feasible"] = generations_to_first_feasible
    summary["pilot_mean_constraint_violation"] = pilot_mean_violation
    summary["pilot_output_dir"] = str(pilot_result.output_dir.resolve())
    summary["rerun_output_dir"] = rerun_output_dir

    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(output_dir / "summary.json", summary)
    _write_rows_csv(output_dir / "history.csv", combined_history)
    return RunResult(summary=summary, history=combined_history, output_dir=output_dir)


def _coerce_metric_value(problem: str, metric_name: str, row: dict[str, Any]) -> float | None:
    value = row.get(metric_name)
    if metric_name == "best_route_distance":
        if isinstance(value, int | float):
            return float(value)
        best_fitness = row.get("best_fitness")
        if problem == "tsp" and isinstance(best_fitness, int | float):
            return float(-best_fitness)
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def _metric_is_lower_better(metric_name: str) -> bool:
    return metric_name in _LOWER_IS_BETTER_METRICS


def _problem_history_metric(problem: str, plotting: dict[str, Any]) -> str:
    configured = plotting.get("history_metric")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if problem == "zdt1":
        return "hypervolume"
    if problem == "tsp":
        return "best_route_distance"
    return "best_fitness"


def _parameter_columns(combo: dict[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for key, value in combo.items():
        columns[key] = value
        columns[f"param__{key.replace('.', '_')}"] = value
    return columns


def _resolved_config_columns(config_data: dict[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for key in _RESOLVED_CONFIG_KEYS:
        value = config_data.get(key)
        if value is not None:
            columns[key] = value
    options = config_data.get("algorithm_options", {})
    if isinstance(options, dict):
        for key in _RESOLVED_ALGORITHM_OPTION_KEYS:
            value = options.get(key)
            if value is not None:
                columns[f"algorithm_options.{key}"] = value
    return columns


def _summary_config_columns(row: dict[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for key in _RESOLVED_CONFIG_KEYS:
        value = row.get(key)
        if value is not None:
            columns[key] = value
    for key in _RESOLVED_ALGORITHM_OPTION_KEYS:
        flat_key = f"algorithm_options.{key}"
        value = row.get(flat_key)
        if value is not None:
            columns[flat_key] = value
    return columns


def _variant_payload(
    study: LocalStudy,
    combo: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_name = combo.get("study_variant")
    combo_without_variant = {
        key: value for key, value in combo.items() if key != "study_variant"
    }
    if variant_name is None:
        return {}, combo_without_variant
    overrides = study.variant_overrides.get(str(variant_name))
    if overrides is None:
        raise ValueError(f"Unknown study_variant: {variant_name}")
    return dict(overrides), combo_without_variant


def _combo_label(combo: dict[str, Any]) -> str:
    return " | ".join(f"{key}={value}" for key, value in combo.items())


def _run_name(study_name: str, label: str, seed: int) -> str:
    return f"{_safe_name(study_name)}__{_safe_name(label)}__seed{seed}"


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        if value.startswith("[") or value.startswith("{"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


def _read_history_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _parse_scalar(value) for key, value in row.items()} for row in reader]


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: _csv_value(row.get(field)) for field in fieldnames} for row in rows
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _summary_metric_columns(problem: str) -> tuple[str, ...]:
    if problem == "onemax":
        return (
            "best_fitness_mean",
            "target_hit_rate",
            "evaluations_to_target_mean",
            "generations_to_target_mean",
            "primary_metric_median",
            "primary_metric_ci_low",
            "primary_metric_ci_high",
            "configured_budget",
            "actual_evaluations_used_mean",
            "extra_evaluations_from_adaptation_mean",
            "trigger_fire_count_mean",
            "first_trigger_generation_mean",
            "average_refresh_fraction_realized_mean",
            "runtime_seconds_mean",
        )
    if problem == "knapsack":
        return (
            "best_feasible_fitness_mean",
            "regret_vs_greedy_local_search_mean",
            "regret_vs_none_mean",
            "regret_vs_repair_only_mean",
            "regret_vs_multi_none_reference_mean",
            "regret_vs_current_restart_experimental_mean",
            "feasible_rate",
            "mean_violation_mean",
            "initial_feasible_fraction_mean",
            "generations_to_first_feasible_mean",
            "init_to_final_gain_mean",
            "primary_metric_median",
            "primary_metric_ci_low",
            "primary_metric_ci_high",
            "configured_budget",
            "actual_evaluations_used_mean",
            "rerun_trigger_rate",
            "extra_evaluations_from_adaptation_mean",
            "trigger_fire_count_mean",
            "first_trigger_generation_mean",
            "post_trigger_improvement_mean",
            "average_refresh_fraction_realized_mean",
            "runtime_seconds_mean",
        )
    if problem == "tsp":
        return (
            "best_route_distance_mean",
            "best_route_distance_std",
            "best_fitness_mean",
            "initial_best_route_distance_mean",
            "initial_mean_route_distance_mean",
            "initial_population_diversity_mean",
            "init_to_final_gain_mean",
            "generations_to_first_improvement_mean",
            "regret_vs_oracle_tested_candidate_mean",
            "regret_vs_current_preferred_profile_mean",
            "regret_vs_oracle_fixed_policy_mean",
            "regret_vs_always_decay_mutation_mean",
            "regret_vs_always_low_diversity_injection_mean",
            "win_rate_vs_decay_mutation",
            "win_rate_vs_low_diversity_injection",
            "win_rate_vs_none",
            "regret_vs_canonical_once_mean",
            "regret_vs_fast_once_mean",
            "best_of_k_improvement_over_single_fast_mean",
            "primary_metric_median",
            "primary_metric_ci_low",
            "primary_metric_ci_high",
            "configured_budget",
            "actual_evaluations_used_mean",
            "portfolio_restart_count_mean",
            "early_stop_trigger_rate",
            "extra_evaluations_from_adaptation_mean",
            "trigger_fire_count_mean",
            "first_trigger_generation_mean",
            "mode_switch_count_mean",
            "first_switch_generation_mean",
            "time_in_decay_mode_mean",
            "time_in_trigger_mode_mean",
            "post_trigger_improvement_mean",
            "time_to_first_nontrivial_improvement_after_trigger_mean",
            "collapse_onset_generation_mean",
            "trigger_delay_from_collapse_mean",
            "switch_delay_from_collapse_mean",
            "useless_trigger_rate_mean",
            "generations_to_last_improvement_mean",
            "average_refresh_fraction_realized_mean",
            "realized_refresh_volume_mean",
            "runtime_seconds_mean",
        )
    return (
        "hypervolume_mean",
        "merged_archive_hv_mean",
        "pareto_ratio_mean",
        "spread_mean",
        "front_size_mean",
        "pareto_front_size_mean",
        "regret_vs_current_profile_mean",
        "regret_vs_none_mean",
        "regret_vs_canonical_once_mean",
        "regret_vs_fast_once_mean",
        "merged_archive_gain_over_single_fast_mean",
        "primary_metric_median",
        "primary_metric_ci_low",
        "primary_metric_ci_high",
        "configured_budget",
        "actual_evaluations_used_mean",
        "portfolio_restart_count_mean",
        "early_stop_trigger_rate",
        "extra_evaluations_from_adaptation_mean",
        "trigger_fire_count_mean",
        "first_trigger_generation_mean",
        "post_trigger_improvement_mean",
        "hv_plateau_generation_mean",
        "average_refresh_fraction_realized_mean",
        "runtime_seconds_mean",
    )


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _knapsack_variant_key(row: dict[str, Any]) -> str:
    variant = row.get("study_variant")
    if isinstance(variant, str) and variant.strip():
        return variant.strip()
    label = row.get("variant_label")
    return str(label).strip() if isinstance(label, str) else ""


def _annotate_knapsack_regret_rows(raw_rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, float], dict[str, list[float]]] = {}
    for row in raw_rows:
        case_id = row.get("case_id")
        configured_budget = row.get("configured_budget")
        best_feasible = row.get("best_feasible_fitness")
        if (
            not isinstance(case_id, str)
            or not isinstance(configured_budget, int | float)
            or not isinstance(best_feasible, int | float)
            or isinstance(best_feasible, bool)
        ):
            continue
        grouped.setdefault((case_id, float(configured_budget)), {}).setdefault(
            _knapsack_variant_key(row),
            [],
        ).append(float(best_feasible))

    means: dict[tuple[str, float], dict[str, float]] = {}
    for key, variants in grouped.items():
        means[key] = {variant: mean(values) for variant, values in variants.items() if values}

    for row in raw_rows:
        case_id = row.get("case_id")
        configured_budget = row.get("configured_budget")
        current_value = row.get("best_feasible_fitness")
        if (
            not isinstance(case_id, str)
            or not isinstance(configured_budget, int | float)
            or not isinstance(current_value, int | float)
            or isinstance(current_value, bool)
        ):
            continue
        label_means = means.get((case_id, float(configured_budget)), {})
        greedy_value = label_means.get("greedy_local_search")
        none_value = label_means.get("none")
        repair_value = label_means.get("repair_only")
        restart_value = label_means.get("stagnation_restart")
        if isinstance(greedy_value, int | float):
            row["regret_vs_greedy_local_search"] = float(greedy_value) - float(current_value)
        if isinstance(none_value, int | float):
            row["regret_vs_none"] = float(none_value) - float(current_value)
        if isinstance(repair_value, int | float):
            row["regret_vs_repair_only"] = float(repair_value) - float(current_value)
        if isinstance(restart_value, int | float):
            row["regret_vs_current_restart_experimental"] = float(restart_value) - float(
                current_value
            )


def _study_metadata_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in _STUDY_METADATA_KEYS
        if key in payload and payload.get(key) is not None
    }


def _extract_problem_metrics(problem: str, summary: dict[str, Any]) -> dict[str, Any]:
    configured_budget = summary.get("configured_evaluation_budget")
    actual_evaluations = summary.get("actual_evaluations_used")
    metrics: dict[str, Any] = {
        "runtime_seconds": summary.get("runtime_seconds"),
        "final_generation": summary.get("final_generation"),
        "stop_reason": summary.get("stop_reason"),
        "best_fitness": summary.get("best_fitness"),
        "configured_budget": configured_budget,
        "actual_evaluations_used": actual_evaluations,
        "extra_evaluations_from_adaptation": (
            summary.get("extra_evaluations_from_adaptation")
            if summary.get("extra_evaluations_from_adaptation") is not None
            else summary.get("hybrid_extra_evaluations")
        ),
        "stopped_before_budget": (
            1.0
            if isinstance(configured_budget, int | float)
            and isinstance(actual_evaluations, int | float)
            and not isinstance(configured_budget, bool)
            and not isinstance(actual_evaluations, bool)
            and float(actual_evaluations) < float(configured_budget)
            else 0.0
        ),
        "early_stop_triggered": (
            1.0 if bool(summary.get("early_stop_triggered")) else 0.0
        ),
        "early_stop_generation": summary.get("early_stop_generation"),
        "early_stop_policy": summary.get("early_stop_policy"),
        "adaptive_policy": summary.get("adaptive_policy"),
        "hybrid_seeded_individuals": summary.get("hybrid_seeded_individuals"),
        "adaptive_event_count": summary.get("adaptive_event_count"),
        "trigger_fire_count": summary.get(
            "trigger_fire_count",
            summary.get("adaptive_event_count"),
        ),
        "first_trigger_generation": summary.get("first_trigger_generation"),
        "mode_switch_count": summary.get("mode_switch_count"),
        "first_switch_generation": summary.get("first_switch_generation"),
        "time_in_decay_mode": summary.get("time_in_decay_mode"),
        "time_in_trigger_mode": summary.get("time_in_trigger_mode"),
        "mode_switch_generations": summary.get("mode_switch_generations"),
        "mode_switch_modes": summary.get("mode_switch_modes"),
        "post_trigger_improvement": summary.get("post_trigger_improvement"),
        "average_refresh_fraction_realized": summary.get("average_refresh_fraction_realized"),
        "total_refresh_fraction_realized": summary.get("total_refresh_fraction_realized"),
        "trigger_event_generations": summary.get("trigger_event_generations", []),
        "trigger_event_names": summary.get("trigger_event_names", []),
        "rerun_triggered": 1.0 if bool(summary.get("rerun_triggered")) else 0.0,
        "rerun_trigger_reason": summary.get("rerun_trigger_reason"),
        "pilot_actual_evaluations_used": summary.get("pilot_actual_evaluations_used"),
        "rerun_actual_evaluations_used": summary.get("rerun_actual_evaluations_used"),
        "total_actual_evaluations_used": summary.get(
            "total_actual_evaluations_used",
            actual_evaluations,
        ),
        "pilot_initial_feasible_fraction": summary.get("pilot_initial_feasible_fraction"),
        "pilot_generations_to_first_feasible": summary.get(
            "pilot_generations_to_first_feasible"
        ),
        "pilot_mean_constraint_violation": summary.get("pilot_mean_constraint_violation"),
        "selected_profile": summary.get("selected_profile"),
        "escalation_triggered": 1.0 if bool(summary.get("escalation_triggered")) else 0.0,
        "escalation_reason": summary.get("escalation_reason"),
        "pilot_budget_fraction": summary.get("pilot_budget_fraction"),
        "pilot_configured_budget": summary.get("pilot_configured_budget"),
        "escalation_actual_evaluations_used": summary.get("escalation_actual_evaluations_used"),
        "pilot_generations": summary.get("pilot_generations"),
        "pilot_initial_best_route_distance": summary.get("pilot_initial_best_route_distance"),
        "pilot_init_to_final_gain": summary.get("pilot_init_to_final_gain"),
        "pilot_generations_to_last_improvement": summary.get(
            "pilot_generations_to_last_improvement"
        ),
        "pilot_final_diversity_signal": summary.get("pilot_final_diversity_signal"),
        "pilot_final_hypervolume": summary.get("pilot_final_hypervolume"),
        "pilot_final_front_size": summary.get("pilot_final_front_size"),
        "pilot_final_spread": summary.get("pilot_final_spread"),
        "target_total_budget": summary.get("target_total_budget"),
        "portfolio_mode": summary.get("portfolio_mode"),
        "portfolio_profile": summary.get("portfolio_profile"),
        "portfolio_restart_count": summary.get("portfolio_restart_count"),
        "portfolio_total_budget_factor": summary.get("portfolio_total_budget_factor"),
        "portfolio_member_target_budget": summary.get("portfolio_member_target_budget"),
        "portfolio_member_configured_budget_mean": summary.get(
            "portfolio_member_configured_budget_mean"
        ),
        "portfolio_member_actual_evaluations_mean": summary.get(
            "portfolio_member_actual_evaluations_mean"
        ),
        "selected_restart_index": summary.get("selected_restart_index"),
    }
    if problem == "onemax":
        best_fitness = summary.get("best_fitness")
        target_fitness = summary.get("target_fitness")
        target_hit = (
            summary.get("stop_reason") == "target_fitness_reached"
            or (
                isinstance(best_fitness, int | float)
                and isinstance(target_fitness, int | float)
                and float(best_fitness) >= float(target_fitness)
            )
        )
        metrics.update(
            {
                "target_hit": bool(target_hit),
                "evaluations_to_target": summary.get("evaluations_to_target"),
                "generations_to_target": summary.get("convergence_generation")
                if summary.get("convergence_generation") is not None
                else summary.get("final_generation"),
            }
        )
        return metrics
    if problem == "knapsack":
        feasible = bool(summary.get("best_is_feasible"))
        best_feasible = summary.get("best_feasible_fitness")
        if not isinstance(best_feasible, int | float) or isinstance(best_feasible, bool):
            best_feasible = summary.get("best_total_value") if feasible else None
        metrics.update(
            {
                "best_feasible_fitness": best_feasible,
                "feasible": feasible,
                "mean_violation": summary.get("best_constraint_violation"),
                "mean_violation_rate": summary.get("best_constraint_violation_rate"),
                "best_total_value": summary.get("best_total_value"),
            }
        )
        return metrics
    if problem == "tsp":
        metrics["best_route_distance"] = summary.get("best_route_distance")
        return metrics
    metrics.update(
        {
            "hypervolume": summary.get("hypervolume"),
            "merged_archive_hv": summary.get("merged_archive_hv"),
            "pareto_ratio": summary.get("pareto_ratio"),
            "spread": summary.get("spread"),
            "pareto_front_size": summary.get("pareto_front_size"),
            "pareto_front_vectors": summary.get("pareto_front_vectors", []),
        }
    )
    return metrics


def _extract_final_history_metrics(
    problem: str,
    history_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not history_rows:
        return {}
    final_row = history_rows[-1]
    metrics: dict[str, Any] = {}
    for key in (
        "generations_since_last_improvement",
        "adaptive_event_count",
        "adaptive_mutation_rate",
        "diversity_signal",
    ):
        value = final_row.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            metrics[f"final_{key}"] = float(value)

    final_generation_value = final_row.get("generation")
    final_stagnation_value = final_row.get("generations_since_last_improvement")
    if (
        isinstance(final_generation_value, int | float)
        and not isinstance(final_generation_value, bool)
        and isinstance(final_stagnation_value, int | float)
        and not isinstance(final_stagnation_value, bool)
    ):
        metrics["generations_to_last_improvement"] = float(final_generation_value) - float(
            final_stagnation_value
        )

    diversity_column = _problem_diversity_column(problem)
    if diversity_column:
        value = final_row.get(diversity_column)
        if isinstance(value, int | float) and not isinstance(value, bool):
            metrics[f"final_{diversity_column}"] = float(value)

    if problem == "tsp":
        for key in ("best_route_distance", "edge_diversity_ratio", "positional_diversity"):
            value = final_row.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                metrics[f"final_{key}"] = float(value)
    elif problem == "knapsack":
        for key in ("feasible_ratio", "mean_constraint_violation"):
            value = final_row.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                metrics[f"final_{key}"] = float(value)
    elif problem == "zdt1":
        for key in ("hypervolume", "spread", "pareto_ratio", "front_size", "population_spread"):
            value = final_row.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                metrics[f"final_{key}"] = float(value)
        pareto_front_size = final_row.get("pareto_front_size")
        if isinstance(pareto_front_size, int | float) and not isinstance(
            pareto_front_size,
            bool,
        ):
            metrics["final_front_size"] = float(pareto_front_size)
        plateau_generation = metrics.get("generations_to_last_improvement")
        if isinstance(plateau_generation, int | float) and not isinstance(
            plateau_generation,
            bool,
        ):
            metrics["hv_plateau_generation"] = float(plateau_generation)
    return metrics


def _extract_initial_history_metrics(
    problem: str,
    history_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not history_rows:
        return {}

    if problem == "knapsack":
        first_row = history_rows[0]
        metrics: dict[str, Any] = {}
        initial_feasible = first_row.get("feasible_ratio")
        if isinstance(initial_feasible, int | float) and not isinstance(initial_feasible, bool):
            metrics["initial_feasible_fraction"] = float(initial_feasible)
        initial_best_value = first_row.get("best_total_value")
        final_best_value = history_rows[-1].get("best_total_value")
        if (
            isinstance(initial_best_value, int | float)
            and not isinstance(initial_best_value, bool)
            and isinstance(final_best_value, int | float)
            and not isinstance(final_best_value, bool)
        ):
            metrics["init_to_final_gain"] = float(final_best_value) - float(initial_best_value)
        for row in history_rows:
            generation = row.get("generation")
            feasible_ratio = row.get("feasible_ratio")
            if (
                isinstance(generation, int | float)
                and not isinstance(generation, bool)
                and isinstance(feasible_ratio, int | float)
                and not isinstance(feasible_ratio, bool)
                and float(feasible_ratio) > 0.0
            ):
                metrics["generations_to_first_feasible"] = float(generation)
                break
        return metrics

    if problem != "tsp":
        return {}

    first_row = history_rows[0]
    metrics: dict[str, Any] = {}
    initial_best = first_row.get("best_route_distance")
    if not isinstance(initial_best, int | float) or isinstance(initial_best, bool):
        best_fitness = first_row.get("best_fitness")
        if isinstance(best_fitness, int | float) and not isinstance(best_fitness, bool):
            initial_best = float(-best_fitness)
    initial_mean_fitness = first_row.get("mean_fitness")
    initial_diversity = first_row.get("edge_diversity_ratio")

    if isinstance(initial_best, int | float) and not isinstance(initial_best, bool):
        initial_best_value = float(initial_best)
        metrics["initial_best_route_distance"] = initial_best_value
        final_best = history_rows[-1].get("best_route_distance")
        if not isinstance(final_best, int | float) or isinstance(final_best, bool):
            final_best_fitness = history_rows[-1].get("best_fitness")
            if isinstance(
                final_best_fitness, int | float
            ) and not isinstance(final_best_fitness, bool):
                final_best = float(-final_best_fitness)
        if isinstance(final_best, int | float) and not isinstance(final_best, bool):
            metrics["init_to_final_gain"] = initial_best_value - float(final_best)
        for row in history_rows[1:]:
            best_distance = row.get("best_route_distance")
            if not isinstance(best_distance, int | float) or isinstance(best_distance, bool):
                best_fitness = row.get("best_fitness")
                if isinstance(best_fitness, int | float) and not isinstance(best_fitness, bool):
                    best_distance = float(-best_fitness)
            generation = row.get("generation")
            if (
                isinstance(best_distance, int | float)
                and not isinstance(best_distance, bool)
                and isinstance(generation, int | float)
                and not isinstance(generation, bool)
                and float(best_distance) < (initial_best_value - 1e-9)
            ):
                metrics["generations_to_first_improvement"] = float(generation)
                break

    if isinstance(initial_mean_fitness, int | float) and not isinstance(initial_mean_fitness, bool):
        metrics["initial_mean_route_distance"] = float(-initial_mean_fitness)
    if isinstance(initial_diversity, int | float) and not isinstance(initial_diversity, bool):
        metrics["initial_population_diversity"] = float(initial_diversity)
    return metrics


def _selectivity_float(options: dict[str, Any], key: str, default: float) -> float:
    value = options.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def _selectivity_int(options: dict[str, Any], key: str, default: int) -> int:
    value = options.get(key, default)
    return int(value) if isinstance(value, int) else default


def _history_lookup(
    history_rows: list[dict[str, Any]],
    generation: int,
) -> dict[str, Any] | None:
    for row in history_rows:
        if row.get("generation") == generation:
            return row
    return None


def _metric_improvement(
    *,
    baseline: float,
    candidate: float,
    lower_is_better: bool,
) -> float:
    if lower_is_better:
        return baseline - candidate
    return candidate - baseline


def _trigger_selectivity_metrics(
    problem: str,
    history_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config_data: dict[str, Any],
) -> dict[str, Any]:
    if problem != "tsp" or not history_rows:
        return {}

    options = config_data.get("algorithm_options", {})
    if not isinstance(options, dict):
        options = {}

    collapse_threshold = _selectivity_float(
        options,
        "collapse_diversity_threshold",
        _selectivity_float(options, "diversity_threshold", 0.15),
    )
    collapse_epsilon = _selectivity_float(
        options,
        "collapse_improvement_epsilon",
        _selectivity_float(options, "improvement_epsilon", 0.5),
    )
    collapse_warmup = max(
        0,
        _selectivity_int(
            options,
            "collapse_warmup_generation",
            _selectivity_int(options, "warmup_generation", 0),
        ),
    )
    useless_window = max(1, _selectivity_int(options, "useless_trigger_window", 5))
    nontrivial_improvement = _selectivity_float(
        options,
        "post_trigger_nontrivial_improvement",
        collapse_epsilon,
    )

    collapse_onset_generation: int | None = None
    for row in history_rows:
        generation = row.get("generation")
        diversity_signal = row.get("diversity_signal")
        recent_improvement = row.get("recent_window_improvement")
        if not isinstance(generation, int) or generation < collapse_warmup:
            continue
        if not isinstance(diversity_signal, int | float):
            continue
        if not isinstance(recent_improvement, int | float):
            continue
        if (
            float(diversity_signal) <= collapse_threshold
            and float(recent_improvement) <= collapse_epsilon
        ):
            collapse_onset_generation = generation
            break

    trigger_generations = summary.get("trigger_event_generations", [])
    if not isinstance(trigger_generations, list):
        trigger_generations = []
    trigger_generations = [value for value in trigger_generations if isinstance(value, int)]

    first_trigger_generation = summary.get("first_trigger_generation")
    if not isinstance(first_trigger_generation, int):
        first_trigger_generation = None
    first_switch_generation = summary.get("first_switch_generation")
    if not isinstance(first_switch_generation, int):
        first_switch_generation = None

    time_to_nontrivial_improvement: int | None = None
    if first_trigger_generation is not None:
        trigger_row = _history_lookup(history_rows, first_trigger_generation)
        if trigger_row is not None:
            baseline_distance = trigger_row.get("best_route_distance")
            if isinstance(baseline_distance, int | float):
                baseline_value = float(baseline_distance)
                for row in history_rows:
                    generation = row.get("generation")
                    candidate_distance = row.get("best_route_distance")
                    if not isinstance(generation, int) or generation <= first_trigger_generation:
                        continue
                    if not isinstance(candidate_distance, int | float):
                        continue
                    if _metric_improvement(
                        baseline=baseline_value,
                        candidate=float(candidate_distance),
                        lower_is_better=True,
                    ) >= nontrivial_improvement:
                        time_to_nontrivial_improvement = generation - first_trigger_generation
                        break

    useless_trigger_rate: float | None = None
    if trigger_generations:
        useless_triggers = 0
        for trigger_generation in trigger_generations:
            trigger_row = _history_lookup(history_rows, trigger_generation)
            baseline_distance = (
                trigger_row.get("best_route_distance") if trigger_row is not None else None
            )
            if not isinstance(baseline_distance, int | float):
                useless_triggers += 1
                continue
            improved = False
            baseline_value = float(baseline_distance)
            for row in history_rows:
                generation = row.get("generation")
                candidate_distance = row.get("best_route_distance")
                if not isinstance(generation, int):
                    continue
                if (
                    generation <= trigger_generation
                    or generation > trigger_generation + useless_window
                ):
                    continue
                if not isinstance(candidate_distance, int | float):
                    continue
                if _metric_improvement(
                    baseline=baseline_value,
                    candidate=float(candidate_distance),
                    lower_is_better=True,
                ) >= nontrivial_improvement:
                    improved = True
                    break
            if not improved:
                useless_triggers += 1
        useless_trigger_rate = useless_triggers / len(trigger_generations)

    trigger_delay_from_collapse: int | None = None
    if collapse_onset_generation is not None and first_trigger_generation is not None:
        trigger_delay_from_collapse = first_trigger_generation - collapse_onset_generation
    switch_delay_from_collapse: int | None = None
    if collapse_onset_generation is not None and first_switch_generation is not None:
        switch_delay_from_collapse = first_switch_generation - collapse_onset_generation

    return {
        "collapse_onset_generation": collapse_onset_generation,
        "trigger_delay_from_collapse": trigger_delay_from_collapse,
        "switch_delay_from_collapse": switch_delay_from_collapse,
        "time_to_first_nontrivial_improvement_after_trigger": time_to_nontrivial_improvement,
        "useless_trigger_rate": useless_trigger_rate,
        "realized_refresh_volume": summary.get("total_refresh_fraction_realized"),
    }


def _aggregate_numeric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    if fraction <= 0.0:
        return ordered[0]
    if fraction >= 1.0:
        return ordered[-1]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _qf_tolerance_config(study: LocalStudy) -> dict[str, Any] | None:
    configured = study.analysis.get("qf_tolerance")
    return configured if isinstance(configured, dict) else None


def _tsp_fast_tail_config(study: LocalStudy) -> dict[str, Any] | None:
    configured = study.analysis.get("tsp_fast_tail")
    if study.problem != "tsp":
        return None
    return configured if isinstance(configured, dict) else None


def _zdt1_fast_hardening_config(study: LocalStudy) -> dict[str, Any] | None:
    configured = study.analysis.get("zdt1_fast_hardening")
    if study.problem != "zdt1":
        return None
    return configured if isinstance(configured, dict) else None


def _zdt1_spread_candidate_validation_config(study: LocalStudy) -> dict[str, Any] | None:
    configured = study.analysis.get("zdt1_spread_candidate_boundary")
    if not isinstance(configured, dict):
        configured = study.analysis.get("zdt1_spread_candidate_validation")
    if study.problem != "zdt1":
        return None
    return configured if isinstance(configured, dict) else None


def _tsp_tail_freeze_config(study: LocalStudy) -> dict[str, Any] | None:
    configured = study.analysis.get("tsp_tail_freeze")
    if study.problem != "tsp":
        return None
    return configured if isinstance(configured, dict) else None


def _stress_suite_config(study: LocalStudy) -> dict[str, Any] | None:
    configured = study.analysis.get("stress_suite")
    return configured if isinstance(configured, dict) else None


def _stress_target_reduction_config(study: LocalStudy) -> dict[str, Any] | None:
    configured = study.analysis.get("stress_target_reduction")
    return configured if isinstance(configured, dict) else None


def _failure_trace_config(study: LocalStudy) -> dict[str, Any] | None:
    configured = study.analysis.get("failure_trace")
    return configured if isinstance(configured, dict) else None


def _plot_file_name(base_name: str, suffix: str | None = None) -> str:
    if not suffix:
        return base_name
    path = Path(base_name)
    return f"{path.stem}{suffix}{path.suffix}"


def _stress_plot_file_name(config: dict[str, Any] | None, key: str, base_name: str) -> str:
    if isinstance(config, dict):
        plot_file_names = config.get("plot_file_names")
        if isinstance(plot_file_names, dict):
            configured_name = plot_file_names.get(key)
            if isinstance(configured_name, str) and configured_name.strip():
                return configured_name.strip()
        suffix = config.get("plot_name_suffix")
        if isinstance(suffix, str) and suffix.strip():
            return _plot_file_name(base_name, suffix.strip())
    return base_name


def _stress_future_target_label(
    problem: str,
    case_group: str,
    why_selected: str,
    source_metric: str,
    row: dict[str, Any],
) -> str:
    why_lower = why_selected.lower()
    group_lower = case_group.lower()
    metric_lower = source_metric.lower()

    if problem == "tsp":
        if "anti_case" in group_lower:
            return "tsp_fast_anti_case_tail"
        if "rescue" in group_lower or "ambigu" in why_lower or "decision_flip" in why_lower:
            return "tsp_rescue_target_ambiguity"
        return "tsp_fast_general_tail"

    if problem == "zdt1":
        joint_flag = _stress_numeric(row.get("joint_safety_fail")) or 0.0
        spread_flag = _stress_numeric(row.get("spread_safety_fail")) or 0.0
        pareto_flag = _stress_numeric(row.get("pareto_ratio_safety_fail")) or 0.0
        if joint_flag > 0.0 or "joint" in why_lower:
            return "zdt1_fast_joint_safety_fail"
        if spread_flag > 0.0 or pareto_flag > 0.0 or "spread" in why_lower or "safety" in group_lower:
            return "zdt1_fast_spread_safety_fail"
        if "ambigu" in why_lower or "decision" in why_lower:
            return "zdt1_fast_joint_safety_fail"
        if metric_lower == "hv_loss_pct" or "hv" in group_lower:
            return "zdt1_fast_hv_tail"
        return "zdt1_fast_joint_safety_fail"

    if problem == "knapsack":
        if group_lower in {"subset_sum_like_small", "tight_capacity_small"}:
            return "knapsack_repair_boundary_subset_sum_tight_capacity"
        if "borderline" in why_lower or group_lower in {"weakly_correlated_small", "uncorrelated_small"}:
            return "knapsack_repair_borderline_family"
        return "knapsack_repair_note_regression"

    if problem == "onemax":
        return "onemax_no_active_target"

    return f"{problem}_future_target"


def _study_variant_key(row: dict[str, Any]) -> str:
    variant = row.get("study_variant")
    if isinstance(variant, str) and variant.strip():
        return variant.strip()
    label = row.get("variant_label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return ""


def _pairwise_group_id(row: dict[str, Any]) -> tuple[str, int] | None:
    seed = row.get("seed")
    if not isinstance(seed, int):
        return None
    case_id = row.get("case_id")
    if isinstance(case_id, str) and case_id.strip():
        return (case_id.strip(), seed)
    return ("__study__", seed)


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _bootstrap_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 400,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        bootstrap_means.append(mean(sample))
    bootstrap_means.sort()
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    low_index = int((alpha / 2.0) * (len(bootstrap_means) - 1))
    high_index = int((1.0 - (alpha / 2.0)) * (len(bootstrap_means) - 1))
    return bootstrap_means[low_index], bootstrap_means[high_index]


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), int | float) and not isinstance(row.get(key), bool)
    ]


def _win_rate(values: list[float], *, epsilon: float = 1e-9) -> float | None:
    if not values:
        return None
    return mean(1.0 if value < -epsilon else 0.0 for value in values)


def _tsp_policy_name(row: dict[str, Any]) -> str:
    parameter_policy = row.get("algorithm_options.adaptive_policy")
    if isinstance(parameter_policy, str) and parameter_policy.strip():
        return parameter_policy.strip()
    summary_policy = row.get("adaptive_policy")
    if isinstance(summary_policy, str) and summary_policy.strip():
        return summary_policy.strip()
    return "none"


def _is_current_tsp_preferred_profile(row: dict[str, Any]) -> bool:
    return (
        row.get("algorithm") == "hybrid_ga"
        and row.get("mutation") == "swap"
        and row.get("algorithm_options.adaptive_policy") == "none"
        and row.get("algorithm_options.init_strategy") == "tsp_nearest_neighbor_mix"
        and row.get("algorithm_options.local_search_strategy") == "none"
        and row.get("algorithm_options.seed_fraction") == 0.5
    )


def _is_tsp_fixed_policy_reference(row: dict[str, Any]) -> bool:
    return row.get("algorithm") in {"ga", "hybrid_ga"}


def _is_current_zdt1_default_profile(row: dict[str, Any]) -> bool:
    threshold = row.get("algorithm_options.diversity_threshold")
    refresh_fraction = row.get("algorithm_options.refresh_fraction")
    cooldown = row.get("algorithm_options.adaptation_cooldown")
    return (
        row.get("algorithm") == "nsga2"
        and row.get("algorithm_options.adaptive_policy") == "low_diversity_injection"
        and isinstance(threshold, int | float)
        and isinstance(refresh_fraction, int | float)
        and isinstance(cooldown, int | float)
        and math.isclose(float(threshold), 0.55, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(float(refresh_fraction), 0.1, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(float(cooldown), 4.0, rel_tol=0.0, abs_tol=1e-9)
    )


def _tsp_regret_value(row: dict[str, Any]) -> float | None:
    for key in (
        "regret_vs_current_preferred_profile",
        "regret_vs_oracle_tested_candidate",
        "regret_vs_oracle_fixed_policy",
    ):
        value = row.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _tsp_regret_group_key(row: dict[str, Any]) -> str | None:
    case_id = row.get("case_id")
    if isinstance(case_id, str) and case_id.strip():
        return case_id.strip()
    return "__study__"


def _annotate_tsp_regret_rows(raw_rows: list[dict[str, Any]]) -> None:
    if not raw_rows:
        return

    tested_means: dict[tuple[str, float], dict[str, float]] = {}
    tested_grouped: dict[tuple[str, float, str], list[float]] = {}
    fixed_means: dict[tuple[str, float], dict[str, float]] = {}
    grouped: dict[tuple[str, float, str], list[float]] = {}
    preferred_grouped: dict[tuple[str, float], list[float]] = {}
    for row in raw_rows:
        group_key = _tsp_regret_group_key(row)
        configured_budget = row.get("configured_budget")
        distance = row.get("best_route_distance")
        if group_key is None or not isinstance(configured_budget, int | float):
            continue
        if not isinstance(distance, int | float):
            continue
        tested_grouped.setdefault(
            (group_key, float(configured_budget), str(row["variant_label"])),
            [],
        ).append(float(distance))
        policy = _tsp_policy_name(row)
        if _is_current_tsp_preferred_profile(row):
            preferred_grouped.setdefault((group_key, float(configured_budget)), []).append(
                float(distance)
            )
        if (
            policy not in {"none", "low_diversity_injection", "decay_mutation"}
            or not _is_tsp_fixed_policy_reference(row)
        ):
            continue
        grouped.setdefault((group_key, float(configured_budget), policy), []).append(
            float(distance)
        )

    for (group_key, configured_budget, variant_label), values in tested_grouped.items():
        tested_means.setdefault((group_key, configured_budget), {})[variant_label] = mean(values)

    for (group_key, configured_budget, policy), values in grouped.items():
        fixed_means.setdefault((group_key, configured_budget), {})[policy] = mean(values)

    preferred_means = {
        key: mean(values)
        for key, values in preferred_grouped.items()
        if values
    }

    for row in raw_rows:
        group_key = _tsp_regret_group_key(row)
        configured_budget = row.get("configured_budget")
        distance = row.get("best_route_distance")
        if group_key is None or not isinstance(configured_budget, int | float):
            continue
        if not isinstance(distance, int | float):
            continue
        tested = tested_means.get((group_key, float(configured_budget)))
        if tested:
            oracle_variant, oracle_distance = min(tested.items(), key=lambda item: item[1])
            row["oracle_tested_variant"] = oracle_variant
            row["oracle_tested_distance"] = oracle_distance
            row["regret_vs_oracle_tested_candidate"] = float(distance) - oracle_distance
        preferred_distance = preferred_means.get((group_key, float(configured_budget)))
        if preferred_distance is not None:
            row["regret_vs_current_preferred_profile"] = float(distance) - preferred_distance
        group_means = fixed_means.get((group_key, float(configured_budget)))
        if not group_means:
            continue
        oracle_policy, oracle_distance = min(group_means.items(), key=lambda item: item[1])
        row["oracle_fixed_policy"] = oracle_policy
        row["oracle_fixed_policy_distance"] = oracle_distance
        row["regret_vs_oracle_fixed_policy"] = float(distance) - oracle_distance
        if "decay_mutation" in group_means:
            row["regret_vs_always_decay_mutation"] = (
                float(distance) - group_means["decay_mutation"]
            )
        if "low_diversity_injection" in group_means:
            row["regret_vs_always_low_diversity_injection"] = (
                float(distance) - group_means["low_diversity_injection"]
            )
        if "none" in group_means:
            row["regret_vs_always_none"] = float(distance) - group_means["none"]


def _annotate_tsp_regret_summary(
    summary_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> None:
    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        rows_by_label.setdefault(str(row["variant_label"]), []).append(row)

    for summary_row in summary_rows:
        label = str(summary_row["variant_label"])
        rows = rows_by_label.get(label, [])
        oracle_policies = {
            str(row["oracle_fixed_policy"])
            for row in rows
            if isinstance(row.get("oracle_fixed_policy"), str)
        }
        if len(oracle_policies) == 1:
            summary_row["oracle_fixed_policy"] = next(iter(oracle_policies))
        for metric_name in (
            "regret_vs_oracle_tested_candidate",
            "regret_vs_current_preferred_profile",
            "regret_vs_oracle_fixed_policy",
            "regret_vs_always_decay_mutation",
            "regret_vs_always_low_diversity_injection",
            "regret_vs_always_none",
        ):
            stats = _aggregate_numeric(_numeric_values(rows, metric_name))
            summary_row[f"{metric_name}_mean"] = stats["mean"]
            summary_row[f"{metric_name}_std"] = stats["std"]
        summary_row["win_rate_vs_decay_mutation"] = _win_rate(
            _numeric_values(rows, "regret_vs_always_decay_mutation")
        )
        summary_row["win_rate_vs_low_diversity_injection"] = _win_rate(
            _numeric_values(rows, "regret_vs_always_low_diversity_injection")
        )
        summary_row["win_rate_vs_none"] = _win_rate(
            _numeric_values(rows, "regret_vs_always_none")
        )


def _annotate_zdt1_regret_summary(summary_rows: list[dict[str, Any]]) -> None:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in summary_rows:
        configured_budget = row.get("configured_budget")
        if isinstance(configured_budget, int | float):
            grouped.setdefault(float(configured_budget), []).append(row)

    for rows in grouped.values():
        current_profile_hv = None
        none_hv = None
        for row in rows:
            hypervolume = row.get("hypervolume_mean")
            if not isinstance(hypervolume, int | float):
                continue
            if _is_current_zdt1_default_profile(row):
                current_profile_hv = float(hypervolume)
            if row.get("algorithm_options.adaptive_policy") == "none":
                none_hv = float(hypervolume)

        for row in rows:
            hypervolume = row.get("hypervolume_mean")
            if not isinstance(hypervolume, int | float):
                continue
            hv_value = float(hypervolume)
            if current_profile_hv is not None:
                row["reference_current_profile_hv"] = current_profile_hv
                row["regret_vs_current_profile_mean"] = current_profile_hv - hv_value
            if none_hv is not None:
                row["reference_none_hv"] = none_hv
                row["regret_vs_none_mean"] = none_hv - hv_value


def _portfolio_group_key(row: dict[str, Any]) -> tuple[str, int] | None:
    seed = row.get("seed")
    if not isinstance(seed, int | float) or isinstance(seed, bool):
        return None
    case_id = row.get("case_id")
    case_key = case_id.strip() if isinstance(case_id, str) and case_id.strip() else "__study__"
    return (case_key, int(seed))


def _portfolio_primary_value(problem: str, row: dict[str, Any]) -> float | None:
    if problem == "tsp":
        value = row.get("best_route_distance")
    elif problem == "zdt1":
        value = row.get("hypervolume")
    elif problem == "knapsack":
        value = row.get("best_feasible_fitness")
    else:
        value = row.get("evaluations_to_target")
        if value is None:
            value = row.get("best_fitness")
            if isinstance(value, int | float) and not isinstance(value, bool):
                value = -float(value)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _annotate_restart_portfolio_rows(problem: str, raw_rows: list[dict[str, Any]]) -> None:
    if problem not in {"tsp", "zdt1", "knapsack"} or not raw_rows:
        return

    references: dict[tuple[str, int], dict[str, float]] = {}
    lower_is_better = problem == "tsp"
    for row in raw_rows:
        group_key = _portfolio_group_key(row)
        study_variant = row.get("study_variant")
        metric_value = _portfolio_primary_value(problem, row)
        if group_key is None or not isinstance(study_variant, str) or metric_value is None:
            continue
        label = study_variant.strip()
        if label == "canonical_once":
            references.setdefault(group_key, {})["canonical_once"] = metric_value
            row["portfolio_restart_count"] = 1
        elif label == "fast_once":
            references.setdefault(group_key, {})["fast_once"] = metric_value
            row["portfolio_restart_count"] = 1
        elif label == "none":
            references.setdefault(group_key, {})["none"] = metric_value
            row.setdefault("portfolio_restart_count", 1)
        elif label == "repair_only":
            references.setdefault(group_key, {})["repair_only"] = metric_value
            row.setdefault("portfolio_restart_count", 1)
        elif label == "greedy_local_search":
            references.setdefault(group_key, {})["greedy_local_search"] = metric_value
            row.setdefault("portfolio_restart_count", 1)

    for row in raw_rows:
        group_key = _portfolio_group_key(row)
        metric_value = _portfolio_primary_value(problem, row)
        if group_key is None or metric_value is None:
            continue
        reference_values = references.get(group_key, {})
        canonical_value = reference_values.get("canonical_once")
        fast_value = reference_values.get("fast_once")
        none_value = reference_values.get("none")
        repair_value = reference_values.get("repair_only")
        if canonical_value is not None:
            row["regret_vs_canonical_once"] = (
                metric_value - canonical_value
                if lower_is_better
                else canonical_value - metric_value
            )
        if fast_value is not None:
            row["regret_vs_fast_once"] = (
                metric_value - fast_value if lower_is_better else fast_value - metric_value
            )
            if problem == "tsp":
                row["best_of_k_improvement_over_single_fast"] = fast_value - metric_value
            elif problem == "zdt1":
                row["merged_archive_gain_over_single_fast"] = metric_value - fast_value
        if none_value is not None and problem == "knapsack":
            row["regret_vs_multi_none_reference"] = none_value - metric_value
        if repair_value is not None and problem == "knapsack":
            row["regret_vs_repair_only_reference"] = repair_value - metric_value


def _qf_pair_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _qf_tolerance_config(study)
    if config is None or study.problem not in {"tsp", "zdt1"}:
        return []

    quality_variant = config.get("quality_variant")
    fast_variant = config.get("fast_variant")
    if not isinstance(quality_variant, str) or not quality_variant.strip():
        return []
    if not isinstance(fast_variant, str) or not fast_variant.strip():
        return []
    quality_variant = quality_variant.strip()
    fast_variant = fast_variant.strip()

    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in raw_rows:
        variant = _study_variant_key(row)
        if variant not in {quality_variant, fast_variant}:
            continue
        group_id = _pairwise_group_id(row)
        if group_id is None:
            continue
        grouped.setdefault(group_id, {})[variant] = row

    pareto_drop_threshold = config.get("pareto_ratio_drop_threshold", 0.01)
    spread_degradation_threshold = config.get("spread_degradation_threshold", 0.05)
    pair_rows: list[dict[str, Any]] = []
    for (_case_id, _seed), variants in sorted(grouped.items(), key=lambda item: item[0]):
        quality_row = variants.get(quality_variant)
        fast_row = variants.get(fast_variant)
        if quality_row is None or fast_row is None:
            continue
        seed = quality_row.get("seed")
        case_id = quality_row.get("case_id") or fast_row.get("case_id") or ""
        case_group = quality_row.get("case_group") or fast_row.get("case_group") or "overall"
        case_note = quality_row.get("case_note") or fast_row.get("case_note") or ""
        actual_q = quality_row.get("actual_evaluations_used")
        actual_f = fast_row.get("actual_evaluations_used")
        runtime_q = quality_row.get("runtime_seconds")
        runtime_f = fast_row.get("runtime_seconds")
        actual_eval_savings_pct = (
            (float(actual_q) - float(actual_f)) / float(actual_q) * 100.0
            if isinstance(actual_q, int | float)
            and not isinstance(actual_q, bool)
            and float(actual_q) != 0.0
            and isinstance(actual_f, int | float)
            and not isinstance(actual_f, bool)
            else None
        )
        runtime_savings_pct = (
            (float(runtime_q) - float(runtime_f)) / float(runtime_q) * 100.0
            if isinstance(runtime_q, int | float)
            and not isinstance(runtime_q, bool)
            and float(runtime_q) != 0.0
            and isinstance(runtime_f, int | float)
            and not isinstance(runtime_f, bool)
            else None
        )
        base_row = {
            "seed": seed,
            "case_id": case_id,
            "case_group": case_group,
            "case_note": case_note,
            "quality_variant": quality_variant,
            "fast_variant": fast_variant,
            "quality_configured_budget": quality_row.get("configured_budget"),
            "fast_configured_budget": fast_row.get("configured_budget"),
            "quality_actual_evaluations_used": actual_q,
            "fast_actual_evaluations_used": actual_f,
            "quality_runtime_seconds": runtime_q,
            "fast_runtime_seconds": runtime_f,
            "actual_eval_savings_pct": actual_eval_savings_pct,
            "runtime_savings_pct": runtime_savings_pct,
        }
        if study.problem == "tsp":
            q_metric = quality_row.get("best_route_distance")
            f_metric = fast_row.get("best_route_distance")
            if not isinstance(q_metric, int | float) or isinstance(q_metric, bool):
                continue
            if not isinstance(f_metric, int | float) or isinstance(f_metric, bool):
                continue
            if float(q_metric) == 0.0:
                continue
            loss_pct = (float(f_metric) - float(q_metric)) / float(q_metric) * 100.0
            if abs(loss_pct) <= 1e-9:
                outcome = "tie"
            elif loss_pct < 0.0:
                outcome = "fast_better"
            else:
                outcome = "quality_better"
            pair_rows.append(
                {
                    **base_row,
                    "quality_metric": float(q_metric),
                    "fast_metric": float(f_metric),
                    "route_distance_loss_pct": loss_pct,
                    "comparison_outcome": outcome,
                }
            )
            continue

        q_hv = quality_row.get("hypervolume")
        f_hv = fast_row.get("hypervolume")
        if not isinstance(q_hv, int | float) or isinstance(q_hv, bool):
            continue
        if not isinstance(f_hv, int | float) or isinstance(f_hv, bool):
            continue
        if float(q_hv) == 0.0:
            continue
        hv_loss_pct = (float(q_hv) - float(f_hv)) / float(q_hv) * 100.0
        q_pareto = quality_row.get("pareto_ratio")
        f_pareto = fast_row.get("pareto_ratio")
        q_spread = quality_row.get("spread")
        f_spread = fast_row.get("spread")
        pareto_ratio_delta = (
            float(f_pareto) - float(q_pareto)
            if isinstance(q_pareto, int | float)
            and not isinstance(q_pareto, bool)
            and isinstance(f_pareto, int | float)
            and not isinstance(f_pareto, bool)
            else None
        )
        spread_delta = (
            float(f_spread) - float(q_spread)
            if isinstance(q_spread, int | float)
            and not isinstance(q_spread, bool)
            and isinstance(f_spread, int | float)
            and not isinstance(f_spread, bool)
            else None
        )
        pareto_fail = (
            pareto_ratio_delta is not None
            and isinstance(pareto_drop_threshold, int | float)
            and pareto_ratio_delta < -float(pareto_drop_threshold)
        )
        spread_fail = (
            spread_delta is not None
            and isinstance(spread_degradation_threshold, int | float)
            and spread_delta > float(spread_degradation_threshold)
        )
        pair_rows.append(
            {
                **base_row,
                "quality_metric": float(q_hv),
                "fast_metric": float(f_hv),
                "hv_loss_pct": hv_loss_pct,
                "pareto_ratio_delta": pareto_ratio_delta,
                "spread_delta": spread_delta,
                "pareto_ratio_safety_fail": 1.0 if pareto_fail else 0.0,
                "spread_safety_fail": 1.0 if spread_fail else 0.0,
                "joint_safety_fail": 1.0 if pareto_fail or spread_fail else 0.0,
            }
        )
    return pair_rows


def _qf_tolerance_rows(
    study: LocalStudy,
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _qf_tolerance_config(study)
    if config is None or not pair_rows:
        return []
    tolerance_bins = config.get("tolerance_bins_pct", [0.1, 0.25, 0.5, 1.0])
    if not isinstance(tolerance_bins, list):
        return []
    tolerance_values = [
        float(value)
        for value in tolerance_bins
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    if not tolerance_values:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {"overall": pair_rows}
    if study.problem == "tsp":
        for case_group in sorted({str(row.get("case_group") or "overall") for row in pair_rows}):
            grouped[case_group] = [
                row for row in pair_rows if str(row.get("case_group") or "overall") == case_group
            ]

    rows: list[dict[str, Any]] = []
    for scope_key, members in grouped.items():
        if not members:
            continue
        if study.problem == "tsp":
            losses = [float(row["route_distance_loss_pct"]) for row in members]
            fast_better_rate = _mean_or_none(
                [1.0 if row.get("comparison_outcome") == "fast_better" else 0.0 for row in members]
            )
            tie_rate = _mean_or_none(
                [1.0 if row.get("comparison_outcome") == "tie" else 0.0 for row in members]
            )
            quality_better_rate = _mean_or_none(
                [1.0 if row.get("comparison_outcome") == "quality_better" else 0.0 for row in members]
            )
            for tolerance in tolerance_values:
                acceptable_rate = _mean_or_none(
                    [1.0 if float(row["route_distance_loss_pct"]) <= tolerance else 0.0 for row in members]
                )
                rows.append(
                    {
                        "scope": "overall" if scope_key == "overall" else "case_group",
                        "case_group": scope_key,
                        "run_count": len(members),
                        "tolerance_bin_pct": tolerance,
                        "acceptable_rate": acceptable_rate,
                "mean_loss_pct": mean(losses),
                "median_loss_pct": median(losses),
                "p75_loss_pct": _quantile(losses, 0.75),
                "p90_loss_pct": _quantile(losses, 0.9),
                "p95_loss_pct": _quantile(losses, 0.95),
                "max_loss_pct": max(losses),
                        "actual_eval_savings_pct_mean": _mean_or_none(
                            [
                                float(value)
                                for row in members
                                for value in [row.get("actual_eval_savings_pct")]
                                if isinstance(value, int | float)
                            ]
                        ),
                        "runtime_savings_pct_mean": _mean_or_none(
                            [
                                float(value)
                                for row in members
                                for value in [row.get("runtime_savings_pct")]
                                if isinstance(value, int | float)
                            ]
                        ),
                        "fast_better_rate": fast_better_rate,
                        "tie_rate": tie_rate,
                        "quality_better_rate": quality_better_rate,
                    }
                )
            continue

        hv_losses = [float(row["hv_loss_pct"]) for row in members]
        pareto_deltas = [
            float(value)
            for row in members
            for value in [row.get("pareto_ratio_delta")]
            if isinstance(value, int | float)
        ]
        spread_deltas = [
            float(value)
            for row in members
            for value in [row.get("spread_delta")]
            if isinstance(value, int | float)
        ]
        pareto_fail_rate = _mean_or_none(
            [float(row.get("pareto_ratio_safety_fail", 0.0)) for row in members]
        )
        spread_fail_rate = _mean_or_none(
            [float(row.get("spread_safety_fail", 0.0)) for row in members]
        )
        joint_fail_rate = _mean_or_none(
            [float(row.get("joint_safety_fail", 0.0)) for row in members]
        )
        for tolerance in tolerance_values:
            hv_only_accept_rate = _mean_or_none(
                [1.0 if float(row["hv_loss_pct"]) <= tolerance else 0.0 for row in members]
            )
            acceptable_rate = _mean_or_none(
                [
                    1.0
                    if float(row["hv_loss_pct"]) <= tolerance
                    and float(row.get("joint_safety_fail", 0.0)) < 0.5
                    else 0.0
                    for row in members
                ]
            )
            rows.append(
                {
                    "scope": "overall",
                    "case_group": "overall",
                    "run_count": len(members),
                    "tolerance_bin_pct": tolerance,
                    "hv_only_accept_rate": hv_only_accept_rate,
                    "acceptable_rate": acceptable_rate,
                    "mean_loss_pct": mean(hv_losses),
                    "median_loss_pct": median(hv_losses),
                    "p75_loss_pct": _quantile(hv_losses, 0.75),
                    "p90_loss_pct": _quantile(hv_losses, 0.9),
                    "max_loss_pct": max(hv_losses),
                    "actual_eval_savings_pct_mean": _mean_or_none(
                        [
                            float(value)
                            for row in members
                            for value in [row.get("actual_eval_savings_pct")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "runtime_savings_pct_mean": _mean_or_none(
                        [
                            float(value)
                            for row in members
                            for value in [row.get("runtime_savings_pct")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "pareto_ratio_delta_mean": _mean_or_none(pareto_deltas),
                    "spread_delta_mean": _mean_or_none(spread_deltas),
                    "pareto_ratio_fail_rate": pareto_fail_rate,
                    "spread_fail_rate": spread_fail_rate,
                    "joint_safety_fail_rate": joint_fail_rate,
                }
            )
    return rows


def _seed_budget_config(study: LocalStudy) -> dict[str, Any] | None:
    analysis = study.analysis
    if not isinstance(analysis, dict):
        return None
    config = analysis.get("seed_budget")
    return config if isinstance(config, dict) else None


def _stress_numeric(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    coerced = float(value)
    if not math.isfinite(coerced):
        return None
    return coerced


def _stress_case_count(rows: list[dict[str, Any]]) -> int:
    return len(
        {
            str(row["case_id"])
            for row in rows
            if isinstance(row.get("case_id"), str) and str(row["case_id"]).strip()
        }
    )


def _stress_case_catalog_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
    qf_pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _stress_suite_config(study)
    if config is None:
        return []

    budget_band = str(config.get("budget_band_label") or "current_local_default")
    catalog_top_overall = config.get("catalog_top_overall", 3)
    if not isinstance(catalog_top_overall, int) or isinstance(catalog_top_overall, bool):
        catalog_top_overall = 3
    catalog_top_group = config.get("catalog_top_group", 2)
    if not isinstance(catalog_top_group, int) or isinstance(catalog_top_group, bool):
        catalog_top_group = 2
    ambiguity_band = _stress_numeric(config.get("ambiguity_band_pct")) or 0.5
    hv_boundary = _stress_numeric(config.get("hv_boundary_pct")) or 0.5

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None, str]] = set()

    def _append_catalog_row(
        *,
        case_id: str,
        instance_label: str,
        seed: int | None,
        profile_compared: str,
        regret_or_loss: float | None,
        why_selected_as_stress_case: str,
        case_group: str,
        source_metric: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        dedupe_key = (case_id, profile_compared, seed, why_selected_as_stress_case)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        row: dict[str, Any] = {
            "problem": study.problem,
            "case_id": case_id,
            "instance_label": instance_label,
            "seed": seed,
            "budget_band": budget_band,
            "profile_compared": profile_compared,
            "compared_profiles": profile_compared,
            "regret_or_loss": regret_or_loss,
            "why_selected_as_stress_case": why_selected_as_stress_case,
            "why_selected": why_selected_as_stress_case,
            "case_group": case_group,
            "source_metric": source_metric,
        }
        if extra:
            row.update(extra)
        row["future_target_label"] = _stress_future_target_label(
            study.problem,
            case_group,
            why_selected_as_stress_case,
            source_metric,
            row,
        )
        rows.append(row)

    if study.problem == "tsp" and qf_pair_rows:
        qf_config = _qf_tolerance_config(study) or {}
        quality_variant = str(qf_config.get("quality_variant") or "quality_first")
        fast_variant = str(qf_config.get("fast_variant") or "budget_first")
        comparison_label = f"{fast_variant}_vs_{quality_variant}"
        loss_rows = [
            row
            for row in qf_pair_rows
            if _stress_numeric(row.get("route_distance_loss_pct")) is not None
        ]
        loss_rows.sort(
            key=lambda row: _stress_numeric(row.get("route_distance_loss_pct")) or -1e18,
            reverse=True,
        )
        for row in loss_rows[: max(catalog_top_overall, 1)]:
            loss_pct = _stress_numeric(row.get("route_distance_loss_pct"))
            _append_catalog_row(
                case_id=str(row.get("case_id") or f"tsp_seed_{row.get('seed')}"),
                instance_label=str(row.get("case_id") or f"tsp_seed_{row.get('seed')}"),
                seed=int(row["seed"]) if isinstance(row.get("seed"), int) else None,
                profile_compared=comparison_label,
                regret_or_loss=loss_pct,
                why_selected_as_stress_case="highest_qf_loss",
                case_group=str(row.get("case_group") or "overall"),
                source_metric="route_distance_loss_pct",
                extra={
                    "quality_metric": row.get("quality_metric"),
                    "fast_metric": row.get("fast_metric"),
                    "comparison_outcome": row.get("comparison_outcome"),
                },
            )
        for group in sorted({str(row.get("case_group") or "overall") for row in loss_rows}):
            group_rows = [
                row for row in loss_rows if str(row.get("case_group") or "overall") == group
            ]
            for row in group_rows[: max(catalog_top_group, 1)]:
                _append_catalog_row(
                    case_id=str(row.get("case_id") or f"tsp_seed_{row.get('seed')}"),
                    instance_label=str(row.get("case_id") or f"tsp_seed_{row.get('seed')}"),
                    seed=int(row["seed"]) if isinstance(row.get("seed"), int) else None,
                    profile_compared=comparison_label,
                    regret_or_loss=_stress_numeric(row.get("route_distance_loss_pct")),
                    why_selected_as_stress_case=f"{group}_tail",
                    case_group=group,
                    source_metric="route_distance_loss_pct",
                    extra={
                        "quality_metric": row.get("quality_metric"),
                        "fast_metric": row.get("fast_metric"),
                        "comparison_outcome": row.get("comparison_outcome"),
                    },
                )
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in loss_rows:
            by_case.setdefault(str(row.get("case_id") or ""), []).append(row)
        for case_id, members in sorted(by_case.items()):
            signs = {
                "positive"
                if (_stress_numeric(row.get("route_distance_loss_pct")) or 0.0) > 0.0
                else "negative"
                if (_stress_numeric(row.get("route_distance_loss_pct")) or 0.0) < 0.0
                else "zero"
                for row in members
            }
            if len(signs) <= 1:
                continue
            pivot = min(
                members,
                key=lambda row: abs(_stress_numeric(row.get("route_distance_loss_pct")) or 0.0),
            )
            _append_catalog_row(
                case_id=case_id,
                instance_label=case_id,
                seed=int(pivot["seed"]) if isinstance(pivot.get("seed"), int) else None,
                profile_compared=comparison_label,
                regret_or_loss=_stress_numeric(pivot.get("route_distance_loss_pct")),
                why_selected_as_stress_case="qf_decision_flip",
                case_group=str(pivot.get("case_group") or "overall"),
                source_metric="route_distance_loss_pct",
                extra={
                    "quality_metric": pivot.get("quality_metric"),
                    "fast_metric": pivot.get("fast_metric"),
                    "comparison_outcome": pivot.get("comparison_outcome"),
                },
            )
            rescue_members = [
                row
                for row in members
                if str(row.get("case_group") or "overall") == "rescue_target"
                and abs(_stress_numeric(row.get("route_distance_loss_pct")) or 0.0)
                <= ambiguity_band
            ]
            if rescue_members:
                pivot = min(
                    rescue_members,
                    key=lambda row: abs(
                        _stress_numeric(row.get("route_distance_loss_pct")) or 0.0
                    ),
                )
                _append_catalog_row(
                    case_id=case_id,
                    instance_label=case_id,
                    seed=int(pivot["seed"]) if isinstance(pivot.get("seed"), int) else None,
                    profile_compared=comparison_label,
                    regret_or_loss=_stress_numeric(pivot.get("route_distance_loss_pct")),
                    why_selected_as_stress_case="rescue_target_ambiguity",
                    case_group="rescue_target",
                    source_metric="route_distance_loss_pct",
                    extra={
                        "quality_metric": pivot.get("quality_metric"),
                        "fast_metric": pivot.get("fast_metric"),
                        "comparison_outcome": pivot.get("comparison_outcome"),
                    },
                )
        return rows

    if study.problem == "zdt1" and qf_pair_rows:
        qf_config = _qf_tolerance_config(study) or {}
        quality_variant = str(qf_config.get("quality_variant") or "quality_first")
        fast_variant = str(qf_config.get("fast_variant") or "budget_first")
        comparison_label = f"{fast_variant}_vs_{quality_variant}"
        hv_rows = [
            row for row in qf_pair_rows if _stress_numeric(row.get("hv_loss_pct")) is not None
        ]
        hv_rows.sort(
            key=lambda row: _stress_numeric(row.get("hv_loss_pct")) or -1e18,
            reverse=True,
        )
        for row in hv_rows[: max(catalog_top_overall, 1)]:
            seed = int(row["seed"]) if isinstance(row.get("seed"), int) else None
            case_id = f"zdt1_seed_{seed}" if seed is not None else "zdt1_seed"
            _append_catalog_row(
                case_id=case_id,
                instance_label=case_id,
                seed=seed,
                profile_compared=comparison_label,
                regret_or_loss=_stress_numeric(row.get("hv_loss_pct")),
                why_selected_as_stress_case="highest_hv_loss",
                case_group="hv_tail",
                source_metric="hv_loss_pct",
                extra={
                    "quality_metric": row.get("quality_metric"),
                    "fast_metric": row.get("fast_metric"),
                    "pareto_ratio_delta": row.get("pareto_ratio_delta"),
                    "spread_delta": row.get("spread_delta"),
                    "pareto_ratio_safety_fail": row.get("pareto_ratio_safety_fail"),
                    "spread_safety_fail": row.get("spread_safety_fail"),
                },
            )
        spread_fail_rows = [
            row for row in hv_rows if _stress_numeric(row.get("spread_safety_fail")) == 1.0
        ]
        spread_fail_rows.sort(
            key=lambda row: _stress_numeric(row.get("spread_delta")) or -1e18,
            reverse=True,
        )
        for row in spread_fail_rows[: max(catalog_top_group, 1)]:
            seed = int(row["seed"]) if isinstance(row.get("seed"), int) else None
            case_id = f"zdt1_seed_{seed}" if seed is not None else "zdt1_seed"
            _append_catalog_row(
                case_id=case_id,
                instance_label=case_id,
                seed=seed,
                profile_compared=comparison_label,
                regret_or_loss=_stress_numeric(row.get("hv_loss_pct")),
                why_selected_as_stress_case="spread_safety_fail",
                case_group="safety_fail",
                source_metric="spread_delta",
                extra={
                    "spread_delta": row.get("spread_delta"),
                    "pareto_ratio_delta": row.get("pareto_ratio_delta"),
                    "joint_safety_fail": row.get("joint_safety_fail"),
                },
            )
        pareto_fail_rows = [
            row
            for row in hv_rows
            if _stress_numeric(row.get("pareto_ratio_safety_fail")) == 1.0
        ]
        pareto_fail_rows.sort(
            key=lambda row: _stress_numeric(row.get("pareto_ratio_delta")) or 1e18
        )
        for row in pareto_fail_rows[: max(catalog_top_group, 1)]:
            seed = int(row["seed"]) if isinstance(row.get("seed"), int) else None
            case_id = f"zdt1_seed_{seed}" if seed is not None else "zdt1_seed"
            _append_catalog_row(
                case_id=case_id,
                instance_label=case_id,
                seed=seed,
                profile_compared=comparison_label,
                regret_or_loss=_stress_numeric(row.get("hv_loss_pct")),
                why_selected_as_stress_case="pareto_ratio_safety_fail",
                case_group="safety_fail",
                source_metric="pareto_ratio_delta",
                extra={
                    "spread_delta": row.get("spread_delta"),
                    "pareto_ratio_delta": row.get("pareto_ratio_delta"),
                    "joint_safety_fail": row.get("joint_safety_fail"),
                },
            )
        boundary_rows = [
            row
            for row in hv_rows
            if abs(_stress_numeric(row.get("hv_loss_pct")) or 0.0) <= hv_boundary
        ]
        boundary_rows.sort(key=lambda row: abs(_stress_numeric(row.get("hv_loss_pct")) or 0.0))
        for row in boundary_rows[: max(catalog_top_group, 1)]:
            seed = int(row["seed"]) if isinstance(row.get("seed"), int) else None
            case_id = f"zdt1_seed_{seed}" if seed is not None else "zdt1_seed"
            _append_catalog_row(
                case_id=case_id,
                instance_label=case_id,
                seed=seed,
                profile_compared=comparison_label,
                regret_or_loss=_stress_numeric(row.get("hv_loss_pct")),
                why_selected_as_stress_case="decision_boundary",
                case_group="decision_boundary",
                source_metric="hv_loss_pct",
                extra={
                    "spread_delta": row.get("spread_delta"),
                    "pareto_ratio_delta": row.get("pareto_ratio_delta"),
                    "joint_safety_fail": row.get("joint_safety_fail"),
                },
            )
        return rows

    if study.problem == "knapsack" and raw_rows:
        default_variant = str(config.get("default_variant") or "greedy_local_search")
        baseline_variant = str(config.get("baseline_variant") or "none")
        repair_variant = str(config.get("repair_variant") or "repair_only")
        note_groups = {
            str(value)
            for value in config.get("note_groups", [])
            if isinstance(value, str) and value.strip()
        }
        borderline_groups = {
            str(value)
            for value in config.get("borderline_groups", [])
            if isinstance(value, str) and value.strip()
        }
        grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        for row in raw_rows:
            variant = _study_variant_key(row)
            case_id = str(row.get("case_id") or "")
            seed = row.get("seed")
            if variant not in {default_variant, baseline_variant, repair_variant}:
                continue
            if not case_id or not isinstance(seed, int):
                continue
            grouped.setdefault((case_id, seed), {})[variant] = row
        candidate_rows: list[dict[str, Any]] = []
        for (case_id, seed), variants in sorted(grouped.items()):
            default_row = variants.get(default_variant)
            baseline_row = variants.get(baseline_variant)
            repair_row = variants.get(repair_variant)
            if default_row is None or baseline_row is None or repair_row is None:
                continue
            default_metric = _stress_numeric(default_row.get("best_feasible_fitness"))
            baseline_metric = _stress_numeric(baseline_row.get("best_feasible_fitness"))
            repair_metric = _stress_numeric(repair_row.get("best_feasible_fitness"))
            if default_metric is None or baseline_metric is None or repair_metric is None:
                continue
            candidate_rows.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "case_group": str(repair_row.get("case_group") or "overall"),
                    "repair_gain_vs_none": repair_metric - baseline_metric,
                    "repair_gap_vs_greedy": repair_metric - default_metric,
                    "repair_feasible_ratio": repair_row.get("feasible_ratio"),
                    "repair_generations_to_first_feasible": repair_row.get(
                        "generations_to_first_feasible"
                    ),
                }
            )
        strongest = sorted(
            candidate_rows,
            key=lambda row: _stress_numeric(row.get("repair_gain_vs_none")) or -1e18,
            reverse=True,
        )
        for row in strongest[: max(catalog_top_overall, 1)]:
            _append_catalog_row(
                case_id=str(row["case_id"]),
                instance_label=str(row["case_id"]),
                seed=int(row["seed"]),
                profile_compared=f"{repair_variant}_vs_{baseline_variant}",
                regret_or_loss=_stress_numeric(row.get("repair_gain_vs_none")),
                why_selected_as_stress_case="repair_note_gain",
                case_group=str(row.get("case_group") or "overall"),
                source_metric="repair_gain_vs_none",
                extra={
                    "repair_gap_vs_greedy": row.get("repair_gap_vs_greedy"),
                    "repair_feasible_ratio": row.get("repair_feasible_ratio"),
                    "repair_generations_to_first_feasible": row.get(
                        "repair_generations_to_first_feasible"
                    ),
                },
            )
        for group in sorted(note_groups):
            group_rows = [
                row for row in strongest if str(row.get("case_group") or "overall") == group
            ]
            if not group_rows:
                continue
            for row in group_rows[:1]:
                _append_catalog_row(
                    case_id=str(row["case_id"]),
                    instance_label=str(row["case_id"]),
                    seed=int(row["seed"]),
                    profile_compared=f"{repair_variant}_vs_{baseline_variant}",
                    regret_or_loss=_stress_numeric(row.get("repair_gain_vs_none")),
                    why_selected_as_stress_case="repair_note_family",
                    case_group=group,
                    source_metric="repair_gain_vs_none",
                    extra={
                        "repair_gap_vs_greedy": row.get("repair_gap_vs_greedy"),
                        "repair_feasible_ratio": row.get("repair_feasible_ratio"),
                        "repair_generations_to_first_feasible": row.get(
                            "repair_generations_to_first_feasible"
                        ),
                    },
                )
        borderline_count = 0
        for row in sorted(
            candidate_rows,
            key=lambda row: abs(_stress_numeric(row.get("repair_gap_vs_greedy")) or 0.0),
        ):
            case_group = str(row.get("case_group") or "overall")
            if borderline_groups and case_group not in borderline_groups:
                continue
            _append_catalog_row(
                case_id=str(row["case_id"]),
                instance_label=str(row["case_id"]),
                seed=int(row["seed"]),
                profile_compared=f"{repair_variant}_vs_{default_variant}",
                regret_or_loss=_stress_numeric(row.get("repair_gap_vs_greedy")),
                why_selected_as_stress_case="borderline_family",
                case_group=case_group,
                source_metric="repair_gap_vs_greedy",
                extra={
                    "repair_gain_vs_none": row.get("repair_gain_vs_none"),
                    "repair_feasible_ratio": row.get("repair_feasible_ratio"),
                },
            )
            borderline_count += 1
            if borderline_count >= max(catalog_top_group, 1):
                break
        return rows

    if study.problem == "onemax" and raw_rows:
        control_variant = str(config.get("control_variant") or "none")
        reference_variant = str(config.get("reference_variant") or "early_stop_reference")
        grouped: dict[int, dict[str, dict[str, Any]]] = {}
        for row in raw_rows:
            variant = _study_variant_key(row)
            seed = row.get("seed")
            if variant not in {control_variant, reference_variant} or not isinstance(seed, int):
                continue
            grouped.setdefault(seed, {})[variant] = row
        for seed, variants in sorted(grouped.items()):
            control_row = variants.get(control_variant)
            reference_row = variants.get(reference_variant)
            if control_row is None or reference_row is None:
                continue
            control_eval = _stress_numeric(control_row.get("evaluations_to_target"))
            reference_eval = _stress_numeric(reference_row.get("evaluations_to_target"))
            delta = (
                control_eval - reference_eval
                if control_eval is not None and reference_eval is not None
                else None
            )
            _append_catalog_row(
                case_id=f"onemax_seed_{seed}",
                instance_label=f"onemax_seed_{seed}",
                seed=seed,
                profile_compared=f"{control_variant}_vs_{reference_variant}",
                regret_or_loss=delta,
                why_selected_as_stress_case="control_reference_check",
                case_group="control",
                source_metric="evaluations_to_target_delta",
                extra={
                    "control_evaluations_to_target": control_eval,
                    "reference_evaluations_to_target": reference_eval,
                },
            )
            break
        return rows

    return rows


def _stress_tail_summary_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
    qf_pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _stress_suite_config(study)
    if config is None:
        return []

    budget_band = str(config.get("budget_band_label") or "current_local_default")

    def _common_tail_fields(
        members: list[dict[str, Any]],
        values: list[float],
    ) -> dict[str, Any]:
        case_count = _stress_case_count(members)
        if case_count == 0:
            case_count = len(values)
        return {
            "pair_count": len(values),
            "case_count": case_count,
            "mean_loss_pct": mean(values),
            "median_loss_pct": median(values),
            "p75_loss_pct": _quantile(values, 0.75),
            "p90_loss_pct": _quantile(values, 0.9),
            "p95_loss_pct": _quantile(values, 0.95),
            "max_loss_pct": max(values),
            "actual_eval_savings_pct_mean": _mean_or_none(
                [
                    float(value)
                    for row in members
                    for value in [row.get("actual_eval_savings_pct")]
                    if isinstance(value, int | float)
                ]
            ),
            "runtime_savings_pct_mean": _mean_or_none(
                [
                    float(value)
                    for row in members
                    for value in [row.get("runtime_savings_pct")]
                    if isinstance(value, int | float)
                ]
            ),
        }

    rows: list[dict[str, Any]] = []
    if study.problem == "tsp" and qf_pair_rows:
        qf_config = _qf_tolerance_config(study) or {}
        quality_variant = str(qf_config.get("quality_variant") or "quality_first")
        fast_variant = str(qf_config.get("fast_variant") or "budget_first")
        comparison_label = f"{fast_variant}_vs_{quality_variant}"
        grouped: dict[str, list[dict[str, Any]]] = {"overall": qf_pair_rows}
        for case_group in sorted(
            {str(row.get("case_group") or "overall") for row in qf_pair_rows}
        ):
            grouped[case_group] = [
                row
                for row in qf_pair_rows
                if str(row.get("case_group") or "overall") == case_group
            ]
        for case_group, members in grouped.items():
            values = [
                float(value)
                for row in members
                for value in [_stress_numeric(row.get("route_distance_loss_pct"))]
                if value is not None
            ]
            if not values:
                continue
            case_signs: dict[str, set[str]] = {}
            for row in members:
                case_id = str(row.get("case_id") or "")
                loss = _stress_numeric(row.get("route_distance_loss_pct"))
                if not case_id or loss is None:
                    continue
                sign = "positive" if loss > 0.0 else "negative" if loss < 0.0 else "zero"
                case_signs.setdefault(case_id, set()).add(sign)
            rows.append(
                {
                    "problem": study.problem,
                    "scope": "overall" if case_group == "overall" else "case_group",
                    "case_group": case_group,
                    "budget_band": budget_band,
                    "profile_compared": comparison_label,
                    **_common_tail_fields(members, values),
                    "decision_flip_rate": _mean_or_none(
                        [1.0 if len(signs) > 1 else 0.0 for signs in case_signs.values()]
                    ),
                    "cost_ratio_fast_to_quality_mean": _mean_or_none(
                        [
                            float(row["fast_actual_evaluations_used"])
                            / float(row["quality_actual_evaluations_used"])
                            for row in members
                            if isinstance(row.get("fast_actual_evaluations_used"), int | float)
                            and isinstance(row.get("quality_actual_evaluations_used"), int | float)
                            and float(row["quality_actual_evaluations_used"]) > 0.0
                        ]
                    ),
                }
            )
        return rows

    if study.problem == "zdt1" and qf_pair_rows:
        qf_config = _qf_tolerance_config(study) or {}
        quality_variant = str(qf_config.get("quality_variant") or "quality_first")
        fast_variant = str(qf_config.get("fast_variant") or "budget_first")
        comparison_label = f"{fast_variant}_vs_{quality_variant}"

        def _zdt1_group(row: dict[str, Any]) -> str:
            pareto_fail = _stress_numeric(row.get("pareto_ratio_safety_fail")) == 1.0
            spread_fail = _stress_numeric(row.get("spread_safety_fail")) == 1.0
            if pareto_fail and spread_fail:
                return "joint_safety_fail"
            if spread_fail:
                return "spread_safety_fail"
            if pareto_fail:
                return "pareto_safety_fail"
            if (_stress_numeric(row.get("hv_loss_pct")) or 0.0) > 0.5:
                return "hv_tail"
            return "clean"

        grouped: dict[str, list[dict[str, Any]]] = {"overall": qf_pair_rows}
        for row in qf_pair_rows:
            grouped.setdefault(_zdt1_group(row), []).append(row)
        for case_group, members in grouped.items():
            values = [
                float(value)
                for row in members
                for value in [_stress_numeric(row.get("hv_loss_pct"))]
                if value is not None
            ]
            if not values:
                continue
            rows.append(
                {
                    "problem": study.problem,
                    "scope": "overall" if case_group == "overall" else "stress_group",
                    "case_group": case_group,
                    "budget_band": budget_band,
                    "profile_compared": comparison_label,
                    **_common_tail_fields(members, values),
                    "pareto_ratio_delta_mean": _mean_or_none(
                        [
                            float(value)
                            for row in members
                            for value in [row.get("pareto_ratio_delta")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "spread_delta_mean": _mean_or_none(
                        [
                            float(value)
                            for row in members
                            for value in [row.get("spread_delta")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "pareto_ratio_fail_rate": _mean_or_none(
                        [float(row.get("pareto_ratio_safety_fail", 0.0)) for row in members]
                    ),
                    "spread_fail_rate": _mean_or_none(
                        [float(row.get("spread_safety_fail", 0.0)) for row in members]
                    ),
                    "joint_safety_fail_rate": _mean_or_none(
                        [float(row.get("joint_safety_fail", 0.0)) for row in members]
                    ),
                }
            )
        return rows

    if study.problem == "knapsack" and raw_rows:
        default_variant = str(config.get("default_variant") or "greedy_local_search")
        baseline_variant = str(config.get("baseline_variant") or "none")
        repair_variant = str(config.get("repair_variant") or "repair_only")
        grouped_runs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        for row in raw_rows:
            variant = _study_variant_key(row)
            case_id = str(row.get("case_id") or "")
            seed = row.get("seed")
            if variant not in {default_variant, baseline_variant, repair_variant}:
                continue
            if not case_id or not isinstance(seed, int):
                continue
            grouped_runs.setdefault((case_id, seed), {})[variant] = row
        derived_rows: list[dict[str, Any]] = []
        for (case_id, seed), variants in grouped_runs.items():
            default_row = variants.get(default_variant)
            baseline_row = variants.get(baseline_variant)
            repair_row = variants.get(repair_variant)
            if default_row is None or baseline_row is None or repair_row is None:
                continue
            default_metric = _stress_numeric(default_row.get("best_feasible_fitness"))
            baseline_metric = _stress_numeric(baseline_row.get("best_feasible_fitness"))
            repair_metric = _stress_numeric(repair_row.get("best_feasible_fitness"))
            if default_metric is None or baseline_metric is None or repair_metric is None:
                continue
            derived_rows.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "case_group": str(repair_row.get("case_group") or "overall"),
                    "repair_gain_vs_none": repair_metric - baseline_metric,
                    "repair_gap_vs_greedy": repair_metric - default_metric,
                    "repair_feasible_ratio": _stress_numeric(repair_row.get("feasible_ratio")),
                    "repair_generations_to_first_feasible": _stress_numeric(
                        repair_row.get("generations_to_first_feasible")
                    ),
                }
            )
        grouped: dict[str, list[dict[str, Any]]] = {"overall": derived_rows}
        for case_group in sorted({str(row.get("case_group") or "overall") for row in derived_rows}):
            grouped[case_group] = [
                row
                for row in derived_rows
                if str(row.get("case_group") or "overall") == case_group
            ]
        for case_group, members in grouped.items():
            gains = [
                float(value)
                for row in members
                for value in [_stress_numeric(row.get("repair_gain_vs_none"))]
                if value is not None
            ]
            if not gains:
                continue
            rows.append(
                {
                    "problem": study.problem,
                    "scope": "overall" if case_group == "overall" else "case_group",
                    "case_group": case_group,
                    "budget_band": budget_band,
                    "profile_compared": f"{repair_variant}_vs_{baseline_variant}_vs_{default_variant}",
                    "pair_count": len(gains),
                    "case_count": len({str(row['case_id']) for row in members}),
                    "repair_gain_vs_none_mean": mean(gains),
                    "repair_gain_vs_none_median": median(gains),
                    "repair_gain_vs_none_p75": _quantile(gains, 0.75),
                    "repair_gain_vs_none_p90": _quantile(gains, 0.9),
                    "repair_gain_vs_none_p95": _quantile(gains, 0.95),
                    "repair_gain_vs_none_max": max(gains),
                    "repair_gap_vs_greedy_mean": _mean_or_none(
                        [
                            float(value)
                            for row in members
                            for value in [row.get("repair_gap_vs_greedy")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "repair_gap_vs_greedy_max_abs": (
                        max(
                            abs(float(value))
                            for row in members
                            for value in [row.get("repair_gap_vs_greedy")]
                            if isinstance(value, int | float)
                        )
                        if any(
                            isinstance(row.get("repair_gap_vs_greedy"), int | float)
                            for row in members
                        )
                        else None
                    ),
                    "repair_feasible_ratio_mean": _mean_or_none(
                        [
                            float(value)
                            for row in members
                            for value in [row.get("repair_feasible_ratio")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "repair_generations_to_first_feasible_mean": _mean_or_none(
                        [
                            float(value)
                            for row in members
                            for value in [row.get("repair_generations_to_first_feasible")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "repair_beats_none_rate": _mean_or_none(
                        [1.0 if float(row["repair_gain_vs_none"]) > 0.0 else 0.0 for row in members]
                    ),
                    "repair_beats_greedy_rate": _mean_or_none(
                        [
                            1.0
                            if (_stress_numeric(row.get("repair_gap_vs_greedy")) or 0.0) > 0.0
                            else 0.0
                            for row in members
                        ]
                    ),
                }
            )
        return rows

    if study.problem == "onemax" and raw_rows:
        control_variant = str(config.get("control_variant") or "none")
        reference_variant = str(config.get("reference_variant") or "early_stop_reference")
        grouped: dict[int, dict[str, dict[str, Any]]] = {}
        for row in raw_rows:
            variant = _study_variant_key(row)
            seed = row.get("seed")
            if variant not in {control_variant, reference_variant} or not isinstance(seed, int):
                continue
            grouped.setdefault(seed, {})[variant] = row
        deltas: list[float] = []
        members: list[dict[str, Any]] = []
        for seed, variants in sorted(grouped.items()):
            control_row = variants.get(control_variant)
            reference_row = variants.get(reference_variant)
            if control_row is None or reference_row is None:
                continue
            control_eval = _stress_numeric(control_row.get("evaluations_to_target"))
            reference_eval = _stress_numeric(reference_row.get("evaluations_to_target"))
            if control_eval is None or reference_eval is None:
                continue
            deltas.append(control_eval - reference_eval)
            members.append({"case_id": f"onemax_seed_{seed}"})
        if deltas:
            rows.append(
                {
                    "problem": study.problem,
                    "scope": "overall",
                    "case_group": "control",
                    "budget_band": budget_band,
                    "profile_compared": f"{control_variant}_vs_{reference_variant}",
                    "pair_count": len(deltas),
                    "case_count": len(members),
                    "control_delta_vs_reference_mean": mean(deltas),
                    "control_delta_vs_reference_median": median(deltas),
                    "control_delta_vs_reference_max_abs": max(abs(value) for value in deltas),
                    "control_stable_rate": _mean_or_none(
                        [1.0 if value <= 0.0 else 0.0 for value in deltas]
                    ),
                }
            )
        return rows

    return rows


def _seed_budget_counts(config: dict[str, Any]) -> list[int]:
    values = config.get("seed_counts")
    if not isinstance(values, list):
        return []
    counts = sorted(
        {
            int(value)
            for value in values
            if isinstance(value, int | float)
            and not isinstance(value, bool)
            and int(value) > 0
        }
    )
    return counts


def _seed_budget_accept_columns(tolerance_values: list[float]) -> list[tuple[float, str]]:
    columns: list[tuple[float, str]] = []
    for tolerance in tolerance_values:
        label = str(tolerance).replace(".", "_")
        columns.append((tolerance, f"accept_rate_at_{label}_pct"))
    return columns


def _prefix_seed_rows(
    rows: list[dict[str, Any]],
    seed_count: int,
) -> tuple[list[dict[str, Any]], int, int]:
    available_seeds = sorted(
        {
            int(seed)
            for row in rows
            for seed in [row.get("seed")]
            if isinstance(seed, int | float) and not isinstance(seed, bool)
        }
    )
    if not available_seeds:
        return [], 0, 0
    actual_seed_count = min(seed_count, len(available_seeds))
    selected = set(available_seeds[:actual_seed_count])
    prefix_rows = [
        row
        for row in rows
        if isinstance(row.get("seed"), int | float)
        and not isinstance(row.get("seed"), bool)
        and int(row["seed"]) in selected
    ]
    return prefix_rows, actual_seed_count, len(available_seeds)


def _ci_width(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    return 2.0 * 1.96 * stdev(values) / math.sqrt(len(values))


def _seed_budget_pair_maps(
    rows: list[dict[str, Any]],
    variant_keys: set[str],
) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        variant = _study_variant_key(row)
        if variant not in variant_keys:
            continue
        group_id = _pairwise_group_id(row)
        if group_id is None:
            continue
        grouped.setdefault(group_id, {})[variant] = row
    return grouped


def _seed_budget_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
    qf_pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _seed_budget_config(study)
    if config is None:
        return []
    seed_counts = _seed_budget_counts(config)
    if not seed_counts:
        return []
    tolerance_values = [
        float(value)
        for value in config.get("tolerance_bins_pct", [0.1, 0.25, 0.5, 1.0])
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    accept_columns = _seed_budget_accept_columns(tolerance_values)
    rows: list[dict[str, Any]] = []

    if study.problem in {"tsp", "zdt1"}:
        if not qf_pair_rows:
            return []
        grouped: dict[str, list[dict[str, Any]]] = {"overall": qf_pair_rows}
        if study.problem == "tsp":
            for case_group in sorted(
                {str(row.get("case_group") or "overall") for row in qf_pair_rows}
            ):
                grouped[case_group] = [
                    row for row in qf_pair_rows if str(row.get("case_group") or "overall") == case_group
                ]

        for scope_key, members in grouped.items():
            for seed_count in seed_counts:
                prefix_rows, actual_seed_count, total_seed_count = _prefix_seed_rows(
                    members,
                    seed_count,
                )
                if not prefix_rows:
                    continue
                scope = "overall" if scope_key == "overall" else "case_group"
                case_group = scope_key if scope == "case_group" else "overall"
                base_row: dict[str, Any] = {
                    "problem": study.problem,
                    "scope": scope,
                    "case_group": case_group,
                    "seed_count": actual_seed_count,
                    "requested_seed_count": seed_count,
                    "total_seed_count_available": total_seed_count,
                    "sample_count": len(prefix_rows),
                    "actual_eval_savings_pct_mean": _mean_or_none(
                        [
                            float(value)
                            for row in prefix_rows
                            for value in [row.get("actual_eval_savings_pct")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "runtime_savings_pct_mean": _mean_or_none(
                        [
                            float(value)
                            for row in prefix_rows
                            for value in [row.get("runtime_savings_pct")]
                            if isinstance(value, int | float)
                        ]
                    ),
                }
                if study.problem == "tsp":
                    losses = [float(row["route_distance_loss_pct"]) for row in prefix_rows]
                    mean_loss = mean(losses)
                    p90_loss = _quantile(losses, 0.9)
                    decision_tolerance = float(config.get("decision_tolerance_pct", 0.5))
                    quality_better_rate = _mean_or_none(
                        [
                            1.0 if row.get("comparison_outcome") == "quality_better" else 0.0
                            for row in prefix_rows
                        ]
                    )
                    fast_better_rate = _mean_or_none(
                        [
                            1.0 if row.get("comparison_outcome") == "fast_better" else 0.0
                            for row in prefix_rows
                        ]
                    )
                    tie_rate = _mean_or_none(
                        [1.0 if row.get("comparison_outcome") == "tie" else 0.0 for row in prefix_rows]
                    )
                    decision = "ambiguous"
                    if mean_loss <= decision_tolerance and (p90_loss or 0.0) <= max(
                        1.0,
                        decision_tolerance * 2.0,
                    ):
                        decision = "f_acceptable"
                    elif (quality_better_rate or 0.0) >= 0.5 or mean_loss > decision_tolerance:
                        decision = "q_preferred"
                    row = {
                        **base_row,
                        "decision": decision,
                        "mean_loss_pct": mean_loss,
                        "median_loss_pct": median(losses),
                        "p75_loss_pct": _quantile(losses, 0.75),
                        "p90_loss_pct": p90_loss,
                        "max_loss_pct": max(losses),
                        "ci_width_pct": _ci_width(losses),
                        "fast_better_rate": fast_better_rate,
                        "tie_rate": tie_rate,
                        "quality_better_rate": quality_better_rate,
                    }
                    for tolerance, column in accept_columns:
                        row[column] = _mean_or_none(
                            [
                                1.0 if float(member["route_distance_loss_pct"]) <= tolerance else 0.0
                                for member in prefix_rows
                            ]
                        )
                    rows.append(row)
                    continue

                hv_losses = [float(row["hv_loss_pct"]) for row in prefix_rows]
                mean_loss = mean(hv_losses)
                joint_fail_rate = _mean_or_none(
                    [float(row.get("joint_safety_fail", 0.0)) for row in prefix_rows]
                )
                decision_tolerance = float(config.get("decision_tolerance_pct", 0.25))
                decision = "ambiguous"
                if mean_loss <= decision_tolerance and (joint_fail_rate or 0.0) <= 0.1:
                    decision = "f_acceptable"
                elif mean_loss > decision_tolerance or (joint_fail_rate or 0.0) > 0.25:
                    decision = "q_preferred"
                row = {
                    **base_row,
                    "decision": decision,
                    "mean_loss_pct": mean_loss,
                    "median_loss_pct": median(hv_losses),
                    "p75_loss_pct": _quantile(hv_losses, 0.75),
                    "p90_loss_pct": _quantile(hv_losses, 0.9),
                    "max_loss_pct": max(hv_losses),
                    "ci_width_pct": _ci_width(hv_losses),
                    "pareto_ratio_delta_mean": _mean_or_none(
                        [
                            float(value)
                            for row in prefix_rows
                            for value in [row.get("pareto_ratio_delta")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "spread_delta_mean": _mean_or_none(
                        [
                            float(value)
                            for row in prefix_rows
                            for value in [row.get("spread_delta")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "pareto_ratio_fail_rate": _mean_or_none(
                        [float(row.get("pareto_ratio_safety_fail", 0.0)) for row in prefix_rows]
                    ),
                    "spread_fail_rate": _mean_or_none(
                        [float(row.get("spread_safety_fail", 0.0)) for row in prefix_rows]
                    ),
                    "joint_safety_fail_rate": joint_fail_rate,
                }
                for tolerance, column in accept_columns:
                    row[column] = _mean_or_none(
                        [
                            1.0
                            if float(member["hv_loss_pct"]) <= tolerance
                            and float(member.get("joint_safety_fail", 0.0)) < 0.5
                            else 0.0
                            for member in prefix_rows
                        ]
                    )
                rows.append(row)

    elif study.problem == "knapsack":
        none_variant = str(config.get("none_variant") or "none")
        repair_variant = str(config.get("repair_variant") or "repair_only")
        greedy_variant = str(config.get("greedy_variant") or "greedy_local_search")
        pair_maps = _seed_budget_pair_maps(
            raw_rows,
            {none_variant, repair_variant, greedy_variant},
        )
        grouped_members: dict[str, list[dict[str, Any]]] = {"overall": raw_rows}
        for case_group in sorted(
            {str(row.get("case_group") or row.get("family_label") or "overall") for row in raw_rows}
        ):
            grouped_members[case_group] = [
                row
                for row in raw_rows
                if str(row.get("case_group") or row.get("family_label") or "overall") == case_group
            ]
        for scope_key, members in grouped_members.items():
            for seed_count in seed_counts:
                prefix_rows, actual_seed_count, total_seed_count = _prefix_seed_rows(
                    members,
                    seed_count,
                )
                if not prefix_rows:
                    continue
                selected_pairs = {
                    group_id: pair_maps[group_id]
                    for group_id in pair_maps
                    if isinstance(group_id[1], int) and group_id[1] in {
                        int(row["seed"])
                        for row in prefix_rows
                        if isinstance(row.get("seed"), int | float) and not isinstance(row.get("seed"), bool)
                    }
                }
                repair_gain_vs_none: list[float] = []
                greedy_minus_repair: list[float] = []
                feasible_ratios: list[float] = []
                first_feasible: list[float] = []
                for variants in selected_pairs.values():
                    none_row = variants.get(none_variant)
                    repair_row = variants.get(repair_variant)
                    greedy_row = variants.get(greedy_variant)
                    if repair_row is None or none_row is None or greedy_row is None:
                        continue
                    if not isinstance(repair_row.get("best_feasible_fitness"), int | float):
                        continue
                    if not isinstance(none_row.get("best_feasible_fitness"), int | float):
                        continue
                    if not isinstance(greedy_row.get("best_feasible_fitness"), int | float):
                        continue
                    repair_gain_vs_none.append(
                        float(repair_row["best_feasible_fitness"]) - float(none_row["best_feasible_fitness"])
                    )
                    greedy_minus_repair.append(
                        float(greedy_row["best_feasible_fitness"]) - float(repair_row["best_feasible_fitness"])
                    )
                    if isinstance(repair_row.get("feasible_ratio"), int | float):
                        feasible_ratios.append(float(repair_row["feasible_ratio"]))
                    if isinstance(repair_row.get("generations_to_first_feasible"), int | float):
                        first_feasible.append(float(repair_row["generations_to_first_feasible"]))
                if not repair_gain_vs_none:
                    continue
                repair_gain_mean = mean(repair_gain_vs_none)
                repair_gap_vs_greedy_mean = _mean_or_none(greedy_minus_repair)
                decision = "ambiguous"
                if repair_gain_mean > 0.0 and (repair_gap_vs_greedy_mean or 0.0) <= 1.0:
                    decision = "repair_note_stable"
                elif repair_gain_mean > 0.0:
                    decision = "repair_worth_trying"
                rows.append(
                    {
                        "problem": study.problem,
                        "scope": "overall" if scope_key == "overall" else "case_group",
                        "case_group": scope_key if scope_key != "overall" else "overall",
                        "seed_count": actual_seed_count,
                        "requested_seed_count": seed_count,
                        "total_seed_count_available": total_seed_count,
                        "sample_count": len(repair_gain_vs_none),
                        "decision": decision,
                        "repair_gain_vs_none_mean": repair_gain_mean,
                        "repair_gap_vs_greedy_mean": repair_gap_vs_greedy_mean,
                        "feasible_ratio_mean": _mean_or_none(feasible_ratios),
                        "generations_to_first_feasible_mean": _mean_or_none(first_feasible),
                        "ci_width_pct": _ci_width(repair_gain_vs_none),
                    }
                )

    elif study.problem == "onemax":
        control_variant = str(config.get("control_variant") or "none")
        reference_variant = str(config.get("reference_variant") or "early_stop_reference")
        pair_maps = _seed_budget_pair_maps(raw_rows, {control_variant, reference_variant})
        for seed_count in seed_counts:
            prefix_rows, actual_seed_count, total_seed_count = _prefix_seed_rows(raw_rows, seed_count)
            if not prefix_rows:
                continue
            selected_seed_set = {
                int(row["seed"])
                for row in prefix_rows
                if isinstance(row.get("seed"), int | float) and not isinstance(row.get("seed"), bool)
            }
            deltas: list[float] = []
            control_hits: list[float] = []
            for group_id, variants in pair_maps.items():
                if group_id[1] not in selected_seed_set:
                    continue
                control_row = variants.get(control_variant)
                reference_row = variants.get(reference_variant)
                if control_row is None or reference_row is None:
                    continue
                if not isinstance(control_row.get("evaluations_to_target"), int | float):
                    continue
                if not isinstance(reference_row.get("evaluations_to_target"), int | float):
                    continue
                deltas.append(
                    float(control_row["evaluations_to_target"])
                    - float(reference_row["evaluations_to_target"])
                )
                if isinstance(control_row.get("target_hit"), int | float):
                    control_hits.append(float(control_row["target_hit"]))
            if not deltas:
                continue
            decision = "control_stable" if abs(mean(deltas)) <= 1e-9 else "ambiguous"
            rows.append(
                {
                    "problem": study.problem,
                    "scope": "overall",
                    "case_group": "overall",
                    "seed_count": actual_seed_count,
                    "requested_seed_count": seed_count,
                    "total_seed_count_available": total_seed_count,
                    "sample_count": len(deltas),
                    "decision": decision,
                    "control_delta_vs_reference_mean": mean(deltas),
                    "target_hit_rate_mean": _mean_or_none(control_hits),
                    "ci_width_pct": _ci_width(deltas),
                }
            )

    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        group_key = (str(row.get("scope") or "overall"), str(row.get("case_group") or "overall"))
        grouped_rows.setdefault(group_key, []).append(row)
    for members in grouped_rows.values():
        full_row = max(members, key=lambda row: int(row.get("seed_count") or 0))
        full_decision = str(full_row.get("decision") or "ambiguous")
        for row in members:
            row["decision_flip_rate_vs_full"] = 0.0 if str(row.get("decision")) == full_decision else 1.0
            row["decision_stability_score"] = 1.0 - float(row["decision_flip_rate_vs_full"])
    return rows


def _sequential_compare_config(study: LocalStudy) -> dict[str, Any] | None:
    analysis = study.analysis
    if not isinstance(analysis, dict):
        return None
    config = analysis.get("sequential_compare")
    return config if isinstance(config, dict) else None


def _sequential_stage_counts(
    config: dict[str, Any],
    *,
    default_counts: list[int],
) -> list[int]:
    values = config.get("seed_stages", config.get("seed_counts", default_counts))
    if not isinstance(values, list):
        values = default_counts
    counts = sorted(
        {
            int(value)
            for value in values
            if isinstance(value, int | float)
            and not isinstance(value, bool)
            and int(value) > 0
        }
    )
    return counts or list(default_counts)


def _sum_or_none(values: list[float]) -> float | None:
    return sum(values) if values else None


def _paired_sign_counts(
    deltas: list[float],
    *,
    epsilon: float = 1e-9,
) -> tuple[int, int, int]:
    candidate_wins = sum(1 for value in deltas if value < -epsilon)
    baseline_wins = sum(1 for value in deltas if value > epsilon)
    ties = len(deltas) - candidate_wins - baseline_wins
    return candidate_wins, baseline_wins, ties


def _paired_seed_ids(rows: list[dict[str, Any]]) -> str:
    return ",".join(
        str(int(row["seed"]))
        for row in rows
        if isinstance(row.get("seed"), int | float) and not isinstance(row.get("seed"), bool)
    )


def _tsp_sequential_seed_rows(
    pair_rows: list[dict[str, Any]],
    *,
    case_group: str,
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in pair_rows:
        row_case_group = str(row.get("case_group") or "overall")
        if case_group != "overall" and row_case_group != case_group:
            continue
        seed = row.get("seed")
        loss_pct = row.get("route_distance_loss_pct")
        if not isinstance(seed, int | float) or isinstance(seed, bool):
            continue
        if not isinstance(loss_pct, int | float) or isinstance(loss_pct, bool):
            continue
        bucket = grouped.setdefault(
            int(seed),
            {
                "seed": int(seed),
                "paired_delta_values": [],
                "member_losses": [],
                "anti_case_losses": [],
                "rescue_losses": [],
                "actual_eval_values": [],
                "runtime_values": [],
            },
        )
        loss_value = float(loss_pct)
        bucket["paired_delta_values"].append(loss_value)
        bucket["member_losses"].append(loss_value)
        if row_case_group == "anti_case":
            bucket["anti_case_losses"].append(loss_value)
        if row_case_group == "rescue_target":
            bucket["rescue_losses"].append(loss_value)
        eval_values = [
            float(value)
            for value in (
                row.get("quality_actual_evaluations_used"),
                row.get("fast_actual_evaluations_used"),
            )
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        if eval_values:
            bucket["actual_eval_values"].append(sum(eval_values))
        runtime_values = [
            float(value)
            for value in (
                row.get("quality_runtime_seconds"),
                row.get("fast_runtime_seconds"),
            )
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        if runtime_values:
            bucket["runtime_values"].append(sum(runtime_values))

    seed_rows: list[dict[str, Any]] = []
    for seed, bucket in sorted(grouped.items()):
        delta_values = list(bucket["paired_delta_values"])
        seed_rows.append(
            {
                "seed": seed,
                "paired_delta": mean(delta_values),
                "member_losses": list(bucket["member_losses"]),
                "anti_case_losses": list(bucket["anti_case_losses"]),
                "rescue_losses": list(bucket["rescue_losses"]),
                "actual_evaluations_used": _sum_or_none(bucket["actual_eval_values"]),
                "runtime_seconds": _sum_or_none(bucket["runtime_values"]),
            }
        )
    return seed_rows


def _zdt1_sequential_seed_rows(
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seed_rows: list[dict[str, Any]] = []
    for row in sorted(pair_rows, key=lambda item: int(item.get("seed") or 0)):
        seed = row.get("seed")
        hv_loss = row.get("hv_loss_pct")
        if not isinstance(seed, int | float) or isinstance(seed, bool):
            continue
        if not isinstance(hv_loss, int | float) or isinstance(hv_loss, bool):
            continue
        eval_values = [
            float(value)
            for value in (
                row.get("quality_actual_evaluations_used"),
                row.get("fast_actual_evaluations_used"),
            )
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        runtime_values = [
            float(value)
            for value in (
                row.get("quality_runtime_seconds"),
                row.get("fast_runtime_seconds"),
            )
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        seed_rows.append(
            {
                "seed": int(seed),
                "paired_delta": float(hv_loss),
                "joint_safety_fail": float(row.get("joint_safety_fail", 0.0) or 0.0),
                "pareto_ratio_safety_fail": float(
                    row.get("pareto_ratio_safety_fail", 0.0) or 0.0
                ),
                "spread_safety_fail": float(row.get("spread_safety_fail", 0.0) or 0.0),
                "pareto_ratio_delta": float(row["pareto_ratio_delta"])
                if isinstance(row.get("pareto_ratio_delta"), int | float)
                and not isinstance(row.get("pareto_ratio_delta"), bool)
                else None,
                "spread_delta": float(row["spread_delta"])
                if isinstance(row.get("spread_delta"), int | float)
                and not isinstance(row.get("spread_delta"), bool)
                else None,
                "actual_evaluations_used": sum(eval_values) if eval_values else None,
                "runtime_seconds": sum(runtime_values) if runtime_values else None,
            }
        )
    return seed_rows


def _knapsack_sequential_pair_rows(
    raw_rows: list[dict[str, Any]],
    *,
    none_variant: str,
    repair_variant: str,
    greedy_variant: str,
) -> list[dict[str, Any]]:
    pair_maps = _seed_budget_pair_maps(raw_rows, {none_variant, repair_variant, greedy_variant})
    rows: list[dict[str, Any]] = []
    for (_case_id, _seed), variants in sorted(pair_maps.items(), key=lambda item: item[0]):
        none_row = variants.get(none_variant)
        repair_row = variants.get(repair_variant)
        greedy_row = variants.get(greedy_variant)
        if none_row is None or repair_row is None or greedy_row is None:
            continue
        none_best = none_row.get("best_feasible_fitness")
        repair_best = repair_row.get("best_feasible_fitness")
        greedy_best = greedy_row.get("best_feasible_fitness")
        if not isinstance(none_best, int | float) or isinstance(none_best, bool):
            continue
        if not isinstance(repair_best, int | float) or isinstance(repair_best, bool):
            continue
        if not isinstance(greedy_best, int | float) or isinstance(greedy_best, bool):
            continue
        eval_values = [
            float(value)
            for value in (
                none_row.get("total_actual_evaluations_used", none_row.get("actual_evaluations_used")),
                repair_row.get(
                    "total_actual_evaluations_used",
                    repair_row.get("actual_evaluations_used"),
                ),
            )
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        runtime_values = [
            float(value)
            for value in (none_row.get("runtime_seconds"), repair_row.get("runtime_seconds"))
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        rows.append(
            {
                "seed": int(repair_row.get("seed") or none_row.get("seed") or 0),
                "case_id": repair_row.get("case_id") or none_row.get("case_id") or "",
                "case_group": repair_row.get("case_group") or none_row.get("case_group") or "overall",
                "paired_delta": float(none_best) - float(repair_best),
                "greedy_gap_vs_repair": float(greedy_best) - float(repair_best),
                "initial_feasible_fraction": repair_row.get("initial_feasible_fraction"),
                "generations_to_first_feasible": repair_row.get("generations_to_first_feasible"),
                "actual_evaluations_used": sum(eval_values) if eval_values else None,
                "runtime_seconds": sum(runtime_values) if runtime_values else None,
            }
        )
    return rows


def _knapsack_sequential_seed_rows(
    pair_rows: list[dict[str, Any]],
    *,
    case_group: str,
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in pair_rows:
        row_case_group = str(row.get("case_group") or "overall")
        if case_group != "overall" and row_case_group != case_group:
            continue
        seed = row.get("seed")
        delta = row.get("paired_delta")
        if not isinstance(seed, int | float) or isinstance(seed, bool):
            continue
        if not isinstance(delta, int | float) or isinstance(delta, bool):
            continue
        bucket = grouped.setdefault(
            int(seed),
            {
                "seed": int(seed),
                "paired_delta_values": [],
                "greedy_gap_values": [],
                "initial_feasible_fraction_values": [],
                "first_feasible_values": [],
                "actual_eval_values": [],
                "runtime_values": [],
            },
        )
        bucket["paired_delta_values"].append(float(delta))
        if isinstance(row.get("greedy_gap_vs_repair"), int | float):
            bucket["greedy_gap_values"].append(float(row["greedy_gap_vs_repair"]))
        if isinstance(row.get("initial_feasible_fraction"), int | float):
            bucket["initial_feasible_fraction_values"].append(float(row["initial_feasible_fraction"]))
        if isinstance(row.get("generations_to_first_feasible"), int | float):
            bucket["first_feasible_values"].append(float(row["generations_to_first_feasible"]))
        if isinstance(row.get("actual_evaluations_used"), int | float):
            bucket["actual_eval_values"].append(float(row["actual_evaluations_used"]))
        if isinstance(row.get("runtime_seconds"), int | float):
            bucket["runtime_values"].append(float(row["runtime_seconds"]))
    seed_rows: list[dict[str, Any]] = []
    for seed, bucket in sorted(grouped.items()):
        seed_rows.append(
            {
                "seed": seed,
                "paired_delta": mean(bucket["paired_delta_values"]),
                "greedy_gap_vs_repair": _mean_or_none(bucket["greedy_gap_values"]),
                "initial_feasible_fraction": _mean_or_none(
                    bucket["initial_feasible_fraction_values"]
                ),
                "generations_to_first_feasible": _mean_or_none(bucket["first_feasible_values"]),
                "actual_evaluations_used": _sum_or_none(bucket["actual_eval_values"]),
                "runtime_seconds": _sum_or_none(bucket["runtime_values"]),
            }
        )
    return seed_rows


def _onemax_sequential_seed_rows(
    raw_rows: list[dict[str, Any]],
    *,
    control_variant: str,
    reference_variant: str,
) -> list[dict[str, Any]]:
    pair_maps = _seed_budget_pair_maps(raw_rows, {control_variant, reference_variant})
    rows: list[dict[str, Any]] = []
    for (_case_id, _seed), variants in sorted(pair_maps.items(), key=lambda item: item[0]):
        control_row = variants.get(control_variant)
        reference_row = variants.get(reference_variant)
        if control_row is None or reference_row is None:
            continue
        control_eval = control_row.get("evaluations_to_target")
        reference_eval = reference_row.get("evaluations_to_target")
        if not isinstance(control_eval, int | float) or isinstance(control_eval, bool):
            continue
        if not isinstance(reference_eval, int | float) or isinstance(reference_eval, bool):
            continue
        eval_values = [
            float(value)
            for value in (
                control_row.get("total_actual_evaluations_used", control_row.get("actual_evaluations_used")),
                reference_row.get(
                    "total_actual_evaluations_used",
                    reference_row.get("actual_evaluations_used"),
                ),
            )
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        runtime_values = [
            float(value)
            for value in (control_row.get("runtime_seconds"), reference_row.get("runtime_seconds"))
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        rows.append(
            {
                "seed": int(control_row.get("seed") or reference_row.get("seed") or 0),
                "paired_delta": float(reference_eval) - float(control_eval),
                "control_target_hit": float(control_row.get("target_hit", 0.0) or 0.0),
                "actual_evaluations_used": sum(eval_values) if eval_values else None,
                "runtime_seconds": sum(runtime_values) if runtime_values else None,
            }
        )
    return rows


def _tsp_sequential_decision(
    *,
    mode: str,
    stage_count: int,
    max_stage: int,
    mean_loss_pct: float,
    ci_low: float | None,
    ci_high: float | None,
    p90_loss_pct: float | None,
    anti_case_p90_loss_pct: float | None,
    config: dict[str, Any],
) -> tuple[str, str]:
    exploratory_tolerance = float(config.get("exploratory_accept_tolerance_pct", 1.5))
    exploratory_reject = float(config.get("exploratory_reject_tolerance_pct", 2.5))
    exploratory_tail = float(config.get("exploratory_tail_tolerance_pct", 4.5))
    final_tolerance = float(config.get("quality_final_tolerance_pct", 0.5))
    final_reject = float(config.get("quality_reject_tolerance_pct", 1.0))
    final_accept_tail = float(config.get("quality_tail_accept_pct", 2.5))
    final_caution_tail = float(config.get("quality_tail_caution_pct", 3.5))
    active_tail = anti_case_p90_loss_pct if anti_case_p90_loss_pct is not None else p90_loss_pct

    if mode == "exploratory":
        clear_accept = (
            mean_loss_pct <= exploratory_tolerance
            and (active_tail is None or active_tail <= exploratory_tail)
        )
        clear_reject = (
            mean_loss_pct > exploratory_reject
            or (ci_low is not None and ci_low > exploratory_tolerance)
        )
        if clear_reject:
            return "reject_early", "stop"
        if clear_accept and (
            stage_count >= 3
            or ci_high is None
            or ci_high <= exploratory_reject
            or stage_count >= max_stage
        ):
            return "accept_fast_exploratory", "stop"
        if stage_count >= max_stage:
            return (
                "accept_fast_exploratory" if clear_accept else "require_q_confirm",
                "stop",
            )
        return (
            "accept_fast_exploratory" if mean_loss_pct <= exploratory_tolerance else "require_q_confirm",
            "escalate",
        )

    clear_accept = (
        (ci_high is not None and ci_high <= final_tolerance)
        and mean_loss_pct <= final_tolerance
        and (active_tail is None or active_tail <= final_accept_tail)
    )
    clear_reject = (
        mean_loss_pct > final_reject
        or (ci_low is not None and ci_low > final_tolerance)
        or (active_tail is not None and active_tail > final_caution_tail)
    )
    if clear_accept:
        return "accept_fast_budget_final", "stop"
    if clear_reject:
        return "reject_early", "stop"
    if stage_count >= max_stage:
        return "require_q_confirm", "stop"
    if mean_loss_pct <= final_reject and (active_tail is None or active_tail <= final_caution_tail):
        return "accept_fast_budget_final", "escalate"
    return "require_q_confirm", "stop"


def _zdt1_sequential_decision(
    *,
    mode: str,
    stage_count: int,
    max_stage: int,
    mean_hv_loss_pct: float,
    ci_low: float | None,
    ci_high: float | None,
    joint_safety_fail_rate: float,
    config: dict[str, Any],
) -> tuple[str, str]:
    exploratory_tolerance = float(config.get("exploratory_hv_tolerance_pct", 0.5))
    exploratory_reject = float(config.get("exploratory_reject_hv_pct", 1.0))
    exploratory_fail = float(config.get("exploratory_joint_safety_fail_rate", 0.34))
    final_tolerance = float(config.get("final_hv_tolerance_pct", 0.25))
    final_reject = float(config.get("final_reject_hv_pct", 0.5))
    final_fail = float(config.get("final_joint_safety_fail_rate", 0.1))
    final_reject_fail = float(config.get("final_reject_joint_safety_fail_rate", 0.25))

    if mode == "exploratory":
        clear_accept = (
            mean_hv_loss_pct <= exploratory_tolerance
            and joint_safety_fail_rate <= exploratory_fail
        )
        clear_reject = (
            mean_hv_loss_pct > exploratory_reject
            or joint_safety_fail_rate > 0.5
            or (ci_low is not None and ci_low > exploratory_tolerance)
        )
        if clear_reject:
            return "reject_early", "stop"
        if clear_accept and (
            stage_count >= 3
            or ci_high is None
            or ci_high <= exploratory_reject
            or stage_count >= max_stage
        ):
            return "accept_fast_exploratory", "stop"
        if stage_count >= max_stage:
            return (
                "accept_fast_exploratory" if clear_accept else "require_q_confirm",
                "stop",
            )
        return (
            "accept_fast_exploratory" if mean_hv_loss_pct <= exploratory_tolerance else "require_q_confirm",
            "escalate",
        )

    clear_accept = (
        mean_hv_loss_pct <= final_tolerance
        and (ci_high is not None and ci_high <= final_tolerance)
        and joint_safety_fail_rate <= final_fail
    )
    clear_reject = (
        mean_hv_loss_pct > final_reject
        or joint_safety_fail_rate > final_reject_fail
        or (ci_low is not None and ci_low > final_tolerance)
    )
    if clear_accept:
        return "accept_fast_budget_final", "stop"
    if clear_reject:
        return "require_q_confirm", "stop"
    if stage_count >= max_stage:
        return "require_q_confirm", "stop"
    return "accept_fast_budget_final", "escalate"


def _knapsack_sequential_decision(
    *,
    stage_count: int,
    max_stage: int,
    mean_delta: float,
    ci_low: float | None,
    ci_high: float | None,
) -> tuple[str, str]:
    if ci_high is not None and ci_high < 0.0:
        return "accept_fast_exploratory", "stop"
    if ci_low is not None and ci_low > 0.0:
        return "reject_early", "stop"
    if stage_count >= max_stage:
        return (
            "accept_fast_exploratory" if mean_delta < 0.0 else "require_q_confirm",
            "stop",
        )
    return (
        "accept_fast_exploratory" if mean_delta < 0.0 else "require_q_confirm",
        "escalate",
    )


def _onemax_sequential_decision(
    *,
    mean_delta: float,
    ci_low: float | None,
    ci_high: float | None,
    config: dict[str, Any],
) -> tuple[str, str]:
    equivalence_tolerance = float(config.get("equivalence_eval_tolerance", 25.0))
    if abs(mean_delta) <= equivalence_tolerance:
        return "reject_early", "stop"
    if ci_high is not None and ci_high < 0.0:
        return "accept_fast_exploratory", "stop"
    if ci_low is not None and ci_low > 0.0:
        return "reject_early", "stop"
    return "reject_early", "stop"


def _sequential_decision_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
    qf_pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _sequential_compare_config(study)
    if config is None:
        return []

    seed_stages = _sequential_stage_counts(
        config,
        default_counts=[3, 5, 8, 10] if study.problem != "onemax" else [1, 3, 5],
    )
    if not seed_stages:
        return []
    max_stage = max(seed_stages)
    bootstrap_seed = int(config.get("bootstrap_seed", 7319))
    bootstrap_replicates = int(config.get("bootstrap_replicates", 400))

    rows: list[dict[str, Any]] = []

    if study.problem == "tsp":
        if not qf_pair_rows:
            return []
        modes = [
            str(value)
            for value in config.get("modes", ["exploratory", "quality_sensitive"])
            if isinstance(value, str) and value
        ]
        scope_seed_rows = {
            "overall": _tsp_sequential_seed_rows(qf_pair_rows, case_group="overall"),
            "rescue_target": _tsp_sequential_seed_rows(qf_pair_rows, case_group="rescue_target"),
            "anti_case": _tsp_sequential_seed_rows(qf_pair_rows, case_group="anti_case"),
        }
        for mode in modes:
            for case_group, seed_rows in scope_seed_rows.items():
                if not seed_rows:
                    continue
                full_rows, _, _ = _prefix_seed_rows(seed_rows, max_stage)
                full_eval = _sum_or_none(
                    [
                        float(row["actual_evaluations_used"])
                        for row in full_rows
                        if isinstance(row.get("actual_evaluations_used"), int | float)
                    ]
                )
                full_runtime = _sum_or_none(
                    [
                        float(row["runtime_seconds"])
                        for row in full_rows
                        if isinstance(row.get("runtime_seconds"), int | float)
                    ]
                )
                for seed_count in seed_stages:
                    prefix_rows, actual_seed_count, total_seed_count = _prefix_seed_rows(
                        seed_rows,
                        seed_count,
                    )
                    if not prefix_rows:
                        continue
                    deltas = [float(row["paired_delta"]) for row in prefix_rows]
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        resamples=bootstrap_replicates,
                        seed=bootstrap_seed + seed_count + len(mode) + len(case_group),
                    )
                    member_losses = [
                        float(value)
                        for row in prefix_rows
                        for value in row.get("member_losses", [])
                    ]
                    anti_case_losses = [
                        float(value)
                        for row in prefix_rows
                        for value in row.get("anti_case_losses", [])
                    ]
                    rescue_losses = [
                        float(value)
                        for row in prefix_rows
                        for value in row.get("rescue_losses", [])
                    ]
                    candidate_wins, baseline_wins, ties = _paired_sign_counts(deltas)
                    actual_eval = _sum_or_none(
                        [
                            float(value)
                            for row in prefix_rows
                            for value in [row.get("actual_evaluations_used")]
                            if isinstance(value, int | float)
                        ]
                    )
                    runtime_seconds = _sum_or_none(
                        [
                            float(value)
                            for row in prefix_rows
                            for value in [row.get("runtime_seconds")]
                            if isinstance(value, int | float)
                        ]
                    )
                    mean_loss = mean(member_losses) if member_losses else mean(deltas)
                    p90_loss = _quantile(member_losses, 0.9) if member_losses else None
                    anti_case_p90 = _quantile(anti_case_losses, 0.9) if anti_case_losses else None
                    decision_label, stage_action = _tsp_sequential_decision(
                        mode=mode,
                        stage_count=actual_seed_count,
                        max_stage=max_stage,
                        mean_loss_pct=mean_loss,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        p90_loss_pct=p90_loss,
                        anti_case_p90_loss_pct=anti_case_p90,
                        config=config,
                    )
                    rows.append(
                        {
                            "problem": study.problem,
                            "mode": mode,
                            "scope": "overall" if case_group == "overall" else "case_group",
                            "case_group": case_group,
                            "seed_count": actual_seed_count,
                            "requested_seed_count": seed_count,
                            "total_seed_count_available": total_seed_count,
                            "sample_count": len(deltas),
                            "seed_ids": _paired_seed_ids(prefix_rows),
                            "paired_delta_mean": mean(deltas),
                            "paired_delta_median": median(deltas),
                            "bootstrap_ci_low": ci_low,
                            "bootstrap_ci_high": ci_high,
                            "bootstrap_ci_width": (
                                ci_high - ci_low
                                if ci_low is not None and ci_high is not None
                                else None
                            ),
                            "paired_win_count": candidate_wins,
                            "paired_loss_count": baseline_wins,
                            "paired_tie_count": ties,
                            "decision_label": decision_label,
                            "stage_action": stage_action,
                            "should_stop": 1.0 if stage_action == "stop" else 0.0,
                            "should_escalate": 1.0 if stage_action == "escalate" else 0.0,
                            "actual_evaluations_used": actual_eval,
                            "runtime_seconds": runtime_seconds,
                            "actual_eval_savings_vs_full_pct": (
                                (float(full_eval) - float(actual_eval)) / float(full_eval) * 100.0
                                if isinstance(full_eval, int | float)
                                and not isinstance(full_eval, bool)
                                and float(full_eval) != 0.0
                                and isinstance(actual_eval, int | float)
                                and not isinstance(actual_eval, bool)
                                else None
                            ),
                            "runtime_savings_vs_full_pct": (
                                (float(full_runtime) - float(runtime_seconds)) / float(full_runtime) * 100.0
                                if isinstance(full_runtime, int | float)
                                and not isinstance(full_runtime, bool)
                                and float(full_runtime) != 0.0
                                and isinstance(runtime_seconds, int | float)
                                and not isinstance(runtime_seconds, bool)
                                else None
                            ),
                            "mean_loss_pct": mean_loss,
                            "median_loss_pct": median(member_losses) if member_losses else median(deltas),
                            "p75_loss_pct": (
                                _quantile(member_losses, 0.75) if member_losses else _quantile(deltas, 0.75)
                            ),
                            "p90_loss_pct": p90_loss,
                            "max_loss_pct": max(member_losses) if member_losses else max(deltas),
                            "rescue_target_mean_loss_pct": mean(rescue_losses) if rescue_losses else None,
                            "anti_case_mean_loss_pct": mean(anti_case_losses) if anti_case_losses else None,
                            "anti_case_p90_loss_pct": anti_case_p90,
                            "anti_case_max_loss_pct": max(anti_case_losses) if anti_case_losses else None,
                        }
                    )

    elif study.problem == "zdt1":
        if not qf_pair_rows:
            return []
        modes = [
            str(value)
            for value in config.get("modes", ["exploratory", "final_safety"])
            if isinstance(value, str) and value
        ]
        seed_rows = _zdt1_sequential_seed_rows(qf_pair_rows)
        if not seed_rows:
            return []
        full_rows, _, _ = _prefix_seed_rows(seed_rows, max_stage)
        full_eval = _sum_or_none(
            [
                float(row["actual_evaluations_used"])
                for row in full_rows
                if isinstance(row.get("actual_evaluations_used"), int | float)
            ]
        )
        full_runtime = _sum_or_none(
            [
                float(row["runtime_seconds"])
                for row in full_rows
                if isinstance(row.get("runtime_seconds"), int | float)
            ]
        )
        for mode in modes:
            for seed_count in seed_stages:
                prefix_rows, actual_seed_count, total_seed_count = _prefix_seed_rows(
                    seed_rows,
                    seed_count,
                )
                if not prefix_rows:
                    continue
                deltas = [float(row["paired_delta"]) for row in prefix_rows]
                ci_low, ci_high = _bootstrap_ci(
                    deltas,
                    resamples=bootstrap_replicates,
                    seed=bootstrap_seed + seed_count + len(mode),
                )
                candidate_wins, baseline_wins, ties = _paired_sign_counts(deltas)
                joint_fail_rate = _mean_or_none(
                    [
                        float(row["joint_safety_fail"])
                        for row in prefix_rows
                        if isinstance(row.get("joint_safety_fail"), int | float)
                    ]
                ) or 0.0
                actual_eval = _sum_or_none(
                    [
                        float(value)
                        for row in prefix_rows
                        for value in [row.get("actual_evaluations_used")]
                        if isinstance(value, int | float)
                    ]
                )
                runtime_seconds = _sum_or_none(
                    [
                        float(value)
                        for row in prefix_rows
                        for value in [row.get("runtime_seconds")]
                        if isinstance(value, int | float)
                    ]
                )
                decision_label, stage_action = _zdt1_sequential_decision(
                    mode=mode,
                    stage_count=actual_seed_count,
                    max_stage=max_stage,
                    mean_hv_loss_pct=mean(deltas),
                    ci_low=ci_low,
                    ci_high=ci_high,
                    joint_safety_fail_rate=joint_fail_rate,
                    config=config,
                )
                rows.append(
                    {
                        "problem": study.problem,
                        "mode": mode,
                        "scope": "overall",
                        "case_group": "overall",
                        "seed_count": actual_seed_count,
                        "requested_seed_count": seed_count,
                        "total_seed_count_available": total_seed_count,
                        "sample_count": len(deltas),
                        "seed_ids": _paired_seed_ids(prefix_rows),
                        "paired_delta_mean": mean(deltas),
                        "paired_delta_median": median(deltas),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "bootstrap_ci_width": (
                            ci_high - ci_low
                            if ci_low is not None and ci_high is not None
                            else None
                        ),
                        "paired_win_count": candidate_wins,
                        "paired_loss_count": baseline_wins,
                        "paired_tie_count": ties,
                        "decision_label": decision_label,
                        "stage_action": stage_action,
                        "should_stop": 1.0 if stage_action == "stop" else 0.0,
                        "should_escalate": 1.0 if stage_action == "escalate" else 0.0,
                        "actual_evaluations_used": actual_eval,
                        "runtime_seconds": runtime_seconds,
                        "actual_eval_savings_vs_full_pct": (
                            (float(full_eval) - float(actual_eval)) / float(full_eval) * 100.0
                            if isinstance(full_eval, int | float)
                            and not isinstance(full_eval, bool)
                            and float(full_eval) != 0.0
                            and isinstance(actual_eval, int | float)
                            and not isinstance(actual_eval, bool)
                            else None
                        ),
                        "runtime_savings_vs_full_pct": (
                            (float(full_runtime) - float(runtime_seconds)) / float(full_runtime) * 100.0
                            if isinstance(full_runtime, int | float)
                            and not isinstance(full_runtime, bool)
                            and float(full_runtime) != 0.0
                            and isinstance(runtime_seconds, int | float)
                            and not isinstance(runtime_seconds, bool)
                            else None
                        ),
                        "mean_loss_pct": mean(deltas),
                        "median_loss_pct": median(deltas),
                        "p75_loss_pct": _quantile(deltas, 0.75),
                        "p90_loss_pct": _quantile(deltas, 0.9),
                        "max_loss_pct": max(deltas),
                        "pareto_ratio_delta_mean": _mean_or_none(
                            [
                                float(value)
                                for row in prefix_rows
                                for value in [row.get("pareto_ratio_delta")]
                                if isinstance(value, int | float)
                            ]
                        ),
                        "spread_delta_mean": _mean_or_none(
                            [
                                float(value)
                                for row in prefix_rows
                                for value in [row.get("spread_delta")]
                                if isinstance(value, int | float)
                            ]
                        ),
                        "pareto_ratio_fail_rate": _mean_or_none(
                            [
                                float(row["pareto_ratio_safety_fail"])
                                for row in prefix_rows
                                if isinstance(row.get("pareto_ratio_safety_fail"), int | float)
                            ]
                        ),
                        "spread_fail_rate": _mean_or_none(
                            [
                                float(row["spread_safety_fail"])
                                for row in prefix_rows
                                if isinstance(row.get("spread_safety_fail"), int | float)
                            ]
                        ),
                        "joint_safety_fail_rate": joint_fail_rate,
                    }
                )

    elif study.problem == "knapsack":
        none_variant = str(config.get("none_variant") or "none")
        repair_variant = str(config.get("repair_variant") or "repair_only")
        greedy_variant = str(config.get("greedy_variant") or "greedy_local_search")
        pair_rows = _knapsack_sequential_pair_rows(
            raw_rows,
            none_variant=none_variant,
            repair_variant=repair_variant,
            greedy_variant=greedy_variant,
        )
        scope_seed_rows = {
            "overall": _knapsack_sequential_seed_rows(pair_rows, case_group="overall"),
        }
        for case_group in sorted({str(row.get("case_group") or "overall") for row in pair_rows}):
            scope_seed_rows[case_group] = _knapsack_sequential_seed_rows(pair_rows, case_group=case_group)
        for case_group, seed_rows in scope_seed_rows.items():
            if not seed_rows:
                continue
            full_rows, _, _ = _prefix_seed_rows(seed_rows, max_stage)
            full_eval = _sum_or_none(
                [
                    float(row["actual_evaluations_used"])
                    for row in full_rows
                    if isinstance(row.get("actual_evaluations_used"), int | float)
                ]
            )
            for seed_count in seed_stages:
                prefix_rows, actual_seed_count, total_seed_count = _prefix_seed_rows(
                    seed_rows,
                    seed_count,
                )
                if not prefix_rows:
                    continue
                deltas = [float(row["paired_delta"]) for row in prefix_rows]
                ci_low, ci_high = _bootstrap_ci(
                    deltas,
                    resamples=bootstrap_replicates,
                    seed=bootstrap_seed + seed_count + len(case_group),
                )
                candidate_wins, baseline_wins, ties = _paired_sign_counts(deltas)
                actual_eval = _sum_or_none(
                    [
                        float(value)
                        for row in prefix_rows
                        for value in [row.get("actual_evaluations_used")]
                        if isinstance(value, int | float)
                    ]
                )
                decision_label, stage_action = _knapsack_sequential_decision(
                    stage_count=actual_seed_count,
                    max_stage=max_stage,
                    mean_delta=mean(deltas),
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
                rows.append(
                    {
                        "problem": study.problem,
                        "mode": "repair_note",
                        "scope": "overall" if case_group == "overall" else "case_group",
                        "case_group": case_group,
                        "seed_count": actual_seed_count,
                        "requested_seed_count": seed_count,
                        "total_seed_count_available": total_seed_count,
                        "sample_count": len(deltas),
                        "seed_ids": _paired_seed_ids(prefix_rows),
                        "paired_delta_mean": mean(deltas),
                        "paired_delta_median": median(deltas),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "bootstrap_ci_width": (
                            ci_high - ci_low
                            if ci_low is not None and ci_high is not None
                            else None
                        ),
                        "paired_win_count": candidate_wins,
                        "paired_loss_count": baseline_wins,
                        "paired_tie_count": ties,
                        "decision_label": decision_label,
                        "stage_action": stage_action,
                        "should_stop": 1.0 if stage_action == "stop" else 0.0,
                        "should_escalate": 1.0 if stage_action == "escalate" else 0.0,
                        "actual_evaluations_used": actual_eval,
                        "runtime_seconds": _sum_or_none(
                            [
                                float(value)
                                for row in prefix_rows
                                for value in [row.get("runtime_seconds")]
                                if isinstance(value, int | float)
                            ]
                        ),
                        "actual_eval_savings_vs_full_pct": (
                            (float(full_eval) - float(actual_eval)) / float(full_eval) * 100.0
                            if isinstance(full_eval, int | float)
                            and not isinstance(full_eval, bool)
                            and float(full_eval) != 0.0
                            and isinstance(actual_eval, int | float)
                            and not isinstance(actual_eval, bool)
                            else None
                        ),
                        "repair_gain_vs_none_mean": -mean(deltas),
                        "repair_gap_vs_greedy_mean": _mean_or_none(
                            [
                                float(value)
                                for row in prefix_rows
                                for value in [row.get("greedy_gap_vs_repair")]
                                if isinstance(value, int | float)
                            ]
                        ),
                        "initial_feasible_fraction_mean": _mean_or_none(
                            [
                                float(value)
                                for row in prefix_rows
                                for value in [row.get("initial_feasible_fraction")]
                                if isinstance(value, int | float)
                            ]
                        ),
                        "generations_to_first_feasible_mean": _mean_or_none(
                            [
                                float(value)
                                for row in prefix_rows
                                for value in [row.get("generations_to_first_feasible")]
                                if isinstance(value, int | float)
                            ]
                        ),
                    }
                )

    elif study.problem == "onemax":
        control_variant = str(config.get("control_variant") or "none")
        reference_variant = str(config.get("reference_variant") or "early_stop_reference")
        seed_rows = _onemax_sequential_seed_rows(
            raw_rows,
            control_variant=control_variant,
            reference_variant=reference_variant,
        )
        if not seed_rows:
            return []
        full_rows, _, _ = _prefix_seed_rows(seed_rows, max_stage)
        full_eval = _sum_or_none(
            [
                float(row["actual_evaluations_used"])
                for row in full_rows
                if isinstance(row.get("actual_evaluations_used"), int | float)
            ]
        )
        for seed_count in seed_stages:
            prefix_rows, actual_seed_count, total_seed_count = _prefix_seed_rows(seed_rows, seed_count)
            if not prefix_rows:
                continue
            deltas = [float(row["paired_delta"]) for row in prefix_rows]
            ci_low, ci_high = _bootstrap_ci(
                deltas,
                resamples=bootstrap_replicates,
                seed=bootstrap_seed + seed_count,
            )
            candidate_wins, baseline_wins, ties = _paired_sign_counts(deltas)
            actual_eval = _sum_or_none(
                [
                    float(value)
                    for row in prefix_rows
                    for value in [row.get("actual_evaluations_used")]
                    if isinstance(value, int | float)
                ]
            )
            decision_label, stage_action = _onemax_sequential_decision(
                mean_delta=mean(deltas),
                ci_low=ci_low,
                ci_high=ci_high,
                config=config,
            )
            rows.append(
                {
                    "problem": study.problem,
                    "mode": "control",
                    "scope": "overall",
                    "case_group": "overall",
                    "seed_count": actual_seed_count,
                    "requested_seed_count": seed_count,
                    "total_seed_count_available": total_seed_count,
                    "sample_count": len(deltas),
                    "seed_ids": _paired_seed_ids(prefix_rows),
                    "paired_delta_mean": mean(deltas),
                    "paired_delta_median": median(deltas),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "bootstrap_ci_width": (
                        ci_high - ci_low
                        if ci_low is not None and ci_high is not None
                        else None
                    ),
                    "paired_win_count": candidate_wins,
                    "paired_loss_count": baseline_wins,
                    "paired_tie_count": ties,
                    "decision_label": decision_label,
                    "stage_action": stage_action,
                    "should_stop": 1.0 if stage_action == "stop" else 0.0,
                    "should_escalate": 1.0 if stage_action == "escalate" else 0.0,
                    "actual_evaluations_used": actual_eval,
                    "runtime_seconds": _sum_or_none(
                        [
                            float(value)
                            for row in prefix_rows
                            for value in [row.get("runtime_seconds")]
                            if isinstance(value, int | float)
                        ]
                    ),
                    "actual_eval_savings_vs_full_pct": (
                        (float(full_eval) - float(actual_eval)) / float(full_eval) * 100.0
                        if isinstance(full_eval, int | float)
                        and not isinstance(full_eval, bool)
                        and float(full_eval) != 0.0
                        and isinstance(actual_eval, int | float)
                        and not isinstance(actual_eval, bool)
                        else None
                    ),
                    "control_target_hit_rate_mean": _mean_or_none(
                        [
                            float(value)
                            for row in prefix_rows
                            for value in [row.get("control_target_hit")]
                            if isinstance(value, int | float)
                        ]
                    ),
                }
            )

    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("mode") or ""),
            str(row.get("scope") or "overall"),
            str(row.get("case_group") or "overall"),
        )
        grouped_rows.setdefault(key, []).append(row)
    for members in grouped_rows.values():
        final_row = max(members, key=lambda row: int(row.get("seed_count") or 0))
        final_label = str(final_row.get("decision_label") or "inconclusive_even_after_10")
        for row in members:
            row["decision_flip_rate_vs_full"] = (
                0.0 if str(row.get("decision_label") or "") == final_label else 1.0
            )
            row["decision_stability_score"] = 1.0 - float(row["decision_flip_rate_vs_full"])
    return rows


def _tsp_fast_tail_pair_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _tsp_fast_tail_config(study)
    if config is None:
        return []

    quality_variant = config.get("quality_variant")
    baseline_fast_variant = config.get("baseline_fast_variant")
    if not isinstance(quality_variant, str) or not quality_variant.strip():
        return []
    if not isinstance(baseline_fast_variant, str) or not baseline_fast_variant.strip():
        return []
    quality_variant = quality_variant.strip()
    baseline_fast_variant = baseline_fast_variant.strip()
    candidate_variants = config.get("candidate_variants")
    if isinstance(candidate_variants, list):
        candidate_keys = {
            str(value).strip()
            for value in candidate_variants
            if isinstance(value, str) and value.strip()
        }
    else:
        candidate_keys = set()

    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in raw_rows:
        variant = _study_variant_key(row)
        if not variant:
            continue
        group_id = _pairwise_group_id(row)
        if group_id is None:
            continue
        grouped.setdefault(group_id, {})[variant] = row

    pair_rows: list[dict[str, Any]] = []
    for (_case_id, _seed), variants in sorted(grouped.items(), key=lambda item: item[0]):
        quality_row = variants.get(quality_variant)
        baseline_row = variants.get(baseline_fast_variant)
        if quality_row is None or baseline_row is None:
            continue
        quality_metric = quality_row.get("best_route_distance")
        baseline_metric = baseline_row.get("best_route_distance")
        if not isinstance(quality_metric, int | float) or isinstance(quality_metric, bool):
            continue
        if not isinstance(baseline_metric, int | float) or isinstance(baseline_metric, bool):
            continue
        if float(quality_metric) == 0.0 or float(baseline_metric) == 0.0:
            continue

        case_id = quality_row.get("case_id") or baseline_row.get("case_id") or ""
        case_group = quality_row.get("case_group") or baseline_row.get("case_group") or "overall"
        case_note = quality_row.get("case_note") or baseline_row.get("case_note") or ""
        seed = quality_row.get("seed")

        for variant_key, row in sorted(variants.items(), key=lambda item: item[0]):
            if variant_key == quality_variant:
                continue
            if candidate_keys and variant_key not in candidate_keys:
                continue
            metric_value = row.get("best_route_distance")
            if not isinstance(metric_value, int | float) or isinstance(metric_value, bool):
                continue
            metric_value = float(metric_value)
            pair_rows.append(
                {
                    "seed": seed,
                    "case_id": case_id,
                    "case_group": case_group,
                    "case_note": case_note,
                    "study_variant": variant_key,
                    "quality_variant": quality_variant,
                    "baseline_fast_variant": baseline_fast_variant,
                    "best_route_distance": metric_value,
                    "quality_best_route_distance": float(quality_metric),
                    "baseline_fast_best_route_distance": float(baseline_metric),
                    "route_distance_loss_pct_vs_quality": (
                        (metric_value - float(quality_metric)) / float(quality_metric) * 100.0
                    ),
                    "route_distance_delta_vs_quality": metric_value - float(quality_metric),
                    "route_distance_loss_pct_vs_current_fast": (
                        (metric_value - float(baseline_metric)) / float(baseline_metric) * 100.0
                    ),
                    "route_distance_delta_vs_current_fast": metric_value - float(baseline_metric),
                    "configured_budget": row.get("configured_budget"),
                    "actual_evaluations_used": row.get("actual_evaluations_used"),
                    "runtime_seconds": row.get("runtime_seconds"),
                    "population_size": row.get("population_size"),
                    "generations": row.get("generations"),
                    "mutation": row.get("mutation"),
                    "algorithm_options.seed_fraction": row.get("algorithm_options.seed_fraction"),
                }
            )
    return pair_rows


def _tsp_fast_tail_summary_rows(
    study: LocalStudy,
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _tsp_fast_tail_config(study)
    if config is None or not pair_rows:
        return []

    baseline_fast_variant = config.get("baseline_fast_variant")
    if not isinstance(baseline_fast_variant, str) or not baseline_fast_variant.strip():
        return []
    baseline_fast_variant = baseline_fast_variant.strip()

    def _numeric_members(members: list[dict[str, Any]], key: str) -> list[float]:
        return [
            float(value)
            for row in members
            for value in [row.get(key)]
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        variant = str(row.get("study_variant") or "")
        if variant:
            grouped.setdefault(variant, []).append(row)

    baseline_members = grouped.get(baseline_fast_variant, [])
    baseline_anti_losses = _numeric_members(
        [
            row
            for row in baseline_members
            if str(row.get("case_group") or "overall") == "anti_case"
        ],
        "route_distance_loss_pct_vs_quality",
    )
    baseline_anti_p90 = _quantile(
        baseline_anti_losses,
        0.9,
    )
    baseline_anti_p95 = _quantile(baseline_anti_losses, 0.95)
    baseline_anti_max = max(baseline_anti_losses) if baseline_anti_losses else None
    baseline_rescue_mean = _mean_or_none(
        _numeric_members(
            [
                row
                for row in baseline_members
                if str(row.get("case_group") or "overall") == "rescue_target"
            ],
            "route_distance_loss_pct_vs_quality",
        )
    )

    rows: list[dict[str, Any]] = []
    for variant, members in sorted(grouped.items()):
        overall_losses = _numeric_members(members, "route_distance_loss_pct_vs_quality")
        anti_members = [
            row for row in members if str(row.get("case_group") or "overall") == "anti_case"
        ]
        rescue_members = [
            row
            for row in members
            if str(row.get("case_group") or "overall") == "rescue_target"
        ]
        anti_losses = _numeric_members(anti_members, "route_distance_loss_pct_vs_quality")
        rescue_losses = _numeric_members(rescue_members, "route_distance_loss_pct_vs_quality")
        anti_p90 = _quantile(anti_losses, 0.9)
        anti_p95 = _quantile(anti_losses, 0.95)
        rescue_mean = _mean_or_none(rescue_losses)
        rows.append(
            {
                "scope": "overall",
                "case_group": "overall",
                "study_variant": variant,
                "run_count": len(members),
                "configured_budget_mean": _mean_or_none(_numeric_members(members, "configured_budget")),
                "actual_evaluations_used_mean": _mean_or_none(
                    _numeric_members(members, "actual_evaluations_used")
                ),
                "runtime_seconds_mean": _mean_or_none(_numeric_members(members, "runtime_seconds")),
                "mean_loss_pct": mean(overall_losses) if overall_losses else None,
                "median_loss_pct": median(overall_losses) if overall_losses else None,
                "p75_loss_pct": _quantile(overall_losses, 0.75),
                "p90_loss_pct": _quantile(overall_losses, 0.9),
                "p95_loss_pct": _quantile(overall_losses, 0.95),
                "max_loss_pct": max(overall_losses) if overall_losses else None,
                "mean_regret_vs_q": _mean_or_none(
                    _numeric_members(members, "route_distance_delta_vs_quality")
                ),
                "mean_regret_vs_current_fast": _mean_or_none(
                    _numeric_members(members, "route_distance_delta_vs_current_fast")
                ),
                "rescue_target_mean_loss_pct": rescue_mean,
                "anti_case_mean_loss_pct": _mean_or_none(anti_losses),
                "anti_case_p90_loss_pct": anti_p90,
                "anti_case_p95_loss_pct": anti_p95,
                "anti_case_max_loss_pct": max(anti_losses) if anti_losses else None,
                "tail_hardening_score": (
                    baseline_anti_p90 - anti_p90
                    if baseline_anti_p90 is not None and anti_p90 is not None
                    else None
                ),
                "tail_hardening_score_p95": (
                    baseline_anti_p95 - anti_p95
                    if baseline_anti_p95 is not None and anti_p95 is not None
                    else None
                ),
                "anti_case_p95_reduction_score": (
                    baseline_anti_p95 - anti_p95
                    if baseline_anti_p95 is not None and anti_p95 is not None
                    else None
                ),
                "anti_case_max_reduction_score": (
                    baseline_anti_max - max(anti_losses)
                    if baseline_anti_max is not None and anti_losses
                    else None
                ),
                "rescue_preservation_score": (
                    baseline_rescue_mean - rescue_mean
                    if baseline_rescue_mean is not None and rescue_mean is not None
                    else None
                ),
                "mutation": members[0].get("mutation"),
                "algorithm_options.seed_fraction": members[0].get("algorithm_options.seed_fraction"),
                "population_size": members[0].get("population_size"),
                "generations": members[0].get("generations"),
            }
        )
        for case_group, scoped_members in (
            ("rescue_target", rescue_members),
            ("anti_case", anti_members),
        ):
            scoped_losses = _numeric_members(scoped_members, "route_distance_loss_pct_vs_quality")
            if not scoped_losses:
                continue
            rows.append(
                {
                    "scope": "case_group",
                    "case_group": case_group,
                    "study_variant": variant,
                    "run_count": len(scoped_members),
                    "configured_budget_mean": _mean_or_none(
                        _numeric_members(scoped_members, "configured_budget")
                    ),
                    "actual_evaluations_used_mean": _mean_or_none(
                        _numeric_members(scoped_members, "actual_evaluations_used")
                    ),
                    "runtime_seconds_mean": _mean_or_none(
                        _numeric_members(scoped_members, "runtime_seconds")
                    ),
                    "mean_loss_pct": mean(scoped_losses),
                    "median_loss_pct": median(scoped_losses),
                    "p75_loss_pct": _quantile(scoped_losses, 0.75),
                    "p90_loss_pct": _quantile(scoped_losses, 0.9),
                    "p95_loss_pct": _quantile(scoped_losses, 0.95),
                    "max_loss_pct": max(scoped_losses),
                    "mean_regret_vs_q": _mean_or_none(
                        _numeric_members(scoped_members, "route_distance_delta_vs_quality")
                    ),
                    "mean_regret_vs_current_fast": _mean_or_none(
                        _numeric_members(scoped_members, "route_distance_delta_vs_current_fast")
                    ),
                    "tail_hardening_score": (
                        baseline_anti_p90 - _quantile(scoped_losses, 0.9)
                        if case_group == "anti_case" and baseline_anti_p90 is not None
                        else None
                    ),
                    "tail_hardening_score_p95": (
                        baseline_anti_p95 - _quantile(scoped_losses, 0.95)
                        if case_group == "anti_case" and baseline_anti_p95 is not None
                        else None
                    ),
                    "anti_case_p95_reduction_score": (
                        baseline_anti_p95 - _quantile(scoped_losses, 0.95)
                        if case_group == "anti_case" and baseline_anti_p95 is not None
                        else None
                    ),
                    "anti_case_max_reduction_score": (
                        baseline_anti_max - max(scoped_losses)
                        if case_group == "anti_case" and baseline_anti_max is not None
                        else None
                    ),
                    "rescue_preservation_score": (
                        baseline_rescue_mean - mean(scoped_losses)
                        if case_group == "rescue_target" and baseline_rescue_mean is not None
                        else None
                    ),
                    "mutation": scoped_members[0].get("mutation"),
                    "algorithm_options.seed_fraction": scoped_members[0].get(
                        "algorithm_options.seed_fraction"
                    ),
                    "population_size": scoped_members[0].get("population_size"),
                    "generations": scoped_members[0].get("generations"),
                }
            )
    return rows


def _zdt1_fast_hardening_pair_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _zdt1_fast_hardening_config(study)
    if config is None:
        return []

    quality_variant = config.get("quality_variant")
    baseline_fast_variant = config.get("baseline_fast_variant")
    if not isinstance(quality_variant, str) or not quality_variant.strip():
        return []
    if not isinstance(baseline_fast_variant, str) or not baseline_fast_variant.strip():
        return []
    quality_variant = quality_variant.strip()
    baseline_fast_variant = baseline_fast_variant.strip()

    candidate_keys = config.get("candidate_variants")
    if isinstance(candidate_keys, list):
        candidate_keys = {
            str(value).strip()
            for value in candidate_keys
            if isinstance(value, str) and str(value).strip()
        }
    else:
        candidate_keys = None

    pareto_drop_threshold = float(config.get("pareto_ratio_drop_threshold", 0.01))
    spread_degradation_threshold = float(config.get("spread_degradation_threshold", 0.05))

    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in raw_rows:
        variant = _study_variant_key(row)
        if not variant:
            continue
        group_id = _pairwise_group_id(row)
        if group_id is None:
            continue
        grouped.setdefault(group_id, {})[variant] = row

    pair_rows: list[dict[str, Any]] = []
    for (_case_id, _seed), variants in sorted(grouped.items(), key=lambda item: item[0]):
        quality_row = variants.get(quality_variant)
        baseline_row = variants.get(baseline_fast_variant)
        if quality_row is None or baseline_row is None:
            continue
        quality_metric = quality_row.get("hypervolume")
        baseline_metric = baseline_row.get("hypervolume")
        if not isinstance(quality_metric, int | float) or isinstance(quality_metric, bool):
            continue
        if not isinstance(baseline_metric, int | float) or isinstance(baseline_metric, bool):
            continue
        if float(quality_metric) == 0.0 or float(baseline_metric) == 0.0:
            continue

        case_id = quality_row.get("case_id") or baseline_row.get("case_id") or ""
        case_group = quality_row.get("case_group") or baseline_row.get("case_group") or "overall"
        case_note = quality_row.get("case_note") or baseline_row.get("case_note") or ""
        seed = quality_row.get("seed")
        quality_pareto = quality_row.get("pareto_ratio")
        quality_spread = quality_row.get("spread")

        for variant_key, row in sorted(variants.items(), key=lambda item: item[0]):
            if variant_key == quality_variant:
                continue
            if candidate_keys and variant_key not in candidate_keys:
                continue
            metric_value = row.get("hypervolume")
            if not isinstance(metric_value, int | float) or isinstance(metric_value, bool):
                continue
            metric_value = float(metric_value)

            candidate_pareto = row.get("pareto_ratio")
            candidate_spread = row.get("spread")
            pareto_delta_vs_quality = (
                float(candidate_pareto) - float(quality_pareto)
                if isinstance(candidate_pareto, int | float)
                and not isinstance(candidate_pareto, bool)
                and isinstance(quality_pareto, int | float)
                and not isinstance(quality_pareto, bool)
                else None
            )
            spread_delta_vs_quality = (
                float(candidate_spread) - float(quality_spread)
                if isinstance(candidate_spread, int | float)
                and not isinstance(candidate_spread, bool)
                and isinstance(quality_spread, int | float)
                and not isinstance(quality_spread, bool)
                else None
            )
            pareto_fail = (
                pareto_delta_vs_quality is not None
                and pareto_delta_vs_quality < -pareto_drop_threshold
            )
            spread_fail = (
                spread_delta_vs_quality is not None
                and spread_delta_vs_quality > spread_degradation_threshold
            )
            pair_rows.append(
                {
                    "seed": seed,
                    "case_id": case_id,
                    "case_group": case_group,
                    "case_note": case_note,
                    "study_variant": variant_key,
                    "quality_variant": quality_variant,
                    "baseline_fast_variant": baseline_fast_variant,
                    "hypervolume": metric_value,
                    "quality_hypervolume": float(quality_metric),
                    "baseline_fast_hypervolume": float(baseline_metric),
                    "hv_loss_pct_vs_quality": (
                        (float(quality_metric) - metric_value) / float(quality_metric) * 100.0
                    ),
                    "hv_regret_vs_q": float(quality_metric) - metric_value,
                    "hv_loss_pct_vs_current_fast": (
                        (float(baseline_metric) - metric_value) / float(baseline_metric) * 100.0
                    ),
                    "hv_regret_vs_current_fast": float(baseline_metric) - metric_value,
                    "pareto_ratio_delta_vs_quality": pareto_delta_vs_quality,
                    "spread_delta_vs_quality": spread_delta_vs_quality,
                    "pareto_ratio_safety_fail": 1.0 if pareto_fail else 0.0,
                    "spread_safety_fail": 1.0 if spread_fail else 0.0,
                    "joint_safety_fail": 1.0 if pareto_fail or spread_fail else 0.0,
                    "configured_budget": row.get("configured_budget"),
                    "actual_evaluations_used": row.get("actual_evaluations_used"),
                    "runtime_seconds": row.get("runtime_seconds"),
                    "population_size": row.get("population_size"),
                    "generations": row.get("generations"),
                    "algorithm_options.refresh_fraction": row.get(
                        "algorithm_options.refresh_fraction"
                    ),
                    "algorithm_options.adaptation_cooldown": row.get(
                        "algorithm_options.adaptation_cooldown"
                    ),
                }
            )
    return pair_rows


def _zdt1_fast_hardening_summary_rows(
    study: LocalStudy,
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _zdt1_fast_hardening_config(study)
    if config is None or not pair_rows:
        return []

    baseline_fast_variant = str(config.get("baseline_fast_variant") or "").strip()
    if not baseline_fast_variant:
        return []

    def _numeric_members(members: list[dict[str, Any]], key: str) -> list[float]:
        return [
            float(value)
            for row in members
            for value in [row.get(key)]
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        variant = str(row.get("study_variant") or "")
        if variant:
            grouped.setdefault(variant, []).append(row)

    baseline_members = grouped.get(baseline_fast_variant, [])
    baseline_joint_fail = _mean_or_none(_numeric_members(baseline_members, "joint_safety_fail"))
    baseline_spread_fail = _mean_or_none(_numeric_members(baseline_members, "spread_safety_fail"))
    baseline_mean_loss = _mean_or_none(
        _numeric_members(baseline_members, "hv_loss_pct_vs_quality")
    )
    baseline_spread_deltas = _numeric_members(baseline_members, "spread_delta_vs_quality")
    baseline_spread_p90 = _quantile(baseline_spread_deltas, 0.9)
    baseline_spread_p95 = _quantile(baseline_spread_deltas, 0.95)

    rows: list[dict[str, Any]] = []
    for variant, members in sorted(grouped.items()):
        losses = _numeric_members(members, "hv_loss_pct_vs_quality")
        if not losses:
            continue
        joint_fail_rate = _mean_or_none(_numeric_members(members, "joint_safety_fail"))
        spread_fail_rate = _mean_or_none(_numeric_members(members, "spread_safety_fail"))
        mean_loss = mean(losses)
        spread_deltas = _numeric_members(members, "spread_delta_vs_quality")
        rows.append(
            {
                "scope": "overall",
                "case_group": "overall",
                "study_variant": variant,
                "run_count": len(members),
                "configured_budget_mean": _mean_or_none(_numeric_members(members, "configured_budget")),
                "actual_evaluations_used_mean": _mean_or_none(
                    _numeric_members(members, "actual_evaluations_used")
                ),
                "runtime_seconds_mean": _mean_or_none(_numeric_members(members, "runtime_seconds")),
                "mean_loss_pct": mean_loss,
                "median_loss_pct": median(losses),
                "p75_loss_pct": _quantile(losses, 0.75),
                "p90_loss_pct": _quantile(losses, 0.9),
                "p95_loss_pct": _quantile(losses, 0.95),
                "max_loss_pct": max(losses),
                "mean_regret_vs_q": _mean_or_none(_numeric_members(members, "hv_regret_vs_q")),
                "mean_regret_vs_current_fast": _mean_or_none(
                    _numeric_members(members, "hv_regret_vs_current_fast")
                ),
                "pareto_ratio_delta_mean": _mean_or_none(
                    _numeric_members(members, "pareto_ratio_delta_vs_quality")
                ),
                "spread_delta_mean": _mean_or_none(
                    spread_deltas
                ),
                "spread_delta_p75": _quantile(spread_deltas, 0.75),
                "spread_delta_p90": _quantile(spread_deltas, 0.9),
                "spread_delta_p95": _quantile(spread_deltas, 0.95),
                "spread_delta_max": max(spread_deltas) if spread_deltas else None,
                "pareto_ratio_fail_rate": _mean_or_none(
                    _numeric_members(members, "pareto_ratio_safety_fail")
                ),
                "spread_fail_rate": spread_fail_rate,
                "joint_safety_fail_rate": joint_fail_rate,
                "safety_hardening_score": (
                    baseline_joint_fail - joint_fail_rate
                    if baseline_joint_fail is not None and joint_fail_rate is not None
                    else None
                ),
                "spread_hardening_score": (
                    baseline_spread_fail - spread_fail_rate
                    if baseline_spread_fail is not None and spread_fail_rate is not None
                    else None
                ),
                "spread_tail_reduction_score": (
                    baseline_spread_p90 - _quantile(spread_deltas, 0.9)
                    if baseline_spread_p90 is not None and spread_deltas
                    else None
                ),
                "spread_tail_reduction_score_p95": (
                    baseline_spread_p95 - _quantile(spread_deltas, 0.95)
                    if baseline_spread_p95 is not None and spread_deltas
                    else None
                ),
                "joint_fail_reduction_score": (
                    baseline_joint_fail - joint_fail_rate
                    if baseline_joint_fail is not None and joint_fail_rate is not None
                    else None
                ),
                "hv_preservation_score": (
                    baseline_mean_loss - mean_loss
                    if baseline_mean_loss is not None
                    else None
                ),
                "population_size": members[0].get("population_size"),
                "generations": members[0].get("generations"),
                "algorithm_options.refresh_fraction": members[0].get(
                    "algorithm_options.refresh_fraction"
                ),
                "algorithm_options.adaptation_cooldown": members[0].get(
                    "algorithm_options.adaptation_cooldown"
                ),
            }
            )
    return rows


def _zdt1_spread_candidate_validation_rows(
    study: LocalStudy,
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _zdt1_spread_candidate_validation_config(study)
    if config is None or not pair_rows:
        return []

    baseline_variant = str(
        config.get("baseline_fast_variant")
        or config.get("baseline_variant")
        or ""
    ).strip()
    candidate_variant = str(config.get("candidate_variant") or "").strip()
    if not baseline_variant or not candidate_variant:
        return []

    variants = {baseline_variant, candidate_variant}
    filtered_rows = [
        row
        for row in pair_rows
        if str(row.get("study_variant") or "").strip() in variants
    ]
    if not filtered_rows:
        return []

    slice_config = config.get("slices")
    seed_slices: dict[str, set[int]] = {}
    if isinstance(slice_config, dict):
        for slice_name, values in slice_config.items():
            if not isinstance(slice_name, str) or not slice_name.strip():
                continue
            if not isinstance(values, list):
                continue
            parsed = {
                int(value)
                for value in values
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
            if parsed:
                seed_slices[slice_name.strip()] = parsed

    def _numeric_members(members: list[dict[str, Any]], key: str) -> list[float]:
        return [
            float(value)
            for row in members
            for value in [row.get(key)]
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]

    def _scope_rows(scope: str, slice_name: str, members: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in members:
            grouped.setdefault(str(row.get("study_variant") or "").strip(), []).append(row)

        baseline_members = grouped.get(baseline_variant, [])
        candidate_members = grouped.get(candidate_variant, [])

        baseline_spread_p90 = _quantile(_numeric_members(baseline_members, "spread_delta_vs_quality"), 0.9)
        baseline_spread_p95 = _quantile(_numeric_members(baseline_members, "spread_delta_vs_quality"), 0.95)
        baseline_mean_loss = _mean_or_none(_numeric_members(baseline_members, "hv_loss_pct_vs_quality"))
        baseline_p90_loss = _quantile(_numeric_members(baseline_members, "hv_loss_pct_vs_quality"), 0.9)
        baseline_joint_fail = _mean_or_none(_numeric_members(baseline_members, "joint_safety_fail"))
        baseline_pareto_mean = _mean_or_none(
            _numeric_members(baseline_members, "pareto_ratio_delta_vs_quality")
        )

        rows: list[dict[str, Any]] = []
        for variant, variant_members in sorted(grouped.items()):
            losses = _numeric_members(variant_members, "hv_loss_pct_vs_quality")
            if not losses:
                continue
            spread_deltas = _numeric_members(variant_members, "spread_delta_vs_quality")
            mean_loss = mean(losses)
            p90_loss = _quantile(losses, 0.9)
            p95_loss = _quantile(losses, 0.95)
            spread_fail_rate = _mean_or_none(_numeric_members(variant_members, "spread_safety_fail"))
            joint_fail_rate = _mean_or_none(_numeric_members(variant_members, "joint_safety_fail"))
            pareto_delta_mean = _mean_or_none(
                _numeric_members(variant_members, "pareto_ratio_delta_vs_quality")
            )
            spread_delta_p90 = _quantile(spread_deltas, 0.9)
            spread_delta_p95 = _quantile(spread_deltas, 0.95)
            rows.append(
                {
                    "scope": scope,
                    "slice": slice_name,
                    "study_variant": variant,
                    "run_count": len(variant_members),
                    "mean_loss_pct": mean_loss,
                    "p75_loss_pct": _quantile(losses, 0.75),
                    "p90_loss_pct": p90_loss,
                    "p95_loss_pct": p95_loss,
                    "max_loss_pct": max(losses),
                    "pareto_ratio_delta_mean": pareto_delta_mean,
                    "spread_delta_mean": _mean_or_none(spread_deltas),
                    "spread_delta_p75": _quantile(spread_deltas, 0.75),
                    "spread_delta_p90": spread_delta_p90,
                    "spread_delta_p95": spread_delta_p95,
                    "spread_delta_max": max(spread_deltas) if spread_deltas else None,
                    "spread_fail_rate": spread_fail_rate,
                    "joint_safety_fail_rate": joint_fail_rate,
                    "configured_budget_mean": _mean_or_none(
                        _numeric_members(variant_members, "configured_budget")
                    ),
                    "actual_evaluations_used_mean": _mean_or_none(
                        _numeric_members(variant_members, "actual_evaluations_used")
                    ),
                    "runtime_seconds_mean": _mean_or_none(
                        _numeric_members(variant_members, "runtime_seconds")
                    ),
                    "hv_preservation_score": (
                        baseline_mean_loss - mean_loss
                        if variant != baseline_variant and baseline_mean_loss is not None
                        else 0.0 if variant == baseline_variant else None
                    ),
                    "spread_tail_reduction_score": (
                        baseline_spread_p90 - spread_delta_p90
                        if variant != baseline_variant
                        and baseline_spread_p90 is not None
                        and spread_delta_p90 is not None
                        else 0.0 if variant == baseline_variant else None
                    ),
                    "spread_tail_reduction_score_p95": (
                        baseline_spread_p95 - spread_delta_p95
                        if variant != baseline_variant
                        and baseline_spread_p95 is not None
                        and spread_delta_p95 is not None
                        else 0.0 if variant == baseline_variant else None
                    ),
                    "joint_non_regression_score": (
                        baseline_joint_fail - joint_fail_rate
                        if variant != baseline_variant
                        and baseline_joint_fail is not None
                        and joint_fail_rate is not None
                        else 0.0 if variant == baseline_variant else None
                    ),
                    "pareto_non_regression_score": (
                        pareto_delta_mean - baseline_pareto_mean
                        if variant != baseline_variant
                        and pareto_delta_mean is not None
                        and baseline_pareto_mean is not None
                        else 0.0 if variant == baseline_variant else None
                    ),
                    "candidate_decision_hint": (
                        "baseline"
                        if variant == baseline_variant
                        else (
                            "replace_fast_default"
                            if (
                                spread_fail_rate is not None
                                and baseline_members
                                and baseline_p90_loss is not None
                                and p90_loss is not None
                                and (baseline_pareto_mean is None or pareto_delta_mean is None or pareto_delta_mean >= baseline_pareto_mean - 0.002)
                                and joint_fail_rate is not None
                                and baseline_joint_fail is not None
                                and spread_fail_rate <= (_mean_or_none(_numeric_members(baseline_members, "spread_safety_fail")) or 0.0) - 0.05
                                and (spread_delta_p95 is None or baseline_spread_p95 is None or spread_delta_p95 <= baseline_spread_p95 - 0.005)
                                and p90_loss <= baseline_p90_loss + 0.02
                                and joint_fail_rate <= baseline_joint_fail
                            )
                            else (
                                "note_only_stress_slice"
                                if (
                                    spread_fail_rate is not None
                                    and baseline_members
                                    and joint_fail_rate is not None
                                    and baseline_joint_fail is not None
                                    and (spread_fail_rate < (_mean_or_none(_numeric_members(baseline_members, "spread_safety_fail")) or 0.0) or (
                                        spread_delta_p90 is not None
                                        and baseline_spread_p90 is not None
                                        and spread_delta_p90 < baseline_spread_p90
                                    ))
                                    and joint_fail_rate <= baseline_joint_fail
                                    and (baseline_p90_loss is None or p90_loss is None or p90_loss <= baseline_p90_loss + 0.05)
                                )
                                else "monitor_only"
                            )
                        )
                    ),
                }
            )
        return rows

    validation_rows: list[dict[str, Any]] = []
    validation_rows.extend(_scope_rows("overall", "overall", filtered_rows))
    for slice_name, slice_seeds in sorted(seed_slices.items()):
        members = [
            row
            for row in filtered_rows
            if isinstance(row.get("seed"), int | float)
            and not isinstance(row.get("seed"), bool)
            and int(row["seed"]) in slice_seeds
        ]
        if members:
            validation_rows.extend(_scope_rows("slice", slice_name, members))
    return validation_rows


def _render_tsp_tail_freeze_summary(
    study: LocalStudy,
    tsp_fast_tail_summary_rows: list[dict[str, Any]],
    failure_hypothesis_rows: list[dict[str, Any]],
) -> str:
    overall_current = next(
        (
            row
            for row in tsp_fast_tail_summary_rows
            if str(row.get("scope") or "") == "overall"
            and str(row.get("study_variant") or "") == "current_fast"
        ),
        None,
    )
    anti_case_row = next(
        (
            row
            for row in tsp_fast_tail_summary_rows
            if str(row.get("scope") or "") == "case_group"
            and str(row.get("case_group") or "") == "anti_case"
            and str(row.get("study_variant") or "") == "current_fast"
        ),
        None,
    )
    rescue_row = next(
        (
            row
            for row in tsp_fast_tail_summary_rows
            if str(row.get("scope") or "") == "case_group"
            and str(row.get("case_group") or "") == "rescue_target"
            and str(row.get("study_variant") or "") == "current_fast"
        ),
        None,
    )
    late_hypothesis = next(
        (
            row
            for row in failure_hypothesis_rows
            if str(row.get("hypothesis_id") or "") == "tsp_anticase_late_refinement_deficit"
        ),
        None,
    )
    seed_hypothesis = next(
        (
            row
            for row in failure_hypothesis_rows
            if str(row.get("hypothesis_id") or "") == "tsp_anticase_seed_lockin_secondary"
        ),
        None,
    )
    lines = ["# TSP Irreducible Tail Freeze", ""]
    lines.append(
        "- Decision: freeze the current TSP fast default as a budget-first exploratory profile and treat the remaining anti-case p95/max as a protocol limitation under the current fixed stack."
    )
    lines.append(
        "- Protocol: corridor-like or anti-case suspicion still goes straight to `Q`, and quality-sensitive final runs stay on `Q 8-10`."
    )
    lines.append("")
    lines.append("## Current Readout")
    lines.append("")
    if overall_current is not None:
        lines.append(
            f"- Overall mean/p90/p95/max loss vs Q: `{_format_number(overall_current.get('mean_loss_pct'))}` / "
            f"`{_format_number(overall_current.get('p90_loss_pct'))}` / "
            f"`{_format_number(overall_current.get('p95_loss_pct'))}` / "
            f"`{_format_number(overall_current.get('max_loss_pct'))}`"
        )
    if anti_case_row is not None:
        lines.append(
            f"- Anti-case mean/p90/p95/max: `{_format_number(anti_case_row.get('mean_loss_pct'))}` / "
            f"`{_format_number(anti_case_row.get('p90_loss_pct'))}` / "
            f"`{_format_number(anti_case_row.get('p95_loss_pct'))}` / "
            f"`{_format_number(anti_case_row.get('max_loss_pct'))}`"
        )
    if rescue_row is not None:
        lines.append(
            f"- Rescue-target mean/p90/p95/max: `{_format_number(rescue_row.get('mean_loss_pct'))}` / "
            f"`{_format_number(rescue_row.get('p90_loss_pct'))}` / "
            f"`{_format_number(rescue_row.get('p95_loss_pct'))}` / "
            f"`{_format_number(rescue_row.get('max_loss_pct'))}`"
        )
    lines.append("")
    lines.append("## Mechanism Read")
    lines.append("")
    lines.append(
        f"- `late_refinement_deficit`: `{_format_number((late_hypothesis or {}).get('confirmed_or_weakened'))}`"
    )
    lines.append(
        f"- `seed_lockin_and_diversity_collapse`: `{_format_number((seed_hypothesis or {}).get('confirmed_or_weakened'))}`"
    )
    lines.append(
        "- Re-open TSP fast tuning only if a new mechanism hypothesis appears beyond the current same-budget population/generation contour and seed-fraction secondary checks."
    )
    lines.append("")
    return "\n".join(lines)


def _zdt1_spread_boundary_rows(
    study: LocalStudy,
    validation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = _zdt1_spread_candidate_validation_config(study)
    if config is None or not validation_rows:
        return []

    baseline_variant = str(
        config.get("baseline_fast_variant")
        or config.get("baseline_variant")
        or "current_fast"
    ).strip()
    candidate_variant = str(config.get("candidate_variant") or "").strip()
    if not baseline_variant or not candidate_variant:
        return []

    def _pick(scope: str, slice_name: str, variant: str) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in validation_rows
                if str(row.get("scope") or "") == scope
                and str(row.get("slice") or "") == slice_name
                and str(row.get("study_variant") or "") == variant
            ),
            None,
        )

    def _maybe_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    baseline_overall = _pick("overall", "overall", baseline_variant)
    candidate_overall = _pick("overall", "overall", candidate_variant)
    if baseline_overall is None or candidate_overall is None:
        return []

    baseline_spread_stress = _pick("slice", "spread_stress", baseline_variant)
    candidate_spread_stress = _pick("slice", "spread_stress", candidate_variant)
    baseline_stable = _pick("slice", "stable_contrast", baseline_variant)
    candidate_stable = _pick("slice", "stable_contrast", candidate_variant)
    baseline_normal = _pick("slice", "normal_holdout", baseline_variant)
    candidate_normal = _pick("slice", "normal_holdout", candidate_variant)
    baseline_joint = _pick("slice", "joint_non_regression", baseline_variant)
    candidate_joint = _pick("slice", "joint_non_regression", candidate_variant)

    def _score(candidate_row: dict[str, Any] | None, baseline_row: dict[str, Any] | None, key: str) -> float | None:
        candidate_value = _maybe_float((candidate_row or {}).get(key))
        baseline_value = _maybe_float((baseline_row or {}).get(key))
        if candidate_value is None or baseline_value is None:
            return None
        return baseline_value - candidate_value

    spread_gain = _score(candidate_spread_stress, baseline_spread_stress, "spread_fail_rate")
    spread_tail_gain = _score(candidate_spread_stress, baseline_spread_stress, "spread_delta_p95")
    joint_non_regression = _score(candidate_joint, baseline_joint, "joint_safety_fail_rate")
    stable_non_regression = _score(candidate_stable, baseline_stable, "spread_fail_rate")
    normal_non_regression = _score(candidate_normal, baseline_normal, "spread_fail_rate")
    pareto_non_regression = _score(candidate_overall, baseline_overall, "pareto_ratio_delta_mean")
    hv_preservation = _score(candidate_overall, baseline_overall, "mean_loss_pct")
    hv_tail_preservation = _score(candidate_overall, baseline_overall, "p90_loss_pct")

    decision = "monitor_only"
    if (
        spread_gain is not None
        and spread_gain >= 0.15
        and (spread_tail_gain is None or spread_tail_gain >= 0.01)
        and (stable_non_regression is None or stable_non_regression >= -0.05)
        and (normal_non_regression is None or normal_non_regression >= -0.05)
        and (joint_non_regression is None or joint_non_regression >= -0.05)
        and (hv_preservation is None or hv_preservation >= -0.02)
        and (hv_tail_preservation is None or hv_tail_preservation >= -0.02)
        and (pareto_non_regression is None or pareto_non_regression >= -0.002)
    ):
        decision = "replace_fast_default"
    elif (
        spread_gain is not None
        and spread_gain > 0.0
        and (spread_tail_gain is None or spread_tail_gain >= 0.0)
        and (joint_non_regression is None or joint_non_regression >= 0.0)
        and (
            (stable_non_regression is not None and stable_non_regression < 0.0)
            or (normal_non_regression is not None and normal_non_regression < 0.0)
        )
    ):
        decision = "note_only_stress_slice"
    elif (
        spread_gain is not None
        and spread_gain <= 0.0
        and (stable_non_regression is None or stable_non_regression < 0.0)
        and (normal_non_regression is None or normal_non_regression < 0.0)
    ):
        decision = "retire_candidate"

    rows: list[dict[str, Any]] = []
    for row in validation_rows:
        enriched = dict(row)
        enriched["stable_slice_non_regression_score"] = stable_non_regression
        enriched["normal_slice_non_regression_score"] = normal_non_regression
        enriched["joint_non_regression_score_boundary"] = joint_non_regression
        enriched["pareto_non_regression_score_boundary"] = pareto_non_regression
        enriched["hv_tail_preservation_score"] = hv_tail_preservation
        enriched["boundary_decision"] = decision
        rows.append(enriched)
    return rows


def _render_zdt1_spread_boundary_notes(
    study: LocalStudy,
    boundary_rows: list[dict[str, Any]],
) -> str:
    candidate_overall = next(
        (
            row
            for row in boundary_rows
            if str(row.get("scope") or "") == "overall"
            and str(row.get("study_variant") or "") == "spread_pg_pop41_gen88"
        ),
        None,
    )
    baseline_overall = next(
        (
            row
            for row in boundary_rows
            if str(row.get("scope") or "") == "overall"
            and str(row.get("study_variant") or "") == "current_fast"
        ),
        None,
    )
    spread_stress_candidate = next(
        (
            row
            for row in boundary_rows
            if str(row.get("scope") or "") == "slice"
            and str(row.get("slice") or "") == "spread_stress"
            and str(row.get("study_variant") or "") == "spread_pg_pop41_gen88"
        ),
        None,
    )
    stable_candidate = next(
        (
            row
            for row in boundary_rows
            if str(row.get("scope") or "") == "slice"
            and str(row.get("slice") or "") == "stable_contrast"
            and str(row.get("study_variant") or "") == "spread_pg_pop41_gen88"
        ),
        None,
    )
    normal_candidate = next(
        (
            row
            for row in boundary_rows
            if str(row.get("scope") or "") == "slice"
            and str(row.get("slice") or "") == "normal_holdout"
            and str(row.get("study_variant") or "") == "spread_pg_pop41_gen88"
        ),
        None,
    )
    joint_candidate = next(
        (
            row
            for row in boundary_rows
            if str(row.get("scope") or "") == "slice"
            and str(row.get("slice") or "") == "joint_non_regression"
            and str(row.get("study_variant") or "") == "spread_pg_pop41_gen88"
        ),
        None,
    )
    decision = str((candidate_overall or {}).get("boundary_decision") or "monitor_only")

    lines = ["# ZDT1 Spread Candidate Boundary Notes", ""]
    lines.append(f"- Decision: `{decision}`")
    if decision == "replace_fast_default":
        lines.append(
            "- The spread candidate cleared spread-stress gains and non-regression on stable/normal slices well enough to justify a same-name fast replacement."
        )
    elif decision == "note_only_stress_slice":
        lines.append(
            "- The spread candidate is useful only on the pinned spread-stress slice; it does not hold cleanly enough on stable/normal rows to replace the default fast profile."
        )
    elif decision == "retire_candidate":
        lines.append(
            "- The spread candidate does not justify further attention because the spread-stress gain is too weak relative to the stable/normal regressions."
        )
    else:
        lines.append(
            "- The spread candidate remains interesting, but the evidence is still too mixed to treat it as either a replacement or a clean slice-conditioned note."
        )
    lines.append("- Protocol: keep final safety on `Q`, and keep current `F` as the canonical budget-first default unless a wider non-regression story appears.")
    lines.append("")
    lines.append("## Slice Readout")
    lines.append("")
    if spread_stress_candidate is not None:
        lines.append(
            f"- Spread-stress fail/joint/p95 spread delta: `{_format_number(spread_stress_candidate.get('spread_fail_rate'))}` / "
            f"`{_format_number(spread_stress_candidate.get('joint_safety_fail_rate'))}` / "
            f"`{_format_number(spread_stress_candidate.get('spread_delta_p95'))}`"
        )
    if stable_candidate is not None:
        lines.append(
            f"- Stable contrast fail/joint/p90 HV loss: `{_format_number(stable_candidate.get('spread_fail_rate'))}` / "
            f"`{_format_number(stable_candidate.get('joint_safety_fail_rate'))}` / "
            f"`{_format_number(stable_candidate.get('p90_loss_pct'))}`"
        )
    if normal_candidate is not None:
        lines.append(
            f"- Normal holdout fail/joint/p90 HV loss: `{_format_number(normal_candidate.get('spread_fail_rate'))}` / "
            f"`{_format_number(normal_candidate.get('joint_safety_fail_rate'))}` / "
            f"`{_format_number(normal_candidate.get('p90_loss_pct'))}`"
        )
    if joint_candidate is not None:
        lines.append(
            f"- Joint non-regression fail/joint/p90 HV loss: `{_format_number(joint_candidate.get('spread_fail_rate'))}` / "
            f"`{_format_number(joint_candidate.get('joint_safety_fail_rate'))}` / "
            f"`{_format_number(joint_candidate.get('p90_loss_pct'))}`"
        )
    if candidate_overall is not None and baseline_overall is not None:
        lines.append("")
        lines.append("## Overall Boundary Summary")
        lines.append("")
        lines.append(
            f"- Current F mean/p90/p95 HV loss: `{_format_number(baseline_overall.get('mean_loss_pct'))}` / "
            f"`{_format_number(baseline_overall.get('p90_loss_pct'))}` / "
            f"`{_format_number(baseline_overall.get('p95_loss_pct'))}`"
        )
        lines.append(
            f"- Candidate mean/p90/p95 HV loss: `{_format_number(candidate_overall.get('mean_loss_pct'))}` / "
            f"`{_format_number(candidate_overall.get('p90_loss_pct'))}` / "
            f"`{_format_number(candidate_overall.get('p95_loss_pct'))}`"
        )
        lines.append(
            f"- Candidate stable/normal non-regression scores: "
            f"`{_format_number(candidate_overall.get('stable_slice_non_regression_score'))}` / "
            f"`{_format_number(candidate_overall.get('normal_slice_non_regression_score'))}`"
        )
    lines.append("")
    lines.append(
        "- Boundary closeout read: spread_pg_pop41_gen88 buys real spread-stress relief, but current F stays canonical unless those gains survive stable and normal holdouts without reopening joint or HV tail penalties."
    )
    lines.append("")
    return "\n".join(lines)


def _two_stage_gate_group_key(row: dict[str, Any]) -> tuple[str, int] | None:
    seed = row.get("seed")
    if not isinstance(seed, int | float):
        return None
    case_id = row.get("case_id")
    case_key = case_id.strip() if isinstance(case_id, str) and case_id.strip() else "__study__"
    return (case_key, int(seed))


def _two_stage_gate_profile_role(row: dict[str, Any]) -> str | None:
    study_variant = row.get("study_variant")
    if isinstance(study_variant, str):
        normalized = study_variant.strip()
        if normalized == "always_canonical":
            return "canonical"
        if normalized == "always_fast":
            return "fast"
    variant_label = row.get("variant_label")
    if isinstance(variant_label, str):
        if "study_variant=always_canonical" in variant_label:
            return "canonical"
        if "study_variant=always_fast" in variant_label:
            return "fast"
    return None


def _annotate_two_stage_gate_rows(problem: str, raw_rows: list[dict[str, Any]]) -> None:
    if problem not in {"tsp", "zdt1"} or not raw_rows:
        return

    role_values: dict[tuple[str, float, int], dict[str, float]] = {}
    for row in raw_rows:
        role = _two_stage_gate_profile_role(row)
        group_key = _two_stage_gate_group_key(row)
        if role is None or group_key is None:
            continue
        if problem == "tsp":
            metric_value = row.get("best_route_distance")
        else:
            metric_value = row.get("hypervolume")
        if isinstance(metric_value, int | float) and not isinstance(metric_value, bool):
            role_values.setdefault(group_key, {})[role] = float(metric_value)

    for row in raw_rows:
        group_key = _two_stage_gate_group_key(row)
        if group_key is None:
            continue
        references = role_values.get(group_key, {})
        if problem == "tsp":
            metric_value = row.get("best_route_distance")
            lower_is_better = True
        else:
            metric_value = row.get("hypervolume")
            lower_is_better = False
        if not isinstance(metric_value, int | float) or isinstance(metric_value, bool):
            continue
        metric_float = float(metric_value)
        canonical_value = references.get("canonical")
        fast_value = references.get("fast")
        if canonical_value is not None:
            row["reference_canonical_metric"] = canonical_value
            row["regret_vs_canonical_profile"] = (
                metric_float - canonical_value
                if lower_is_better
                else canonical_value - metric_float
            )
        if fast_value is not None:
            row["reference_fast_metric"] = fast_value
            row["regret_vs_fast_profile"] = (
                metric_float - fast_value if lower_is_better else fast_value - metric_float
            )
        pilot_fraction = row.get("pilot_budget_fraction")
        if not isinstance(pilot_fraction, (int, float)) or isinstance(pilot_fraction, bool):
            continue
        escalation_flag = bool(row.get("escalation_triggered"))
        if canonical_value is None or fast_value is None:
            continue
        canonical_better = (
            canonical_value < fast_value - 1e-9
            if lower_is_better
            else canonical_value > fast_value + 1e-9
        )
        fast_better = (
            fast_value < canonical_value - 1e-9
            if lower_is_better
            else fast_value > canonical_value + 1e-9
        )
        row["false_escalation"] = 1.0 if escalation_flag and not canonical_better else 0.0
        row["false_keep"] = 1.0 if (not escalation_flag) and canonical_better else 0.0
        if lower_is_better:
            row["smart_gate_oracle_gap"] = metric_float - min(canonical_value, fast_value)
        else:
            row["smart_gate_oracle_gap"] = max(canonical_value, fast_value) - metric_float


def _triage_candidate_role(row: dict[str, Any]) -> tuple[str, str] | None:
    study_variant = row.get("study_variant")
    if not isinstance(study_variant, str):
        return None
    normalized = study_variant.strip()
    for suffix, role in (("__canonical", "canonical"), ("__fast", "fast")):
        if normalized.endswith(suffix):
            candidate_id = normalized[: -len(suffix)].strip("_")
            if candidate_id:
                return candidate_id, role
    return None


def _triage_metric_value(problem: str, row: dict[str, Any]) -> float | None:
    key = "best_route_distance" if problem == "tsp" else "hypervolume"
    value = row.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _triage_total_actual_evaluations(row: dict[str, Any]) -> float | None:
    value = row.get("total_actual_evaluations_used", row.get("actual_evaluations_used"))
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _triage_group_id(problem: str, row: dict[str, Any]) -> str:
    if problem == "tsp":
        case_id = row.get("case_id")
        if isinstance(case_id, str) and case_id.strip():
            return case_id.strip()
    return "__study__"


def _triage_case_group(problem: str, row: dict[str, Any]) -> str:
    if problem == "tsp":
        case_group = row.get("case_group")
        if isinstance(case_group, str) and case_group.strip():
            return case_group.strip()
    return "overall"


def _candidate_metric_order(
    scores: dict[str, float],
    *,
    lower_is_better: bool,
) -> list[tuple[str, float]]:
    if lower_is_better:
        return sorted(scores.items(), key=lambda item: (item[1], item[0]))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _candidate_rank_lookup(
    scores: dict[str, float],
    *,
    lower_is_better: bool,
) -> dict[str, int]:
    return {
        candidate_id: index
        for index, (candidate_id, _score) in enumerate(
            _candidate_metric_order(scores, lower_is_better=lower_is_better),
            start=1,
        )
    }


def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_denom <= 0.0 or y_denom <= 0.0:
        return None
    return numerator / (x_denom * y_denom)


def _spearman_rank_correlation(
    canonical_scores: dict[str, float],
    fast_scores: dict[str, float],
    *,
    lower_is_better: bool,
) -> float | None:
    common_ids = sorted(set(canonical_scores) & set(fast_scores))
    if len(common_ids) < 2:
        return None
    canonical_ranks = _candidate_rank_lookup(
        {candidate_id: canonical_scores[candidate_id] for candidate_id in common_ids},
        lower_is_better=lower_is_better,
    )
    fast_ranks = _candidate_rank_lookup(
        {candidate_id: fast_scores[candidate_id] for candidate_id in common_ids},
        lower_is_better=lower_is_better,
    )
    xs = [float(canonical_ranks[candidate_id]) for candidate_id in common_ids]
    ys = [float(fast_ranks[candidate_id]) for candidate_id in common_ids]
    return _pearson_correlation(xs, ys)


def _kendall_tau(
    canonical_scores: dict[str, float],
    fast_scores: dict[str, float],
    *,
    lower_is_better: bool,
) -> float | None:
    common_ids = sorted(set(canonical_scores) & set(fast_scores))
    if len(common_ids) < 2:
        return None
    canonical_ranks = _candidate_rank_lookup(
        {candidate_id: canonical_scores[candidate_id] for candidate_id in common_ids},
        lower_is_better=lower_is_better,
    )
    fast_ranks = _candidate_rank_lookup(
        {candidate_id: fast_scores[candidate_id] for candidate_id in common_ids},
        lower_is_better=lower_is_better,
    )
    concordant = 0
    discordant = 0
    for left_index, left_id in enumerate(common_ids[:-1]):
        for right_id in common_ids[left_index + 1 :]:
            canonical_sign = canonical_ranks[left_id] - canonical_ranks[right_id]
            fast_sign = fast_ranks[left_id] - fast_ranks[right_id]
            if canonical_sign == 0 or fast_sign == 0:
                continue
            if canonical_sign * fast_sign > 0:
                concordant += 1
            else:
                discordant += 1
    total_pairs = concordant + discordant
    if total_pairs == 0:
        return None
    return (concordant - discordant) / float(total_pairs)


def _top_k_recall(
    canonical_scores: dict[str, float],
    fast_scores: dict[str, float],
    *,
    lower_is_better: bool,
    k: int,
) -> float | None:
    common_ids = sorted(set(canonical_scores) & set(fast_scores))
    if not common_ids:
        return None
    canonical_top = {
        candidate_id
        for candidate_id, _score in _candidate_metric_order(
            {candidate_id: canonical_scores[candidate_id] for candidate_id in common_ids},
            lower_is_better=lower_is_better,
        )[:k]
    }
    fast_top = {
        candidate_id
        for candidate_id, _score in _candidate_metric_order(
            {candidate_id: fast_scores[candidate_id] for candidate_id in common_ids},
            lower_is_better=lower_is_better,
        )[:k]
    }
    if not canonical_top:
        return None
    return len(canonical_top & fast_top) / float(len(canonical_top))


def _metric_regret(
    *,
    selected_metric: float,
    best_metric: float,
    lower_is_better: bool,
) -> float:
    return (
        selected_metric - best_metric if lower_is_better else best_metric - selected_metric
    )


def _triage_detail_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if study.problem not in {"tsp", "zdt1"}:
        return []

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    group_meta: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        parsed = _triage_candidate_role(row)
        metric_value = _triage_metric_value(study.problem, row)
        if parsed is None or metric_value is None:
            continue
        candidate_id, role = parsed
        group_id = _triage_group_id(study.problem, row)
        grouped.setdefault((group_id, candidate_id, role), []).append(row)
        group_meta.setdefault(
            group_id,
            {
                "case_id": row.get("case_id") if study.problem == "tsp" else "",
                "case_group": _triage_case_group(study.problem, row),
                "scope": "case" if study.problem == "tsp" else "overall",
            },
        )

    grouped_details: dict[str, dict[str, dict[str, Any]]] = {}
    for (group_id, candidate_id, role), rows in grouped.items():
        detail = grouped_details.setdefault(group_id, {}).setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "case_id": group_meta[group_id]["case_id"],
                "case_group": group_meta[group_id]["case_group"],
                "scope": group_meta[group_id]["scope"],
            },
        )
        metric_stats = _aggregate_numeric(
            [_triage_metric_value(study.problem, row) for row in rows if _triage_metric_value(study.problem, row) is not None]
        )
        eval_stats = _aggregate_numeric(
            [_triage_total_actual_evaluations(row) for row in rows if _triage_total_actual_evaluations(row) is not None]
        )
        detail[f"{role}_metric_mean"] = metric_stats["mean"]
        detail[f"{role}_metric_std"] = metric_stats["std"]
        detail[f"{role}_actual_evaluations_used"] = sum(
            value for value in (_triage_total_actual_evaluations(row) for row in rows) if value is not None
        )
        detail[f"{role}_run_count"] = len(rows)
        if study.problem == "zdt1":
            pareto_stats = _aggregate_numeric(_numeric_values(rows, "pareto_ratio"))
            spread_stats = _aggregate_numeric(_numeric_values(rows, "spread"))
            detail[f"{role}_pareto_ratio_mean"] = pareto_stats["mean"]
            detail[f"{role}_spread_mean"] = spread_stats["mean"]

    lower_is_better = study.problem == "tsp"
    detail_rows: list[dict[str, Any]] = []
    for group_id, candidate_rows in sorted(grouped_details.items(), key=lambda item: item[0]):
        canonical_scores = {
            candidate_id: float(detail["canonical_metric_mean"])
            for candidate_id, detail in candidate_rows.items()
            if isinstance(detail.get("canonical_metric_mean"), int | float)
        }
        fast_scores = {
            candidate_id: float(detail["fast_metric_mean"])
            for candidate_id, detail in candidate_rows.items()
            if isinstance(detail.get("fast_metric_mean"), int | float)
        }
        canonical_ranks = _candidate_rank_lookup(
            canonical_scores,
            lower_is_better=lower_is_better,
        )
        fast_ranks = _candidate_rank_lookup(fast_scores, lower_is_better=lower_is_better)
        for candidate_id, detail in sorted(candidate_rows.items(), key=lambda item: item[0]):
            row = dict(detail)
            row["group_id"] = group_id
            row["canonical_rank"] = canonical_ranks.get(candidate_id)
            row["fast_rank"] = fast_ranks.get(candidate_id)
            row["candidate_count"] = len(set(canonical_scores) & set(fast_scores))
            detail_rows.append(row)
    return detail_rows


def _triage_ranking_fidelity_rows(
    study: LocalStudy,
    detail_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if study.problem not in {"tsp", "zdt1"} or not detail_rows:
        return []

    lower_is_better = study.problem == "tsp"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault(str(row.get("group_id", "__study__")), []).append(row)

    case_rows: list[dict[str, Any]] = []
    for group_id, rows in sorted(grouped.items(), key=lambda item: item[0]):
        canonical_scores = {
            str(row["candidate_id"]): float(row["canonical_metric_mean"])
            for row in rows
            if isinstance(row.get("canonical_metric_mean"), int | float)
        }
        fast_scores = {
            str(row["candidate_id"]): float(row["fast_metric_mean"])
            for row in rows
            if isinstance(row.get("fast_metric_mean"), int | float)
        }
        common_ids = sorted(set(canonical_scores) & set(fast_scores))
        if len(common_ids) < 2:
            continue
        canonical_order = _candidate_metric_order(
            {candidate_id: canonical_scores[candidate_id] for candidate_id in common_ids},
            lower_is_better=lower_is_better,
        )
        fast_order = _candidate_metric_order(
            {candidate_id: fast_scores[candidate_id] for candidate_id in common_ids},
            lower_is_better=lower_is_better,
        )
        canonical_best = canonical_order[0][0]
        fast_best = fast_order[0][0]
        fast_total_actual = sum(
            float(row["fast_actual_evaluations_used"])
            for row in rows
            if row.get("candidate_id") in common_ids
            and isinstance(row.get("fast_actual_evaluations_used"), int | float)
        )
        canonical_total_actual = sum(
            float(row["canonical_actual_evaluations_used"])
            for row in rows
            if row.get("candidate_id") in common_ids
            and isinstance(row.get("canonical_actual_evaluations_used"), int | float)
        )
        first = rows[0]
        case_rows.append(
            {
                "scope": first.get("scope", "overall"),
                "case_id": first.get("case_id", ""),
                "case_group": first.get("case_group", "overall"),
                "candidate_count": len(common_ids),
                "spearman_rank_correlation": _spearman_rank_correlation(
                    canonical_scores,
                    fast_scores,
                    lower_is_better=lower_is_better,
                ),
                "kendall_tau": _kendall_tau(
                    canonical_scores,
                    fast_scores,
                    lower_is_better=lower_is_better,
                ),
                "top_1_match_rate": 1.0 if canonical_best == fast_best else 0.0,
                "top_2_recall": _top_k_recall(
                    canonical_scores,
                    fast_scores,
                    lower_is_better=lower_is_better,
                    k=2,
                ),
                "top_3_recall": _top_k_recall(
                    canonical_scores,
                    fast_scores,
                    lower_is_better=lower_is_better,
                    k=3,
                ),
                "fast_top1_regret_vs_canonical_best": _metric_regret(
                    selected_metric=canonical_scores[fast_best],
                    best_metric=canonical_scores[canonical_best],
                    lower_is_better=lower_is_better,
                ),
                "fast_total_actual_evaluations_used": fast_total_actual,
                "canonical_total_actual_evaluations_used": canonical_total_actual,
                "fast_eval_ratio_to_canonical": (
                    fast_total_actual / canonical_total_actual
                    if canonical_total_actual > 0.0
                    else None
                ),
            }
        )

    if study.problem != "tsp":
        return case_rows

    rows: list[dict[str, Any]] = list(case_rows)
    aggregate_groups = {
        "overall": case_rows,
        **{
            case_group: [row for row in case_rows if row.get("case_group") == case_group]
            for case_group in sorted({str(row.get("case_group", "")) for row in case_rows})
        },
    }
    for case_group, members in aggregate_groups.items():
        if not members:
            continue
        row = {
            "scope": "overall" if case_group == "overall" else "case_group",
            "case_id": "",
            "case_group": case_group,
            "case_count": len(members),
        }
        for key in (
            "candidate_count",
            "spearman_rank_correlation",
            "kendall_tau",
            "top_1_match_rate",
            "top_2_recall",
            "top_3_recall",
            "fast_top1_regret_vs_canonical_best",
            "fast_total_actual_evaluations_used",
            "canonical_total_actual_evaluations_used",
            "fast_eval_ratio_to_canonical",
        ):
            stats = _aggregate_numeric(_numeric_values(members, key))
            row[key] = stats["mean"]
        rows.append(row)
    return rows


def _triage_workflow_rows(
    study: LocalStudy,
    detail_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if study.problem not in {"tsp", "zdt1"} or not detail_rows:
        return []

    lower_is_better = study.problem == "tsp"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault(str(row.get("group_id", "__study__")), []).append(row)

    case_rows: list[dict[str, Any]] = []
    for group_id, rows in sorted(grouped.items(), key=lambda item: item[0]):
        canonical_scores = {
            str(row["candidate_id"]): float(row["canonical_metric_mean"])
            for row in rows
            if isinstance(row.get("canonical_metric_mean"), int | float)
        }
        fast_scores = {
            str(row["candidate_id"]): float(row["fast_metric_mean"])
            for row in rows
            if isinstance(row.get("fast_metric_mean"), int | float)
        }
        common_ids = sorted(set(canonical_scores) & set(fast_scores))
        if len(common_ids) < 2:
            continue
        canonical_order = _candidate_metric_order(
            {candidate_id: canonical_scores[candidate_id] for candidate_id in common_ids},
            lower_is_better=lower_is_better,
        )
        fast_order = _candidate_metric_order(
            {candidate_id: fast_scores[candidate_id] for candidate_id in common_ids},
            lower_is_better=lower_is_better,
        )
        canonical_best_id = canonical_order[0][0]
        canonical_best_metric = canonical_scores[canonical_best_id]
        canonical_total_actual = sum(
            float(row["canonical_actual_evaluations_used"])
            for row in rows
            if row.get("candidate_id") in common_ids
            and isinstance(row.get("canonical_actual_evaluations_used"), int | float)
        )
        fast_total_actual = sum(
            float(row["fast_actual_evaluations_used"])
            for row in rows
            if row.get("candidate_id") in common_ids
            and isinstance(row.get("fast_actual_evaluations_used"), int | float)
        )
        canonical_eval_by_candidate = {
            str(row["candidate_id"]): float(row["canonical_actual_evaluations_used"])
            for row in rows
            if row.get("candidate_id") in common_ids
            and isinstance(row.get("canonical_actual_evaluations_used"), int | float)
        }
        pareto_ratio_by_candidate = {
            str(row["candidate_id"]): float(row["canonical_pareto_ratio_mean"])
            for row in rows
            if isinstance(row.get("canonical_pareto_ratio_mean"), int | float)
        }
        spread_by_candidate = {
            str(row["candidate_id"]): float(row["canonical_spread_mean"])
            for row in rows
            if isinstance(row.get("canonical_spread_mean"), int | float)
        }
        best_pareto_ratio = (
            max(pareto_ratio_by_candidate.values()) if pareto_ratio_by_candidate else None
        )
        best_spread = min(spread_by_candidate.values()) if spread_by_candidate else None
        first = rows[0]
        workflow_specs = [
            ("always_canonical_all_candidates", len(common_ids), [], 0.0),
            (
                "always_fast_pick_top1",
                1,
                [candidate_id for candidate_id, _score in fast_order[:1]],
                fast_total_actual,
            ),
            (
                "fast_screen_then_confirm_top2",
                2,
                [candidate_id for candidate_id, _score in fast_order[:2]],
                fast_total_actual,
            ),
            (
                "fast_screen_then_confirm_top3",
                3,
                [candidate_id for candidate_id, _score in fast_order[:3]],
                fast_total_actual,
            ),
        ]
        for workflow, selected_k, selected_ids, screening_cost in workflow_specs:
            if workflow == "always_canonical_all_candidates":
                final_candidate = canonical_best_id
                final_metric = canonical_best_metric
                confirm_cost = canonical_total_actual
                false_negative = 0.0
            else:
                selected_ids = [candidate_id for candidate_id in selected_ids if candidate_id in canonical_scores]
                if not selected_ids:
                    continue
                final_candidate = _candidate_metric_order(
                    {candidate_id: canonical_scores[candidate_id] for candidate_id in selected_ids},
                    lower_is_better=lower_is_better,
                )[0][0]
                final_metric = canonical_scores[final_candidate]
                confirm_cost = sum(
                    canonical_eval_by_candidate.get(candidate_id, 0.0)
                    for candidate_id in selected_ids
                )
                false_negative = 0.0 if canonical_best_id in selected_ids else 1.0
            total_actual = screening_cost if workflow == "always_fast_pick_top1" else (
                canonical_total_actual if workflow == "always_canonical_all_candidates" else screening_cost + confirm_cost
            )
            row = {
                "scope": first.get("scope", "overall"),
                "case_id": first.get("case_id", ""),
                "case_group": first.get("case_group", "overall"),
                "workflow": workflow,
                "selected_k": selected_k,
                "candidate_count": len(common_ids),
                "selected_candidate": final_candidate,
                "canonical_best_candidate": canonical_best_id,
                "final_selected_metric": final_metric,
                "canonical_best_metric": canonical_best_metric,
                "final_regret_vs_full_canonical": _metric_regret(
                    selected_metric=final_metric,
                    best_metric=canonical_best_metric,
                    lower_is_better=lower_is_better,
                ),
                "screening_actual_evaluations_used": screening_cost,
                "confirm_actual_evaluations_used": confirm_cost,
                "total_actual_evaluations_used": total_actual,
                "eval_savings_vs_canonical_all": (
                    1.0 - (total_actual / canonical_total_actual)
                    if canonical_total_actual > 0.0
                    else None
                ),
                "false_negative": false_negative,
                "oracle_hit_rate": 1.0 - false_negative,
            }
            if study.problem == "zdt1":
                selected_pareto = pareto_ratio_by_candidate.get(final_candidate)
                selected_spread = spread_by_candidate.get(final_candidate)
                row["selected_pareto_ratio"] = selected_pareto
                row["selected_spread"] = selected_spread
                row["pareto_ratio_safety_fail"] = (
                    1.0
                    if isinstance(selected_pareto, int | float)
                    and isinstance(best_pareto_ratio, int | float)
                    and float(selected_pareto) < max(0.95, float(best_pareto_ratio) - 0.02)
                    else 0.0
                )
                row["spread_safety_fail"] = (
                    1.0
                    if isinstance(selected_spread, int | float)
                    and isinstance(best_spread, int | float)
                    and float(selected_spread) > max(float(best_spread) * 1.1, float(best_spread) + 0.01)
                    else 0.0
                )
            case_rows.append(row)

    if study.problem != "tsp":
        return case_rows

    rows: list[dict[str, Any]] = list(case_rows)
    case_groups = sorted({str(row.get("case_group", "")) for row in case_rows})
    for workflow in sorted({str(row.get("workflow", "")) for row in case_rows}):
        workflow_members = [row for row in case_rows if row.get("workflow") == workflow]
        aggregate_sets = {
            "overall": workflow_members,
            **{
                case_group: [row for row in workflow_members if row.get("case_group") == case_group]
                for case_group in case_groups
            },
        }
        for case_group, members in aggregate_sets.items():
            if not members:
                continue
            aggregate_row = {
                "scope": "overall" if case_group == "overall" else "case_group",
                "case_id": "",
                "case_group": case_group,
                "workflow": workflow,
                "selected_k": members[0].get("selected_k"),
                "case_count": len(members),
            }
            for key in (
                "candidate_count",
                "final_regret_vs_full_canonical",
                "screening_actual_evaluations_used",
                "confirm_actual_evaluations_used",
                "total_actual_evaluations_used",
                "eval_savings_vs_canonical_all",
                "false_negative",
                "oracle_hit_rate",
                "pareto_ratio_safety_fail",
                "spread_safety_fail",
            ):
                stats = _aggregate_numeric(_numeric_values(members, key))
                aggregate_row[key] = stats["mean"]
            rows.append(aggregate_row)
    return rows


def _case_group_key(combo: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(key), json.dumps(value, ensure_ascii=False, sort_keys=True))
        for key, value in sorted(combo.items(), key=lambda item: item[0])
    )


def _tsp_case_group_summary_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if study.problem != "tsp":
        return []

    grouped: dict[tuple[str, tuple[tuple[str, str], ...]], list[dict[str, Any]]] = {}
    combo_lookup: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for row in raw_rows:
        combo = row.get("parameter_values")
        if not isinstance(combo, dict) or not combo:
            continue
        combo_key = _case_group_key(combo)
        combo_lookup[combo_key] = dict(combo)
        group_names = ["overall"]
        case_group = row.get("case_group")
        if isinstance(case_group, str) and case_group.strip():
            group_names.append(case_group.strip())
        for group_name in group_names:
            grouped.setdefault((group_name, combo_key), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (case_group, combo_key), rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        combo = combo_lookup[combo_key]
        label = f"case_group={case_group} | {_combo_label(combo)}"
        first = rows[0]
        distance_stats = _aggregate_numeric(_numeric_values(rows, "best_route_distance"))
        actual_eval_stats = _aggregate_numeric(_numeric_values(rows, "actual_evaluations_used"))
        extra_eval_stats = _aggregate_numeric(
            _numeric_values(rows, "extra_evaluations_from_adaptation")
        )
        runtime_stats = _aggregate_numeric(_numeric_values(rows, "runtime_seconds"))
        trigger_stats = _aggregate_numeric(_numeric_values(rows, "trigger_fire_count"))
        first_trigger_stats = _aggregate_numeric(_numeric_values(rows, "first_trigger_generation"))
        mode_switch_stats = _aggregate_numeric(_numeric_values(rows, "mode_switch_count"))
        first_switch_stats = _aggregate_numeric(_numeric_values(rows, "first_switch_generation"))
        decay_mode_stats = _aggregate_numeric(_numeric_values(rows, "time_in_decay_mode"))
        trigger_mode_stats = _aggregate_numeric(_numeric_values(rows, "time_in_trigger_mode"))
        post_trigger_stats = _aggregate_numeric(_numeric_values(rows, "post_trigger_improvement"))
        useless_trigger_stats = _aggregate_numeric(_numeric_values(rows, "useless_trigger_rate"))
        refresh_fraction_stats = _aggregate_numeric(
            _numeric_values(rows, "average_refresh_fraction_realized")
        )
        regret_tested = _numeric_values(rows, "regret_vs_oracle_tested_candidate")
        regret_preferred = _numeric_values(rows, "regret_vs_current_preferred_profile")
        regret_oracle = _numeric_values(rows, "regret_vs_oracle_fixed_policy")
        regret_decay = _numeric_values(rows, "regret_vs_always_decay_mutation")
        regret_trigger = _numeric_values(rows, "regret_vs_always_low_diversity_injection")
        regret_none = _numeric_values(rows, "regret_vs_always_none")
        regret_canonical = _numeric_values(rows, "regret_vs_canonical_profile")
        regret_fast = _numeric_values(rows, "regret_vs_fast_profile")
        regret_canonical_once = _numeric_values(rows, "regret_vs_canonical_once")
        regret_fast_once = _numeric_values(rows, "regret_vs_fast_once")
        gain_over_fast = _numeric_values(rows, "best_of_k_improvement_over_single_fast")
        regret_tested_stats = _aggregate_numeric(regret_tested)
        regret_preferred_stats = _aggregate_numeric(regret_preferred)
        regret_oracle_stats = _aggregate_numeric(regret_oracle)
        regret_decay_stats = _aggregate_numeric(regret_decay)
        regret_trigger_stats = _aggregate_numeric(regret_trigger)
        regret_none_stats = _aggregate_numeric(regret_none)
        regret_canonical_stats = _aggregate_numeric(regret_canonical)
        regret_fast_stats = _aggregate_numeric(regret_fast)
        regret_canonical_once_stats = _aggregate_numeric(regret_canonical_once)
        regret_fast_once_stats = _aggregate_numeric(regret_fast_once)
        gain_over_fast_stats = _aggregate_numeric(gain_over_fast)
        restart_count_stats = _aggregate_numeric(_numeric_values(rows, "portfolio_restart_count"))
        escalation_stats = _aggregate_numeric(_numeric_values(rows, "escalation_triggered"))
        false_escalation_stats = _aggregate_numeric(_numeric_values(rows, "false_escalation"))
        false_keep_stats = _aggregate_numeric(_numeric_values(rows, "false_keep"))
        pilot_eval_stats = _aggregate_numeric(
            _numeric_values(rows, "pilot_actual_evaluations_used")
        )
        escalation_eval_stats = _aggregate_numeric(
            _numeric_values(rows, "escalation_actual_evaluations_used")
        )
        pilot_fraction_stats = _aggregate_numeric(_numeric_values(rows, "pilot_budget_fraction"))
        summary_row: dict[str, Any] = {
            "study_name": study.study_name,
            "problem": study.problem,
            "case_group": case_group,
            "variant_label": label,
            "combo_label": _combo_label(combo),
            "run_count": len(rows),
            "case_count": len(
                {
                    str(row["case_id"])
                    for row in rows
                    if isinstance(row.get("case_id"), str) and str(row["case_id"]).strip()
                }
            ),
            "configured_budget": first.get("configured_budget"),
            "best_route_distance_mean": distance_stats["mean"],
            "best_route_distance_std": distance_stats["std"],
            "regret_vs_oracle_tested_candidate_mean": regret_tested_stats["mean"],
            "regret_vs_oracle_tested_candidate_std": regret_tested_stats["std"],
            "regret_vs_current_preferred_profile_mean": regret_preferred_stats["mean"],
            "regret_vs_current_preferred_profile_std": regret_preferred_stats["std"],
            "regret_vs_oracle_fixed_policy_mean": regret_oracle_stats["mean"],
            "regret_vs_oracle_fixed_policy_std": regret_oracle_stats["std"],
            "regret_vs_always_decay_mutation_mean": regret_decay_stats["mean"],
            "regret_vs_always_low_diversity_injection_mean": regret_trigger_stats["mean"],
            "regret_vs_always_none_mean": regret_none_stats["mean"],
            "regret_vs_canonical_profile_mean": regret_canonical_stats["mean"],
            "regret_vs_canonical_profile_std": regret_canonical_stats["std"],
            "regret_vs_fast_profile_mean": regret_fast_stats["mean"],
            "regret_vs_fast_profile_std": regret_fast_stats["std"],
            "regret_vs_canonical_once_mean": regret_canonical_once_stats["mean"],
            "regret_vs_fast_once_mean": regret_fast_once_stats["mean"],
            "best_of_k_improvement_over_single_fast_mean": gain_over_fast_stats["mean"],
            "portfolio_restart_count_mean": restart_count_stats["mean"],
            "win_rate_vs_decay_mutation": _win_rate(regret_decay),
            "win_rate_vs_low_diversity_injection": _win_rate(regret_trigger),
            "win_rate_vs_none": _win_rate(regret_none),
            "actual_evaluations_used_mean": actual_eval_stats["mean"],
            "extra_evaluations_from_adaptation_mean": extra_eval_stats["mean"],
            "runtime_seconds_mean": runtime_stats["mean"],
            "trigger_fire_count_mean": trigger_stats["mean"],
            "first_trigger_generation_mean": first_trigger_stats["mean"],
            "mode_switch_count_mean": mode_switch_stats["mean"],
            "first_switch_generation_mean": first_switch_stats["mean"],
            "time_in_decay_mode_mean": decay_mode_stats["mean"],
            "time_in_trigger_mode_mean": trigger_mode_stats["mean"],
            "post_trigger_improvement_mean": post_trigger_stats["mean"],
            "useless_trigger_rate_mean": useless_trigger_stats["mean"],
            "average_refresh_fraction_realized_mean": refresh_fraction_stats["mean"],
            "escalation_rate": escalation_stats["mean"],
            "false_escalation_rate": false_escalation_stats["mean"],
            "false_keep_rate": false_keep_stats["mean"],
            "pilot_actual_evaluations_used_mean": pilot_eval_stats["mean"],
            "escalation_actual_evaluations_used_mean": escalation_eval_stats["mean"],
            "pilot_budget_fraction_mean": pilot_fraction_stats["mean"],
            "anti_case_damage_mean": regret_decay_stats["mean"]
            if case_group == "anti_case"
            else None,
        }
        for key, value in _parameter_columns(combo).items():
            summary_row[key] = value
        for key, value in _parameter_columns(_summary_config_columns(first)).items():
            summary_row.setdefault(key, value)
        summary_rows.append(summary_row)

    return summary_rows


def _knapsack_case_group_summary_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if study.problem != "knapsack":
        return []

    grouped: dict[tuple[str, tuple[tuple[str, str], ...]], list[dict[str, Any]]] = {}
    combo_lookup: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for row in raw_rows:
        combo = row.get("parameter_values")
        if not isinstance(combo, dict) or not combo:
            continue
        combo_key = _case_group_key(combo)
        combo_lookup[combo_key] = dict(combo)
        group_names = ["overall"]
        case_group = row.get("case_group")
        if isinstance(case_group, str) and case_group.strip():
            group_names.append(case_group.strip())
        for group_name in group_names:
            grouped.setdefault((group_name, combo_key), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (case_group, combo_key), rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        combo = combo_lookup[combo_key]
        first = rows[0]
        regret_greedy_stats = _aggregate_numeric(
            _numeric_values(rows, "regret_vs_greedy_local_search")
        )
        regret_none_stats = _aggregate_numeric(_numeric_values(rows, "regret_vs_none"))
        regret_repair_stats = _aggregate_numeric(_numeric_values(rows, "regret_vs_repair_only"))
        regret_restart_stats = _aggregate_numeric(
            _numeric_values(rows, "regret_vs_current_restart_experimental")
        )
        regret_multi_none_stats = _aggregate_numeric(
            _numeric_values(rows, "regret_vs_multi_none_reference")
        )
        summary_row: dict[str, Any] = {
            "study_name": study.study_name,
            "problem": study.problem,
            "case_group": case_group,
            "variant_label": f"case_group={case_group} | {_combo_label(combo)}",
            "combo_label": _combo_label(combo),
            "run_count": len(rows),
            "case_count": len(
                {
                    str(row["case_id"])
                    for row in rows
                    if isinstance(row.get("case_id"), str) and str(row["case_id"]).strip()
                }
            ),
            "configured_budget": first.get("configured_budget"),
            "best_feasible_fitness_mean": _aggregate_numeric(
                _numeric_values(rows, "best_feasible_fitness")
            )["mean"],
            "feasible_rate": _aggregate_numeric(
                _numeric_values(rows, "final_feasible_ratio")
            )["mean"],
            "mean_violation_mean": _aggregate_numeric(
                _numeric_values(rows, "mean_violation")
            )["mean"],
            "regret_vs_greedy_local_search_mean": regret_greedy_stats["mean"],
            "regret_vs_none_mean": regret_none_stats["mean"],
            "regret_vs_repair_only_mean": regret_repair_stats["mean"],
            "regret_vs_current_restart_experimental_mean": regret_restart_stats["mean"],
            "regret_vs_multi_none_reference_mean": regret_multi_none_stats["mean"],
            "family_average_regret_mean": regret_greedy_stats["mean"],
            "family_gap_vs_greedy_mean": regret_greedy_stats["mean"],
            "family_gap_vs_repair_only_mean": regret_repair_stats["mean"],
            "family_win_rate_vs_none": _win_rate(_numeric_values(rows, "regret_vs_none")),
            "generations_to_first_feasible_mean": _aggregate_numeric(
                _numeric_values(rows, "generations_to_first_feasible")
            )["mean"],
            "init_to_final_gain_mean": _aggregate_numeric(
                _numeric_values(rows, "init_to_final_gain")
            )["mean"],
            "rerun_trigger_rate": _aggregate_numeric(_numeric_values(rows, "rerun_triggered"))["mean"],
            "runtime_seconds_mean": _aggregate_numeric(
                _numeric_values(rows, "runtime_seconds")
            )["mean"],
            "actual_evaluations_used_mean": _aggregate_numeric(
                _numeric_values(rows, "actual_evaluations_used")
            )["mean"],
            "extra_evaluations_from_adaptation_mean": _aggregate_numeric(
                _numeric_values(rows, "extra_evaluations_from_adaptation")
            )["mean"],
        }
        for key, value in _parameter_columns(combo).items():
            summary_row[key] = value
        for key, value in _parameter_columns(_summary_config_columns(first)).items():
            summary_row.setdefault(key, value)
        summary_rows.append(summary_row)
    return summary_rows


def _variant_summary_rows(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault(str(row["variant_label"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for label, rows in grouped.items():
        first = rows[0]
        actual_eval_stats = _aggregate_numeric(_numeric_values(rows, "actual_evaluations_used"))
        extra_eval_stats = _aggregate_numeric(
            _numeric_values(rows, "extra_evaluations_from_adaptation")
        )
        runtime_stats = _aggregate_numeric(_numeric_values(rows, "runtime_seconds"))
        early_stop_stats = _aggregate_numeric(_numeric_values(rows, "early_stop_triggered"))
        early_stop_generation_stats = _aggregate_numeric(
            _numeric_values(rows, "early_stop_generation")
        )
        rerun_stats = _aggregate_numeric(_numeric_values(rows, "rerun_triggered"))
        total_eval_stats = _aggregate_numeric(_numeric_values(rows, "total_actual_evaluations_used"))
        escalation_stats = _aggregate_numeric(_numeric_values(rows, "escalation_triggered"))
        pilot_eval_stats = _aggregate_numeric(
            _numeric_values(rows, "pilot_actual_evaluations_used")
        )
        escalation_eval_stats = _aggregate_numeric(
            _numeric_values(rows, "escalation_actual_evaluations_used")
        )
        pilot_fraction_stats = _aggregate_numeric(_numeric_values(rows, "pilot_budget_fraction"))
        regret_canonical_stats = _aggregate_numeric(
            _numeric_values(rows, "regret_vs_canonical_profile")
        )
        regret_fast_stats = _aggregate_numeric(_numeric_values(rows, "regret_vs_fast_profile"))
        regret_canonical_once_stats = _aggregate_numeric(
            _numeric_values(rows, "regret_vs_canonical_once")
        )
        regret_fast_once_stats = _aggregate_numeric(_numeric_values(rows, "regret_vs_fast_once"))
        false_escalation_stats = _aggregate_numeric(_numeric_values(rows, "false_escalation"))
        false_keep_stats = _aggregate_numeric(_numeric_values(rows, "false_keep"))
        restart_count_stats = _aggregate_numeric(_numeric_values(rows, "portfolio_restart_count"))
        gain_over_fast_stats = _aggregate_numeric(
            _numeric_values(rows, "best_of_k_improvement_over_single_fast")
        )
        merged_gain_over_fast_stats = _aggregate_numeric(
            _numeric_values(rows, "merged_archive_gain_over_single_fast")
        )
        primary_values = _numeric_values(rows, study.primary_metric)
        primary_stats = _aggregate_numeric(primary_values)
        primary_ci_low, primary_ci_high = _bootstrap_ci(primary_values)
        summary_row: dict[str, Any] = {
            "study_name": study.study_name,
            "problem": study.problem,
            "variant_label": label,
            "run_count": len(rows),
            "primary_metric": study.primary_metric,
            "primary_metric_mean": primary_stats["mean"],
            "primary_metric_median": primary_stats["median"],
            "primary_metric_std": primary_stats["std"],
            "primary_metric_ci_low": primary_ci_low,
            "primary_metric_ci_high": primary_ci_high,
            "configured_budget": first.get("configured_budget"),
            "actual_evaluations_used_mean": actual_eval_stats["mean"],
            "extra_evaluations_from_adaptation_mean": extra_eval_stats["mean"],
            "runtime_seconds_mean": runtime_stats["mean"],
            "early_stop_trigger_rate": early_stop_stats["mean"],
            "early_stop_generation_mean": early_stop_generation_stats["mean"],
            "rerun_trigger_rate": rerun_stats["mean"],
            "total_actual_evaluations_used_mean": total_eval_stats["mean"],
            "escalation_rate": escalation_stats["mean"],
            "pilot_actual_evaluations_used_mean": pilot_eval_stats["mean"],
            "escalation_actual_evaluations_used_mean": escalation_eval_stats["mean"],
            "pilot_budget_fraction_mean": pilot_fraction_stats["mean"],
            "regret_vs_canonical_profile_mean": regret_canonical_stats["mean"],
            "regret_vs_canonical_profile_std": regret_canonical_stats["std"],
            "regret_vs_fast_profile_mean": regret_fast_stats["mean"],
            "regret_vs_fast_profile_std": regret_fast_stats["std"],
            "regret_vs_canonical_once_mean": regret_canonical_once_stats["mean"],
            "regret_vs_canonical_once_std": regret_canonical_once_stats["std"],
            "regret_vs_fast_once_mean": regret_fast_once_stats["mean"],
            "regret_vs_fast_once_std": regret_fast_once_stats["std"],
            "false_escalation_rate": false_escalation_stats["mean"],
            "false_keep_rate": false_keep_stats["mean"],
            "portfolio_restart_count_mean": restart_count_stats["mean"],
            "best_of_k_improvement_over_single_fast_mean": gain_over_fast_stats["mean"],
            "merged_archive_gain_over_single_fast_mean": merged_gain_over_fast_stats["mean"],
        }
        trigger_stats = _aggregate_numeric(_numeric_values(rows, "trigger_fire_count"))
        first_trigger_stats = _aggregate_numeric(_numeric_values(rows, "first_trigger_generation"))
        post_trigger_stats = _aggregate_numeric(_numeric_values(rows, "post_trigger_improvement"))
        refresh_fraction_stats = _aggregate_numeric(
            _numeric_values(rows, "average_refresh_fraction_realized")
        )
        total_refresh_stats = _aggregate_numeric(
            _numeric_values(rows, "total_refresh_fraction_realized")
        )
        collapse_onset_stats = _aggregate_numeric(
            _numeric_values(rows, "collapse_onset_generation")
        )
        trigger_delay_stats = _aggregate_numeric(
            _numeric_values(rows, "trigger_delay_from_collapse")
        )
        time_to_improvement_stats = _aggregate_numeric(
            _numeric_values(rows, "time_to_first_nontrivial_improvement_after_trigger")
        )
        useless_trigger_stats = _aggregate_numeric(_numeric_values(rows, "useless_trigger_rate"))
        realized_refresh_stats = _aggregate_numeric(
            _numeric_values(rows, "realized_refresh_volume")
        )
        mode_switch_stats = _aggregate_numeric(_numeric_values(rows, "mode_switch_count"))
        first_switch_stats = _aggregate_numeric(_numeric_values(rows, "first_switch_generation"))
        decay_mode_stats = _aggregate_numeric(_numeric_values(rows, "time_in_decay_mode"))
        trigger_mode_stats = _aggregate_numeric(_numeric_values(rows, "time_in_trigger_mode"))
        switch_delay_stats = _aggregate_numeric(
            _numeric_values(rows, "switch_delay_from_collapse")
        )
        summary_row["trigger_fire_count_mean"] = trigger_stats["mean"]
        summary_row["trigger_fire_count_std"] = trigger_stats["std"]
        summary_row["first_trigger_generation_mean"] = first_trigger_stats["mean"]
        summary_row["first_trigger_generation_std"] = first_trigger_stats["std"]
        summary_row["mode_switch_count_mean"] = mode_switch_stats["mean"]
        summary_row["mode_switch_count_std"] = mode_switch_stats["std"]
        summary_row["first_switch_generation_mean"] = first_switch_stats["mean"]
        summary_row["first_switch_generation_std"] = first_switch_stats["std"]
        summary_row["time_in_decay_mode_mean"] = decay_mode_stats["mean"]
        summary_row["time_in_decay_mode_std"] = decay_mode_stats["std"]
        summary_row["time_in_trigger_mode_mean"] = trigger_mode_stats["mean"]
        summary_row["time_in_trigger_mode_std"] = trigger_mode_stats["std"]
        summary_row["post_trigger_improvement_mean"] = post_trigger_stats["mean"]
        summary_row["post_trigger_improvement_std"] = post_trigger_stats["std"]
        summary_row["time_to_first_nontrivial_improvement_after_trigger_mean"] = (
            time_to_improvement_stats["mean"]
        )
        summary_row["time_to_first_nontrivial_improvement_after_trigger_std"] = (
            time_to_improvement_stats["std"]
        )
        summary_row["collapse_onset_generation_mean"] = collapse_onset_stats["mean"]
        summary_row["collapse_onset_generation_std"] = collapse_onset_stats["std"]
        summary_row["trigger_delay_from_collapse_mean"] = trigger_delay_stats["mean"]
        summary_row["trigger_delay_from_collapse_std"] = trigger_delay_stats["std"]
        summary_row["switch_delay_from_collapse_mean"] = switch_delay_stats["mean"]
        summary_row["switch_delay_from_collapse_std"] = switch_delay_stats["std"]
        summary_row["useless_trigger_rate_mean"] = useless_trigger_stats["mean"]
        summary_row["useless_trigger_rate_std"] = useless_trigger_stats["std"]
        summary_row["average_refresh_fraction_realized_mean"] = refresh_fraction_stats["mean"]
        summary_row["average_refresh_fraction_realized_std"] = refresh_fraction_stats["std"]
        summary_row["total_refresh_fraction_realized_mean"] = total_refresh_stats["mean"]
        summary_row["total_refresh_fraction_realized_std"] = total_refresh_stats["std"]
        summary_row["realized_refresh_volume_mean"] = realized_refresh_stats["mean"]
        summary_row["realized_refresh_volume_std"] = realized_refresh_stats["std"]
        last_improvement_stats = _aggregate_numeric(
            _numeric_values(rows, "generations_to_last_improvement")
        )
        hv_plateau_stats = _aggregate_numeric(_numeric_values(rows, "hv_plateau_generation"))
        summary_row["generations_to_last_improvement_mean"] = last_improvement_stats["mean"]
        summary_row["generations_to_last_improvement_std"] = last_improvement_stats["std"]
        summary_row["hv_plateau_generation_mean"] = hv_plateau_stats["mean"]
        summary_row["hv_plateau_generation_std"] = hv_plateau_stats["std"]
        if first.get("case_id") is not None:
            summary_row["case_id"] = first.get("case_id")
        if first.get("case_note"):
            summary_row["case_note"] = first.get("case_note")
        if first.get("case_group"):
            summary_row["case_group"] = first.get("case_group")
        if first.get("portfolio_mode"):
            summary_row["portfolio_mode"] = first.get("portfolio_mode")
        if first.get("portfolio_profile"):
            summary_row["portfolio_profile"] = first.get("portfolio_profile")
        for key in _STUDY_METADATA_KEYS:
            if first.get(key) is not None:
                summary_row[key] = first.get(key)
        for key, value in _parameter_columns(first["parameter_values"]).items():
            summary_row[key] = value
        for key, value in _parameter_columns(_summary_config_columns(first)).items():
            summary_row.setdefault(key, value)
        final_numeric_keys = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if key.startswith("final_")
                and isinstance(value, int | float)
                and not isinstance(value, bool)
            }
        )
        for key in final_numeric_keys:
            stats = _aggregate_numeric(_numeric_values(rows, key))
            summary_row[f"{key}_mean"] = stats["mean"]
            summary_row[f"{key}_std"] = stats["std"]

        if study.problem == "onemax":
            best_stats = _aggregate_numeric(
                [float(row["best_fitness"]) for row in rows if row.get("best_fitness") is not None]
            )
            gen_stats = _aggregate_numeric(
                [
                    float(row["generations_to_target"])
                    for row in rows
                    if row.get("generations_to_target") is not None
                ]
            )
            eval_target_stats = _aggregate_numeric(
                [
                    float(row["evaluations_to_target"])
                    for row in rows
                    if row.get("evaluations_to_target") is not None
                ]
            )
            summary_row.update(
                {
                    "best_fitness_mean": best_stats["mean"],
                    "best_fitness_median": best_stats["median"],
                    "best_fitness_std": best_stats["std"],
                    "target_hit_rate": mean(float(bool(row["target_hit"])) for row in rows),
                    "evaluations_to_target_mean": eval_target_stats["mean"],
                    "evaluations_to_target_std": eval_target_stats["std"],
                    "generations_to_target_mean": gen_stats["mean"],
                    "generations_to_target_std": gen_stats["std"],
                }
            )
        elif study.problem == "knapsack":
            feasible_stats = [float(bool(row["feasible"])) for row in rows]
            feasible_value_stats = _aggregate_numeric(
                [
                    float(row["best_feasible_fitness"])
                    for row in rows
                    if row.get("best_feasible_fitness") is not None
                ]
            )
            violation_stats = _aggregate_numeric(
                [
                    float(row["mean_violation"])
                    for row in rows
                    if row.get("mean_violation") is not None
                ]
            )
            summary_row.update(
                {
                    "best_feasible_fitness_mean": feasible_value_stats["mean"],
                    "best_feasible_fitness_std": feasible_value_stats["std"],
                    "feasible_rate": mean(feasible_stats),
                    "mean_violation_mean": violation_stats["mean"],
                    "mean_violation_std": violation_stats["std"],
                    "regret_vs_greedy_local_search_mean": _aggregate_numeric(
                        _numeric_values(rows, "regret_vs_greedy_local_search")
                    )["mean"],
                    "regret_vs_none_mean": _aggregate_numeric(
                        _numeric_values(rows, "regret_vs_none")
                    )["mean"],
                    "regret_vs_repair_only_mean": _aggregate_numeric(
                        _numeric_values(rows, "regret_vs_repair_only")
                    )["mean"],
                    "regret_vs_current_restart_experimental_mean": _aggregate_numeric(
                        _numeric_values(rows, "regret_vs_current_restart_experimental")
                    )["mean"],
                    "regret_vs_multi_none_reference_mean": _aggregate_numeric(
                        _numeric_values(rows, "regret_vs_multi_none_reference")
                    )["mean"],
                    "regret_vs_repair_only_reference_mean": _aggregate_numeric(
                        _numeric_values(rows, "regret_vs_repair_only_reference")
                    )["mean"],
                    "initial_feasible_fraction_mean": _aggregate_numeric(
                        _numeric_values(rows, "initial_feasible_fraction")
                    )["mean"],
                    "generations_to_first_feasible_mean": _aggregate_numeric(
                        _numeric_values(rows, "generations_to_first_feasible")
                    )["mean"],
                    "init_to_final_gain_mean": _aggregate_numeric(
                        _numeric_values(rows, "init_to_final_gain")
                    )["mean"],
                    "pilot_initial_feasible_fraction_mean": _aggregate_numeric(
                        _numeric_values(rows, "pilot_initial_feasible_fraction")
                    )["mean"],
                    "pilot_generations_to_first_feasible_mean": _aggregate_numeric(
                        _numeric_values(rows, "pilot_generations_to_first_feasible")
                    )["mean"],
                }
            )
        elif study.problem == "tsp":
            distance_stats = _aggregate_numeric(
                [
                    float(row["best_route_distance"])
                    for row in rows
                    if row.get("best_route_distance") is not None
                ]
            )
            fitness_stats = _aggregate_numeric(
                [float(row["best_fitness"]) for row in rows if row.get("best_fitness") is not None]
            )
            seeded_count_stats = _aggregate_numeric(
                [
                    float(row["hybrid_seeded_individuals"])
                    for row in rows
                    if row.get("hybrid_seeded_individuals") is not None
                ]
            )
            realized_seed_fraction_stats = _aggregate_numeric(
                [
                    float(row["seeded_population_fraction_realized"])
                    for row in rows
                    if row.get("seeded_population_fraction_realized") is not None
                ]
            )
            initial_best_stats = _aggregate_numeric(
                _numeric_values(rows, "initial_best_route_distance")
            )
            initial_mean_route_stats = _aggregate_numeric(
                _numeric_values(rows, "initial_mean_route_distance")
            )
            initial_diversity_stats = _aggregate_numeric(
                _numeric_values(rows, "initial_population_diversity")
            )
            init_to_final_gain_stats = _aggregate_numeric(
                _numeric_values(rows, "init_to_final_gain")
            )
            first_improvement_stats = _aggregate_numeric(
                _numeric_values(rows, "generations_to_first_improvement")
            )
            summary_row.update(
                {
                    "best_route_distance_mean": distance_stats["mean"],
                    "best_route_distance_median": distance_stats["median"],
                    "best_route_distance_std": distance_stats["std"],
                    "best_fitness_mean": fitness_stats["mean"],
                    "initial_best_route_distance_mean": initial_best_stats["mean"],
                    "initial_best_route_distance_std": initial_best_stats["std"],
                    "initial_mean_route_distance_mean": initial_mean_route_stats["mean"],
                    "initial_mean_route_distance_std": initial_mean_route_stats["std"],
                    "initial_population_diversity_mean": initial_diversity_stats["mean"],
                    "initial_population_diversity_std": initial_diversity_stats["std"],
                    "init_to_final_gain_mean": init_to_final_gain_stats["mean"],
                    "init_to_final_gain_std": init_to_final_gain_stats["std"],
                    "generations_to_first_improvement_mean": first_improvement_stats["mean"],
                    "generations_to_first_improvement_std": first_improvement_stats["std"],
                    "hybrid_seeded_individuals_mean": seeded_count_stats["mean"],
                    "seeded_population_fraction_realized_mean": (
                        realized_seed_fraction_stats["mean"]
                    ),
                }
            )
        else:
            hv_stats = _aggregate_numeric(
                [float(row["hypervolume"]) for row in rows if row.get("hypervolume") is not None]
            )
            pareto_ratio_stats = _aggregate_numeric(
                [float(row["pareto_ratio"]) for row in rows if row.get("pareto_ratio") is not None]
            )
            spread_stats = _aggregate_numeric(
                [float(row["spread"]) for row in rows if row.get("spread") is not None]
            )
            front_size_stats = _aggregate_numeric(
                [
                    float(row["pareto_front_size"])
                    for row in rows
                    if row.get("pareto_front_size") is not None
                ]
            )
            summary_row.update(
                {
                    "hypervolume_mean": hv_stats["mean"],
                    "hypervolume_std": hv_stats["std"],
                    "merged_archive_hv_mean": _aggregate_numeric(
                        _numeric_values(rows, "merged_archive_hv")
                    )["mean"],
                    "pareto_ratio_mean": pareto_ratio_stats["mean"],
                    "spread_mean": spread_stats["mean"],
                    "front_size_mean": front_size_stats["mean"],
                    "pareto_front_size_mean": front_size_stats["mean"],
                }
            )
        summary_rows.append(summary_row)
    return sorted(summary_rows, key=lambda row: str(row["variant_label"]))


def _best_variant_row(
    study: LocalStudy,
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not summary_rows:
        return None
    metric_column = f"{study.primary_metric}_mean"
    candidates = [row for row in summary_rows if isinstance(row.get(metric_column), int | float)]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: float(row[metric_column]),
        reverse=not _metric_is_lower_better(study.primary_metric),
    )[0]


def _history_summary_rows(
    study: LocalStudy,
    history_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    history_metric = _problem_history_metric(study.problem, study.plotting)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    param_lookup: dict[str, dict[str, Any]] = {}
    for payload in history_payloads:
        label = str(payload["variant_label"])
        param_lookup[label] = dict(payload["parameter_values"])
        for row in payload["history"]:
            generation = row.get("generation")
            if not isinstance(generation, int):
                continue
            grouped.setdefault((label, generation), []).append(dict(row))

    rows: list[dict[str, Any]] = []
    for (label, generation), values in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        metric_values = [
            value
            for value in (
                _coerce_metric_value(study.problem, history_metric, item) for item in values
            )
            if value is not None
        ]
        metric_stats = _aggregate_numeric(metric_values)
        row: dict[str, Any] = {
            "variant_label": label,
            "generation": generation,
            "history_metric": history_metric,
            "mean_metric": metric_stats["mean"],
            "std_metric": metric_stats["std"],
            "runs": len(values),
        }
        numeric_keys = sorted(
            {
                key
                for item in values
                for key, value in item.items()
                if key != "generation"
                and isinstance(value, int | float)
                and not isinstance(value, bool)
            }
        )
        for key in numeric_keys:
            key_values = [
                float(item[key])
                for item in values
                if isinstance(item.get(key), int | float)
            ]
            stats = _aggregate_numeric(key_values)
            row[key] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
        row.update(_parameter_columns(param_lookup[label]))
        rows.append(row)
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def _escape_cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(_escape_cell(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _render_stress_case_catalog_markdown(
    study: LocalStudy,
    rows: list[dict[str, Any]],
) -> str:
    lines = [f"# Stress Case Catalog: {study.study_name}", ""]
    lines.append(
        "Current defaults stay frozen in this pass. The rows below are the concrete local cases "
        "that most often drove tail loss, ambiguity, or safety failures."
    )
    lines.append("")
    headers = [
        "problem",
        "case_id",
        "seed",
        "budget_band",
        "profile_compared",
        "regret_or_loss",
        "why_selected_as_stress_case",
        "case_group",
    ]
    table_rows = [[_format_number(row.get(header)) for header in headers] for row in rows]
    lines.append(_markdown_table(headers, table_rows))
    lines.append("")
    return "\n".join(lines)


def _render_stress_suite_notes(
    study: LocalStudy,
    catalog_rows: list[dict[str, Any]],
    tail_rows: list[dict[str, Any]],
) -> str:
    lines = [f"# Stress Suite Notes: {study.study_name}", ""]
    config = _stress_suite_config(study) or {}
    protocol_note = str(config.get("protocol_note") or "").strip()
    if protocol_note:
        lines.append(f"- Protocol carry-over: {protocol_note}")
    lines.append("- Defaults stay unchanged in this pass; the goal is to lock future optimization targets.")
    if study.problem == "tsp":
        anti_rows = [row for row in tail_rows if str(row.get("case_group")) == "anti_case"]
        rescue_rows = [row for row in tail_rows if str(row.get("case_group")) == "rescue_target"]
        if anti_rows:
            anti = anti_rows[0]
            lines.append(
                "- TSP anti-case tail remains the main caution zone: "
                f"p90 `{_format_number(anti.get('p90_loss_pct'))}` / "
                f"p95 `{_format_number(anti.get('p95_loss_pct'))}` / "
                f"max `{_format_number(anti.get('max_loss_pct'))}`."
            )
        if rescue_rows:
            rescue = rescue_rows[0]
            lines.append(
                "- Rescue-target rows still matter, but read them separately from corridor-like anti-cases: "
                f"mean `{_format_number(rescue.get('mean_loss_pct'))}`, "
                f"decision flip `{_format_number(rescue.get('decision_flip_rate'))}`."
            )
        lines.append(
            "- Future TSP work should start from the cataloged anti-case tails and rescue-target ambiguity rows, "
            "not from a broad new profile search."
        )
    elif study.problem == "zdt1":
        safety_rows = [
            row
            for row in tail_rows
            if str(row.get("case_group")) in {"spread_safety_fail", "pareto_safety_fail", "joint_safety_fail"}
        ]
        if safety_rows:
            worst = max(
                safety_rows,
                key=lambda row: _stress_numeric(row.get("max_loss_pct")) or -1e18,
            )
            lines.append(
                "- ZDT1 worst stress rows are safety-tagged rather than pure mean-HV rows: "
                f"`{worst.get('case_group')}` max HV loss `{_format_number(worst.get('max_loss_pct'))}`, "
                f"spread fail rate `{_format_number(worst.get('spread_fail_rate'))}`."
            )
        lines.append(
            "- Keep `F 3 exploratory / Q 8-10 final safety` as-is; future work should target the cataloged spread and Pareto failures."
        )
    elif study.problem == "knapsack":
        lines.append(
            "- Knapsack stays parked broadly. Only the narrow repair-only note survives, and the stress suite now pins down which family rows are still borderline."
        )
    else:
        lines.append("- OneMax remains a control problem; the stress check is only there to keep the instrumentation honest.")
    if catalog_rows:
        lines.append(
            f"- Stress catalog rows captured: `{len(catalog_rows)}`. Reuse these rows first before inventing new local study families."
        )
    lines.append("")
    return "\n".join(lines)


def _render_study_markdown(
    study: LocalStudy,
    summary_rows: list[dict[str, Any]],
    best_variant: dict[str, Any] | None,
    case_group_summary_rows: list[dict[str, Any]] | None = None,
    ranking_fidelity_rows: list[dict[str, Any]] | None = None,
    triage_workflow_rows: list[dict[str, Any]] | None = None,
    tolerance_rows: list[dict[str, Any]] | None = None,
    seed_budget_rows: list[dict[str, Any]] | None = None,
    sequential_decision_rows: list[dict[str, Any]] | None = None,
    tsp_fast_tail_summary_rows: list[dict[str, Any]] | None = None,
    zdt1_spread_validation_rows: list[dict[str, Any]] | None = None,
    stress_case_catalog_rows: list[dict[str, Any]] | None = None,
    tail_risk_summary_rows: list[dict[str, Any]] | None = None,
    failure_trace_rows: list[dict[str, Any]] | None = None,
    failure_hypothesis_rows: list[dict[str, Any]] | None = None,
) -> str:
    lines = [f"# Local Study: {study.study_name}", ""]
    if study.description:
        lines.append(study.description)
        lines.append("")
    lines.append(f"- Problem: `{study.problem}`")
    if study.base_preset:
        lines.append(f"- Base preset: `{study.base_preset}`")
    if study.base_config:
        lines.append(f"- Base config: `{study.base_config}`")
    lines.append(f"- Seeds: `{', '.join(str(seed) for seed in study.seeds)}`")
    lines.append(f"- Primary metric: `{study.primary_metric}`")
    if study.budget_ceiling is not None:
        lines.append(f"- Budget ceiling: `{study.budget_ceiling}` evaluations")
    lines.append(
        "- Sweep axes: "
        + ", ".join(f"`{key}` ({len(values)} values)" for key, values in study.sweep.items())
    )
    if study.cases:
        lines.append(
            "- Hard cases: "
            + ", ".join(
                f"`{case.case_id}`"
                + (f" ({case.note})" if case.note else "")
                for case in study.cases
            )
        )
    if study.runtime_budget_note:
        lines.append(f"- Runtime note: {study.runtime_budget_note}")
    lines.append("")

    if best_variant is not None:
        lines.append("## Best Variant")
        lines.append("")
        lines.append(f"- Label: `{best_variant['variant_label']}`")
        metric_column = f"{study.primary_metric}_mean"
        lines.append(f"- {metric_column}: `{_format_number(best_variant.get(metric_column))}`")
        lines.append(
            "- configured_budget: "
            f"`{_format_number(best_variant.get('configured_budget'))}`"
        )
        lines.append(
            "- actual_evaluations_used_mean: "
            f"`{_format_number(best_variant.get('actual_evaluations_used_mean'))}`"
        )
        lines.append(
            "- extra_evaluations_from_adaptation_mean: "
            f"`{_format_number(best_variant.get('extra_evaluations_from_adaptation_mean'))}`"
        )
        lines.append(
            "- trigger_fire_count_mean: "
            f"`{_format_number(best_variant.get('trigger_fire_count_mean'))}`"
        )
        lines.append(
            "- first_trigger_generation_mean: "
            f"`{_format_number(best_variant.get('first_trigger_generation_mean'))}`"
        )
        if best_variant.get("post_trigger_improvement_mean") is not None:
            lines.append(
                "- post_trigger_improvement_mean: "
                f"`{_format_number(best_variant.get('post_trigger_improvement_mean'))}`"
            )
        if best_variant.get("mode_switch_count_mean") is not None:
            lines.append(
                "- mode_switch_count_mean: "
                f"`{_format_number(best_variant.get('mode_switch_count_mean'))}`"
            )
        if best_variant.get("first_switch_generation_mean") is not None:
            lines.append(
                "- first_switch_generation_mean: "
                f"`{_format_number(best_variant.get('first_switch_generation_mean'))}`"
            )
        if best_variant.get("regret_vs_oracle_fixed_policy_mean") is not None:
            lines.append(
                "- regret_vs_oracle_fixed_policy_mean: "
                f"`{_format_number(best_variant.get('regret_vs_oracle_fixed_policy_mean'))}`"
            )
        if best_variant.get("collapse_onset_generation_mean") is not None:
            lines.append(
                "- collapse_onset_generation_mean: "
                f"`{_format_number(best_variant.get('collapse_onset_generation_mean'))}`"
            )
        if best_variant.get("trigger_delay_from_collapse_mean") is not None:
            lines.append(
                "- trigger_delay_from_collapse_mean: "
                f"`{_format_number(best_variant.get('trigger_delay_from_collapse_mean'))}`"
            )
        if best_variant.get("useless_trigger_rate_mean") is not None:
            lines.append(
                "- useless_trigger_rate_mean: "
                f"`{_format_number(best_variant.get('useless_trigger_rate_mean'))}`"
            )
        if best_variant.get("average_refresh_fraction_realized_mean") is not None:
            lines.append(
                "- average_refresh_fraction_realized_mean: "
                f"`{_format_number(best_variant.get('average_refresh_fraction_realized_mean'))}`"
            )
        if best_variant.get("realized_refresh_volume_mean") is not None:
            lines.append(
                "- realized_refresh_volume_mean: "
                f"`{_format_number(best_variant.get('realized_refresh_volume_mean'))}`"
            )
        if best_variant.get("regret_vs_current_profile_mean") is not None:
            lines.append(
                "- regret_vs_current_profile_mean: "
                f"`{_format_number(best_variant.get('regret_vs_current_profile_mean'))}`"
            )
        if best_variant.get("regret_vs_none_mean") is not None:
            lines.append(
                "- regret_vs_none_mean: "
                f"`{_format_number(best_variant.get('regret_vs_none_mean'))}`"
            )
        if best_variant.get("extra_evaluations_from_adaptation_mean") == 0:
            lines.append(
                "- Adaptation note: `0` extra objective evaluations; refresh/injection "
                "reused the normal next-generation budget."
            )
        ci_low = _format_number(best_variant.get("primary_metric_ci_low"))
        ci_high = _format_number(best_variant.get("primary_metric_ci_high"))
        lines.append(
            "- primary_metric_ci_95: "
            f"`[{ci_low}, {ci_high}]`"
        )
        lines.append("")

    lines.append("## Variant Summary")
    lines.append("")
    headers = ["variant_label"]
    if any(row.get("case_id") for row in summary_rows):
        headers.extend(["case_id", "case_note"])
    headers.extend([*study.sweep.keys(), *_summary_metric_columns(study.problem)])
    table_rows = []
    for row in summary_rows:
        table_rows.append([_format_number(row.get(header)) for header in headers])
    lines.append(_markdown_table(headers, table_rows))
    lines.append("")

    if case_group_summary_rows:
        lines.append("## Case Group Summary")
        lines.append("")
        group_headers = [
            "case_group",
            "combo_label",
            "case_count",
            "run_count",
            "configured_budget",
            "best_route_distance_mean",
            "regret_vs_oracle_fixed_policy_mean",
            "regret_vs_always_decay_mutation_mean",
            "win_rate_vs_decay_mutation",
            "win_rate_vs_low_diversity_injection",
            "win_rate_vs_none",
            "anti_case_damage_mean",
            "mode_switch_count_mean",
            "trigger_fire_count_mean",
            "runtime_seconds_mean",
        ]
        group_rows = [
            [_format_number(row.get(header)) for header in group_headers]
            for row in case_group_summary_rows
        ]
        lines.append(_markdown_table(group_headers, group_rows))
        lines.append("")

    if ranking_fidelity_rows:
        lines.append("## Ranking Fidelity")
        lines.append("")
        display_rows = [
            row
            for row in ranking_fidelity_rows
            if str(row.get("scope")) != "case"
        ] or ranking_fidelity_rows
        fidelity_headers = [
            "scope",
            "case_group",
            "case_count",
            "candidate_count",
            "spearman_rank_correlation",
            "kendall_tau",
            "top_1_match_rate",
            "top_2_recall",
            "top_3_recall",
            "fast_top1_regret_vs_canonical_best",
            "fast_eval_ratio_to_canonical",
        ]
        fidelity_rows = [
            [_format_number(row.get(header)) for header in fidelity_headers]
            for row in display_rows
        ]
        lines.append(_markdown_table(fidelity_headers, fidelity_rows))
        lines.append("")

    if triage_workflow_rows:
        lines.append("## Triage Workflow")
        lines.append("")
        display_rows = [
            row
            for row in triage_workflow_rows
            if str(row.get("scope")) != "case"
        ] or triage_workflow_rows
        workflow_headers = [
            "scope",
            "case_group",
            "workflow",
            "selected_k",
            "case_count",
            "final_regret_vs_full_canonical",
            "total_actual_evaluations_used",
            "eval_savings_vs_canonical_all",
            "oracle_hit_rate",
            "pareto_ratio_safety_fail",
            "spread_safety_fail",
        ]
        workflow_rows = [
            [_format_number(row.get(header)) for header in workflow_headers]
            for row in display_rows
        ]
        lines.append(_markdown_table(workflow_headers, workflow_rows))
        lines.append("")

    if tolerance_rows:
        lines.append("## Q/F Tolerance")
        lines.append("")
        tolerance_headers = [
            "scope",
            "case_group",
            "tolerance_bin_pct",
            "acceptable_rate",
            "mean_loss_pct",
            "median_loss_pct",
            "p90_loss_pct",
            "max_loss_pct",
            "actual_eval_savings_pct_mean",
            "runtime_savings_pct_mean",
        ]
        if study.problem == "tsp":
            tolerance_headers.extend(
                ["fast_better_rate", "tie_rate", "quality_better_rate"]
            )
        else:
            tolerance_headers.extend(
                [
                    "hv_only_accept_rate",
                    "pareto_ratio_delta_mean",
                    "spread_delta_mean",
                    "pareto_ratio_fail_rate",
                    "spread_fail_rate",
                    "joint_safety_fail_rate",
                ]
            )
        tolerance_display_rows = [
            [_format_number(row.get(header)) for header in tolerance_headers]
            for row in tolerance_rows
        ]
        lines.append(_markdown_table(tolerance_headers, tolerance_display_rows))
        lines.append("")

    if seed_budget_rows:
        lines.append("## Seed Budget Calibration")
        lines.append("")
        display_rows = [
            row
            for row in seed_budget_rows
            if str(row.get("scope")) != "case"
        ] or seed_budget_rows
        seed_headers = [
            "scope",
            "case_group",
            "seed_count",
            "sample_count",
            "decision",
            "decision_flip_rate_vs_full",
            "decision_stability_score",
            "ci_width_pct",
            "mean_loss_pct",
            "p90_loss_pct",
            "repair_gain_vs_none_mean",
            "repair_gap_vs_greedy_mean",
            "control_delta_vs_reference_mean",
            "actual_eval_savings_pct_mean",
            "runtime_savings_pct_mean",
        ]
        seed_rows = [
            [_format_number(row.get(header)) for header in seed_headers]
            for row in display_rows
        ]
        lines.append(_markdown_table(seed_headers, seed_rows))
        lines.append("")

    if sequential_decision_rows:
        lines.append("## Sequential Compare")
        lines.append("")
        display_rows = [
            row
            for row in sequential_decision_rows
            if str(row.get("scope")) != "case"
        ] or sequential_decision_rows
        sequential_headers = [
            "mode",
            "scope",
            "case_group",
            "seed_count",
            "sample_count",
            "decision_label",
            "stage_action",
            "decision_flip_rate_vs_full",
            "paired_delta_mean",
            "paired_delta_median",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "paired_win_count",
            "paired_loss_count",
            "actual_evaluations_used",
            "actual_eval_savings_vs_full_pct",
        ]
        sequential_rows = [
            [_format_number(row.get(header)) for header in sequential_headers]
            for row in display_rows
        ]
        lines.append(_markdown_table(sequential_headers, sequential_rows))
        lines.append("")

    if failure_trace_rows:
        lines.append("## Failure Trace")
        lines.append("")
        trace_headers = [
            "target_id",
            "case_group",
            "study_variant",
            "seed",
            "mechanism_hint",
            "route_distance_loss_pct_vs_quality" if study.problem == "tsp" else "hv_loss_pct_vs_quality",
            "collapse_onset_generation" if study.problem == "tsp" else "safety_fail_onset_generation",
            "last_improvement_generation" if study.problem == "tsp" else "hv_plateau_generation",
        ]
        trace_display_rows = [
            row
            for row in failure_trace_rows[: min(12, len(failure_trace_rows))]
        ]
        lines.append(
            _markdown_table(
                trace_headers,
                [
                    [_format_number(row.get(header)) for header in trace_headers]
                    for row in trace_display_rows
                ],
            )
        )
        lines.append("")

    if failure_hypothesis_rows:
        lines.append("## Failure Hypotheses")
        lines.append("")
        hypothesis_headers = [
            "hypothesis_id",
            "target_id",
            "current_evidence_strength",
            "confirmed_or_weakened",
            "recommended_next_action",
            "supporting_trace_signals",
        ]
        lines.append(
            _markdown_table(
                hypothesis_headers,
                [
                    [_format_number(row.get(header)) for header in hypothesis_headers]
                    for row in failure_hypothesis_rows
                ],
            )
        )
        lines.append("")

    if tsp_fast_tail_summary_rows:
        lines.append("## Fast Tail Hardening")
        lines.append("")
        display_rows = [
            row for row in tsp_fast_tail_summary_rows if str(row.get("scope")) == "overall"
        ] or tsp_fast_tail_summary_rows
        tail_headers = [
            "scope",
            "case_group",
            "study_variant",
            "mean_loss_pct",
            "median_loss_pct",
            "p75_loss_pct",
            "p90_loss_pct",
            "p95_loss_pct",
            "max_loss_pct",
            "rescue_target_mean_loss_pct",
            "anti_case_mean_loss_pct",
            "anti_case_p90_loss_pct",
            "anti_case_p95_loss_pct",
            "anti_case_max_loss_pct",
            "tail_hardening_score",
            "tail_hardening_score_p95",
            "anti_case_p95_reduction_score",
            "anti_case_max_reduction_score",
            "rescue_preservation_score",
            "mean_regret_vs_q",
            "mean_regret_vs_current_fast",
            "configured_budget_mean",
            "actual_evaluations_used_mean",
            "runtime_seconds_mean",
        ]
        tail_display_rows = [
            [_format_number(row.get(header)) for header in tail_headers]
            for row in display_rows
        ]
        lines.append(_markdown_table(tail_headers, tail_display_rows))
        lines.append("")

    if zdt1_spread_validation_rows:
        lines.append("## ZDT1 Spread Candidate Validation")
        lines.append("")
        validation_headers = [
            "scope",
            "slice",
            "study_variant",
            "run_count",
            "mean_loss_pct",
            "p90_loss_pct",
            "p95_loss_pct",
            "max_loss_pct",
            "spread_fail_rate",
            "joint_safety_fail_rate",
            "spread_delta_p90",
            "spread_delta_p95",
            "pareto_ratio_delta_mean",
            "spread_tail_reduction_score",
            "spread_tail_reduction_score_p95",
            "hv_preservation_score",
            "joint_non_regression_score",
            "pareto_non_regression_score",
            "candidate_decision_hint",
        ]
        validation_display_rows = [
            [_format_number(row.get(header)) for header in validation_headers]
            for row in zdt1_spread_validation_rows
        ]
        lines.append(_markdown_table(validation_headers, validation_display_rows))
        lines.append("")

    if tail_risk_summary_rows:
        lines.append("## Tail Risk Summary")
        lines.append("")
        if any("study_variant" in row for row in tail_risk_summary_rows):
            if study.problem == "tsp":
                tail_headers = [
                    "scope",
                    "case_group",
                    "study_variant",
                    "mean_loss_pct",
                    "median_loss_pct",
                    "p75_loss_pct",
                    "p90_loss_pct",
                    "p95_loss_pct",
                    "max_loss_pct",
                    "rescue_target_mean_loss_pct",
                    "anti_case_mean_loss_pct",
                    "anti_case_p90_loss_pct",
                    "anti_case_p95_loss_pct",
                    "anti_case_max_loss_pct",
                    "tail_hardening_score",
                    "tail_hardening_score_p95",
                    "rescue_preservation_score",
                    "mean_regret_vs_q",
                    "mean_regret_vs_current_fast",
                    "configured_budget_mean",
                    "actual_evaluations_used_mean",
                    "runtime_seconds_mean",
                ]
            else:
                tail_headers = [
                    "scope",
                    "case_group",
                    "study_variant",
                    "mean_loss_pct",
                    "median_loss_pct",
                    "p75_loss_pct",
                    "p90_loss_pct",
                    "p95_loss_pct",
                    "max_loss_pct",
                    "joint_safety_fail_rate",
                    "spread_fail_rate",
                    "pareto_ratio_fail_rate",
                    "spread_delta_mean",
                    "spread_delta_p75",
                    "spread_delta_p90",
                    "spread_delta_p95",
                    "spread_delta_max",
                    "pareto_ratio_delta_mean",
                    "safety_hardening_score",
                    "spread_hardening_score",
                    "spread_tail_reduction_score",
                    "spread_tail_reduction_score_p95",
                    "joint_fail_reduction_score",
                    "hv_preservation_score",
                    "mean_regret_vs_q",
                    "mean_regret_vs_current_fast",
                    "configured_budget_mean",
                    "actual_evaluations_used_mean",
                    "runtime_seconds_mean",
                ]
        elif study.problem in {"tsp", "zdt1"}:
            tail_headers = [
                "scope",
                "case_group",
                "pair_count",
                "case_count",
                "mean_loss_pct",
                "median_loss_pct",
                "p75_loss_pct",
                "p90_loss_pct",
                "p95_loss_pct",
                "max_loss_pct",
                "decision_flip_rate",
                "pareto_ratio_fail_rate",
                "spread_fail_rate",
                "joint_safety_fail_rate",
                "actual_eval_savings_pct_mean",
                "runtime_savings_pct_mean",
            ]
        elif study.problem == "knapsack":
            tail_headers = [
                "scope",
                "case_group",
                "pair_count",
                "case_count",
                "repair_gain_vs_none_mean",
                "repair_gain_vs_none_p90",
                "repair_gain_vs_none_p95",
                "repair_gain_vs_none_max",
                "repair_gap_vs_greedy_mean",
                "repair_gap_vs_greedy_max_abs",
                "repair_feasible_ratio_mean",
                "repair_beats_none_rate",
                "repair_beats_greedy_rate",
            ]
        else:
            tail_headers = [
                "scope",
                "case_group",
                "pair_count",
                "case_count",
                "control_delta_vs_reference_mean",
                "control_delta_vs_reference_median",
                "control_delta_vs_reference_max_abs",
                "control_stable_rate",
            ]
        tail_rows = [
            [_format_number(row.get(header)) for header in tail_headers]
            for row in tail_risk_summary_rows
        ]
        lines.append(_markdown_table(tail_headers, tail_rows))
        lines.append("")

    if stress_case_catalog_rows:
        lines.append("## Stress Cases")
        lines.append("")
        stress_headers = [
            "problem",
            "case_id",
            "seed",
            "budget_band",
            "profile_compared",
            "regret_or_loss",
            "why_selected_as_stress_case",
            "case_group",
        ]
        stress_rows = [
            [_format_number(row.get(header)) for header in stress_headers]
            for row in stress_case_catalog_rows[:12]
        ]
        lines.append(_markdown_table(stress_headers, stress_rows))
        lines.append("")

    lines.append("## What To Read")
    lines.append("")
    if study.problem == "onemax":
        lines.append("- Compare `target_hit_rate` first, then `evaluations_to_target_mean`.")
        lines.append(
            "- Use `plot_stagnation.png` and `plot_diversity.png` to spot premature collapse."
        )
    elif study.problem == "knapsack":
        lines.append("- Prefer `best_feasible_fitness_mean` over penalized objective values.")
        lines.append(
            "- Use `feasible_rate` and `mean_violation_mean` to judge penalty sensitivity."
        )
        lines.append(
            "- Use `plot_feasibility.png` to see whether quality gains "
            "come from feasibility recovery."
        )
    elif study.problem == "tsp":
        lines.append("- Read `best_route_distance_mean` first. `best_fitness_mean` is auxiliary.")
        lines.append(
            "- Use `plot_diversity.png` and `plot_stagnation.png` to catch route-collapse."
        )
        lines.append(
            "- Use `plot_collapse_onset_vs_trigger.png` and `plot_post_trigger_gain.png` to tell"
            " whether the trigger helped after collapse or just fired like a schedule."
        )
        lines.append(
            "- For switching studies, add `plot_mode_switch_timeline.png`, "
            "`plot_collapse_onset_vs_switch.png`, and `plot_regret_vs_policy.png` "
            "to see whether runtime mode changes actually beat the best fixed policy."
        )
    else:
        lines.append("- Read `hypervolume_mean`, `pareto_ratio_mean`, and `spread_mean` together.")
        lines.append(
            "- Use `plot_diversity.png` and `plot_hypervolume.png` to inspect exploration balance."
        )
        lines.append("- Use the final Pareto scatter plot for the best variant.")
    lines.append("")
    return "\n".join(lines)


def _render_single_run_markdown(payload: dict[str, Any]) -> str:
    lines = [f"# Local Run: {payload['label']}", ""]
    lines.append(f"- Kind: `{payload['kind']}`")
    lines.append(f"- Problem: `{payload['problem']}`")
    if payload.get("source_name"):
        lines.append(f"- Source: `{payload['source_name']}`")
    lines.append(f"- Output dir: `{payload['artifact_dir']}`")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    for key, value in payload["metrics"].items():
        if key == "pareto_front_vectors":
            continue
        lines.append(f"- `{key}`: `{_format_number(value)}`")
    lines.append("")
    return "\n".join(lines)


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        return plt
    except ImportError:
        return None


def _plot_history_lines(
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    output_path: Path,
) -> bool:
    plt = _import_pyplot()
    if plt is None or not series:
        return False
    figure = plt.figure(figsize=(8, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    for label, points in series:
        if points:
            axis.plot([point[0] for point in points], [point[1] for point in points], label=label)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    if len(series) <= 8:
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _plot_box_comparison(
    title: str,
    x_labels: list[str],
    grouped_values: list[list[float]],
    y_label: str,
    output_path: Path,
) -> bool:
    plt = _import_pyplot()
    if plt is None or not grouped_values:
        return False
    figure_height = 4.5 if len(x_labels) <= 10 else 5.6
    figure = plt.figure(figsize=(max(8, len(x_labels) * 1.5), figure_height))
    axis = figure.add_subplot(1, 1, 1)
    axis.boxplot(grouped_values, labels=x_labels, patch_artist=True)
    axis.set_title(title)
    axis.set_xlabel("Variant")
    axis.set_ylabel(y_label)
    axis.grid(axis="y", alpha=0.3)
    for tick in axis.get_xticklabels():
        tick.set_rotation(30)
        tick.set_horizontalalignment("right")
    if len(x_labels) > 10:
        figure.subplots_adjust(bottom=0.4, top=0.9)
    else:
        figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _plot_summary_metric_boxes(
    title: str,
    y_label: str,
    raw_rows: list[dict[str, Any]],
    metric_name: str,
    output_path: Path,
) -> bool:
    grouped_values: list[list[float]] = []
    labels: list[str] = []
    for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
        values = [
            float(row[metric_name])
            for row in raw_rows
            if row["variant_label"] == variant
            and isinstance(row.get(metric_name), int | float)
            and not isinstance(row.get(metric_name), bool)
        ]
        if values:
            grouped_values.append(values)
            labels.append(variant)
    if not grouped_values:
        return False
    return _plot_box_comparison(
        title=title,
        x_labels=labels,
        grouped_values=grouped_values,
        y_label=y_label,
        output_path=output_path,
    )


def _plot_pareto_scatter(
    title: str,
    points: list[tuple[float, float]],
    output_path: Path,
) -> bool:
    plt = _import_pyplot()
    if plt is None or not points:
        return False
    figure = plt.figure(figsize=(5.5, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    axis.scatter([point[0] for point in points], [point[1] for point in points], alpha=0.75)
    axis.set_title(title)
    axis.set_xlabel("f1")
    axis.set_ylabel("f2")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _plot_metric_scatter(
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    output_path: Path,
) -> bool:
    plt = _import_pyplot()
    if plt is None or not series:
        return False
    figure = plt.figure(figsize=(6.0, 4.8))
    axis = figure.add_subplot(1, 1, 1)
    for label, points in series:
        if not points:
            continue
        axis.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            alpha=0.8,
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    if len(series) <= 8:
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _plot_categorical_bars(
    title: str,
    x_labels: list[str],
    values: list[float],
    y_label: str,
    output_path: Path,
) -> bool:
    plt = _import_pyplot()
    if plt is None or not x_labels or not values or len(x_labels) != len(values):
        return False
    width = max(6.0, 0.9 * len(x_labels))
    figure = plt.figure(figsize=(width, 4.8))
    axis = figure.add_subplot(1, 1, 1)
    positions = list(range(len(x_labels)))
    axis.bar(positions, values, color="#4c78a8")
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xticks(positions)
    axis.set_xticklabels(x_labels, rotation=35, ha="right")
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _plot_parameter_sweep(
    title: str,
    x_label: str,
    y_label: str,
    points: list[tuple[float, float, float | None, float | None]],
    output_path: Path,
) -> bool:
    plt = _import_pyplot()
    if plt is None or not points:
        return False
    ordered = sorted(points, key=lambda item: item[0])
    figure = plt.figure(figsize=(6.5, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    xs = [point[0] for point in ordered]
    ys = [point[1] for point in ordered]
    lower_errors = [
        max(0.0, point[1] - point[2]) if point[2] is not None else 0.0 for point in ordered
    ]
    upper_errors = [
        max(0.0, point[3] - point[1]) if point[3] is not None else 0.0 for point in ordered
    ]
    axis.errorbar(xs, ys, yerr=[lower_errors, upper_errors], marker="o", capsize=4)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _plot_parameter_sweep_series(
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    output_path: Path,
) -> bool:
    plt = _import_pyplot()
    if plt is None or not series:
        return False
    figure = plt.figure(figsize=(6.5, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    for label, points in series:
        ordered = sorted(points, key=lambda item: item[0])
        if not ordered:
            continue
        axis.plot(
            [point[0] for point in ordered],
            [point[1] for point in ordered],
            marker="o",
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    if len(series) <= 8:
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _plot_trigger_events(
    title: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    output_path: Path,
) -> bool:
    return _plot_metric_scatter(
        title=title,
        x_label="generation",
        y_label="seed",
        series=series,
        output_path=output_path,
    )


def _history_series(
    history_rows: list[dict[str, Any]],
    column: str,
) -> list[tuple[str, list[tuple[float, float]]]]:
    series: list[tuple[str, list[tuple[float, float]]]] = []
    for variant in sorted({str(row["variant_label"]) for row in history_rows}):
        points = [
            (float(row["generation"]), float(row[column]))
            for row in history_rows
            if row["variant_label"] == variant and isinstance(row.get(column), int | float)
        ]
        if points:
            series.append((variant, points))
    return series


def _plot_summary_axis_vs_metric(
    summary_rows: list[dict[str, Any]],
    *,
    axis_column: str,
    metric_column: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    include_row: Any | None = None,
) -> bool:
    series_by_budget: dict[str, list[tuple[float, float]]] = {}
    for row in summary_rows:
        if include_row is not None and not include_row(row):
            continue
        axis_value = row.get(axis_column)
        metric_value = row.get(metric_column)
        configured_budget = row.get("configured_budget")
        if not isinstance(axis_value, int | float) or not isinstance(metric_value, int | float):
            continue
        budget_label = (
            f"budget={int(configured_budget)}"
            if isinstance(configured_budget, int | float)
            else "budget=unknown"
        )
        series_by_budget.setdefault(budget_label, []).append(
            (float(axis_value), float(metric_value))
        )
    if not series_by_budget:
        return False
    return _plot_parameter_sweep_series(
        title=title,
        x_label=x_label,
        y_label=y_label,
        series=sorted(series_by_budget.items()),
        output_path=output_path,
    )


def _problem_diversity_column(problem: str) -> str | None:
    if problem in {"onemax", "knapsack"}:
        return "allele_entropy"
    if problem == "tsp":
        return "edge_diversity_ratio"
    if problem == "zdt1":
        return "population_spread"
    return None


def _study_plot_outputs(
    study: LocalStudy,
    raw_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    case_group_summary_rows: list[dict[str, Any]],
    ranking_detail_rows: list[dict[str, Any]],
    ranking_fidelity_rows: list[dict[str, Any]],
    triage_workflow_rows: list[dict[str, Any]],
    qf_pair_rows: list[dict[str, Any]],
    tolerance_rows: list[dict[str, Any]],
    seed_budget_rows: list[dict[str, Any]],
    sequential_decision_rows: list[dict[str, Any]],
    tsp_fast_tail_pair_rows: list[dict[str, Any]],
    tsp_fast_tail_summary_rows: list[dict[str, Any]],
    zdt1_spread_validation_rows: list[dict[str, Any]],
    stress_case_catalog_rows: list[dict[str, Any]],
    tail_risk_summary_rows: list[dict[str, Any]],
    failure_trace_rows: list[dict[str, Any]],
    study_dir: Path,
) -> dict[str, str]:
    plots: dict[str, str] = {}
    history_metric = _problem_history_metric(study.problem, study.plotting)
    series = _history_series(history_rows, "mean_metric")
    history_path = study_dir / "plot_convergence.png"
    if _plot_history_lines(
        title=f"{study.study_name}: {history_metric} vs generation",
        x_label="Generation",
        y_label=history_metric,
        series=series,
        output_path=history_path,
    ):
        plots["plot_convergence"] = str(history_path.resolve())

    grouped_values: list[list[float]] = []
    labels: list[str] = []
    for row in summary_rows:
        values = [
            float(raw_row[study.primary_metric])
            for raw_row in raw_rows
            if raw_row["variant_label"] == row["variant_label"]
            and isinstance(raw_row.get(study.primary_metric), int | float)
        ]
        if values:
            grouped_values.append(values)
            labels.append(str(row["variant_label"]))
    comparison_path = study_dir / "plot_primary_metric.png"
    if _plot_box_comparison(
        title=f"{study.study_name}: {study.primary_metric} by variant",
        x_labels=labels,
        grouped_values=grouped_values,
        y_label=study.primary_metric,
        output_path=comparison_path,
    ):
        plots["plot_primary_metric"] = str(comparison_path.resolve())

    diversity_column = _problem_diversity_column(study.problem)
    if diversity_column is not None:
        diversity_path = study_dir / "plot_diversity.png"
        if _plot_history_lines(
            title=f"{study.study_name}: {diversity_column} vs generation",
            x_label="Generation",
            y_label=diversity_column,
            series=_history_series(history_rows, diversity_column),
            output_path=diversity_path,
        ):
            plots["plot_diversity"] = str(diversity_path.resolve())

    stagnation_path = study_dir / "plot_stagnation.png"
    if _plot_history_lines(
        title=f"{study.study_name}: generations_since_last_improvement",
        x_label="Generation",
        y_label="generations_since_last_improvement",
        series=_history_series(history_rows, "generations_since_last_improvement"),
        output_path=stagnation_path,
    ):
        plots["plot_stagnation"] = str(stagnation_path.resolve())

    if study.problem == "knapsack":
        feasibility_path = study_dir / "plot_feasibility.png"
        if _plot_history_lines(
            title=f"{study.study_name}: feasible_ratio vs generation",
            x_label="Generation",
            y_label="feasible_ratio",
            series=_history_series(history_rows, "feasible_ratio"),
            output_path=feasibility_path,
        ):
            plots["plot_feasibility"] = str(feasibility_path.resolve())
        violation_path = study_dir / "plot_violation.png"
        if _plot_history_lines(
            title=f"{study.study_name}: mean_constraint_violation vs generation",
            x_label="Generation",
            y_label="mean_constraint_violation",
            series=_history_series(history_rows, "mean_constraint_violation"),
            output_path=violation_path,
        ):
            plots["plot_violation"] = str(violation_path.resolve())
        mutation_rate_path = study_dir / "plot_mutation_rate.png"
        if _plot_history_lines(
            title=f"{study.study_name}: adaptive_mutation_rate vs generation",
            x_label="Generation",
            y_label="adaptive_mutation_rate",
            series=_history_series(history_rows, "adaptive_mutation_rate"),
            output_path=mutation_rate_path,
        ):
            plots["plot_mutation_rate"] = str(mutation_rate_path.resolve())
        family_regret_path = study_dir / "plot_family_vs_regret.png"
        family_regret_values: list[list[float]] = []
        family_regret_labels: list[str] = []
        for family_label in sorted(
            {
                str(row.get("case_group") or row.get("family_label") or "overall")
                for row in raw_rows
                if isinstance(row.get("regret_vs_greedy_local_search"), int | float)
            }
        ):
            for variant in sorted(
                {
                    str(row.get("study_variant") or row.get("variant_label"))
                    for row in raw_rows
                    if str(row.get("case_group") or row.get("family_label") or "overall")
                    == family_label
                    and isinstance(row.get("regret_vs_greedy_local_search"), int | float)
                }
            ):
                values = [
                    float(row["regret_vs_greedy_local_search"])
                    for row in raw_rows
                    if str(row.get("case_group") or row.get("family_label") or "overall")
                    == family_label
                    and str(row.get("study_variant") or row.get("variant_label")) == variant
                    and isinstance(row.get("regret_vs_greedy_local_search"), int | float)
                ]
                if values:
                    family_regret_values.append(values)
                    family_regret_labels.append(f"{family_label}|{variant}")
        if family_regret_values and _plot_box_comparison(
            title=f"{study.study_name}: family vs regret to greedy",
            x_labels=family_regret_labels,
            grouped_values=family_regret_values,
            y_label="regret_vs_greedy_local_search",
            output_path=family_regret_path,
        ):
            plots["plot_family_vs_regret"] = str(family_regret_path.resolve())
        seeded_repair_gap_path = study_dir / "plot_seeded_vs_repair_gap.png"
        seeded_repair_values: list[list[float]] = []
        seeded_repair_labels: list[str] = []
        for variant in sorted(
            {
                str(row.get("study_variant") or row.get("variant_label"))
                for row in raw_rows
                if isinstance(row.get("regret_vs_none"), int | float)
            }
        ):
            values = [
                float(row["regret_vs_none"])
                for row in raw_rows
                if str(row.get("study_variant") or row.get("variant_label")) == variant
                and isinstance(row.get("regret_vs_none"), int | float)
            ]
            if values:
                seeded_repair_values.append(values)
                seeded_repair_labels.append(variant)
        if seeded_repair_values and _plot_box_comparison(
            title=f"{study.study_name}: seeded/repair gap vs plain GA",
            x_labels=seeded_repair_labels,
            grouped_values=seeded_repair_values,
            y_label="regret_vs_none",
            output_path=seeded_repair_gap_path,
        ):
            plots["plot_seeded_vs_repair_gap"] = str(seeded_repair_gap_path.resolve())
        repair_greedy_gap_path = study_dir / "plot_repair_vs_greedy_gap.png"
        repair_greedy_values: list[list[float]] = []
        repair_greedy_labels: list[str] = []
        for variant in sorted(
            {
                str(row.get("study_variant") or row.get("variant_label"))
                for row in raw_rows
                if isinstance(row.get("regret_vs_greedy_local_search"), int | float)
            }
        ):
            values = [
                float(row["regret_vs_greedy_local_search"])
                for row in raw_rows
                if str(row.get("study_variant") or row.get("variant_label")) == variant
                and isinstance(row.get("regret_vs_greedy_local_search"), int | float)
            ]
            if values:
                repair_greedy_values.append(values)
                repair_greedy_labels.append(variant)
        if repair_greedy_values and _plot_box_comparison(
            title=f"{study.study_name}: repair candidates vs greedy",
            x_labels=repair_greedy_labels,
            grouped_values=repair_greedy_values,
            y_label="regret_vs_greedy_local_search",
            output_path=repair_greedy_gap_path,
        ):
            plots["plot_repair_vs_greedy_gap"] = str(repair_greedy_gap_path.resolve())
        init_gain_path = study_dir / "plot_init_feasible_vs_final_gain.png"
        init_gain_series = []
        for variant in sorted(
            {
                str(row.get("study_variant") or row.get("variant_label"))
                for row in raw_rows
                if isinstance(row.get("initial_feasible_fraction"), int | float)
                and isinstance(row.get("init_to_final_gain"), int | float)
            }
        ):
            points = [
                (
                    float(row["initial_feasible_fraction"]),
                    float(row["init_to_final_gain"]),
                )
                for row in raw_rows
                if str(row.get("study_variant") or row.get("variant_label")) == variant
                and isinstance(row.get("initial_feasible_fraction"), int | float)
                and isinstance(row.get("init_to_final_gain"), int | float)
            ]
            if points:
                init_gain_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: initial feasible fraction vs final gain",
            x_label="initial_feasible_fraction",
            y_label="init_to_final_gain",
            series=init_gain_series,
            output_path=init_gain_path,
        ):
            plots["plot_init_feasible_vs_final_gain"] = str(init_gain_path.resolve())
        init_fraction_gain_path = study_dir / "plot_initial_feasible_fraction_vs_gain.png"
        if _plot_metric_scatter(
            title=f"{study.study_name}: initial feasible fraction vs final gain",
            x_label="initial_feasible_fraction",
            y_label="init_to_final_gain",
            series=init_gain_series,
            output_path=init_fraction_gain_path,
        ):
            plots["plot_initial_feasible_fraction_vs_gain"] = str(
                init_fraction_gain_path.resolve()
            )
        budget_gain_path = study_dir / "plot_budget_vs_feasible_gain.png"
        budget_gain_series = []
        for variant in sorted(
            {
                str(row.get("study_variant") or row.get("variant_label"))
                for row in raw_rows
                if isinstance(row.get("configured_budget"), int | float)
                and isinstance(row.get("init_to_final_gain"), int | float)
            }
        ):
            points = [
                (
                    float(row["configured_budget"]),
                    float(row["init_to_final_gain"]),
                )
                for row in raw_rows
                if str(row.get("study_variant") or row.get("variant_label")) == variant
                and isinstance(row.get("configured_budget"), int | float)
                and isinstance(row.get("init_to_final_gain"), int | float)
            ]
            if points:
                budget_gain_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: configured budget vs feasible gain",
            x_label="configured_budget",
            y_label="init_to_final_gain",
            series=budget_gain_series,
            output_path=budget_gain_path,
        ):
            plots["plot_budget_vs_feasible_gain"] = str(budget_gain_path.resolve())
        actual_eval_gain_path = study_dir / "plot_actual_eval_vs_feasible_gain.png"
        actual_eval_gain_series = []
        for variant in sorted(
            {
                str(row.get("study_variant") or row.get("variant_label"))
                for row in raw_rows
                if isinstance(
                    row.get("total_actual_evaluations_used", row.get("actual_evaluations_used")),
                    int | float,
                )
                and isinstance(row.get("init_to_final_gain"), int | float)
            }
        ):
            points = [
                (
                    float(
                        row.get("total_actual_evaluations_used", row.get("actual_evaluations_used"))
                    ),
                    float(row["init_to_final_gain"]),
                )
                for row in raw_rows
                if str(row.get("study_variant") or row.get("variant_label")) == variant
                and isinstance(
                    row.get("total_actual_evaluations_used", row.get("actual_evaluations_used")),
                    int | float,
                )
                and isinstance(row.get("init_to_final_gain"), int | float)
            ]
            if points:
                actual_eval_gain_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: total evaluations vs feasible gain",
            x_label="total_actual_evaluations_used",
            y_label="init_to_final_gain",
            series=actual_eval_gain_series,
            output_path=actual_eval_gain_path,
        ):
            plots["plot_actual_eval_vs_feasible_gain"] = str(actual_eval_gain_path.resolve())
        triage_gain_path = study_dir / "plot_triage_cost_vs_feasible_gain.png"
        triage_gain_series = []
        for variant in sorted(
            {
                str(row.get("study_variant") or row.get("variant_label"))
                for row in raw_rows
                if isinstance(
                    row.get("total_actual_evaluations_used", row.get("actual_evaluations_used")),
                    int | float,
                )
                and isinstance(row.get("best_feasible_fitness"), int | float)
            }
        ):
            points = [
                (
                    float(
                        row.get("total_actual_evaluations_used", row.get("actual_evaluations_used"))
                    ),
                    float(row["best_feasible_fitness"]),
                )
                for row in raw_rows
                if str(row.get("study_variant") or row.get("variant_label")) == variant
                and isinstance(
                    row.get("total_actual_evaluations_used", row.get("actual_evaluations_used")),
                    int | float,
                )
                and isinstance(row.get("best_feasible_fitness"), int | float)
            ]
            if points:
                triage_gain_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: total evaluations vs best feasible fitness",
            x_label="total_actual_evaluations_used",
            y_label="best_feasible_fitness",
            series=triage_gain_series,
            output_path=triage_gain_path,
        ):
            plots["plot_triage_cost_vs_feasible_gain"] = str(triage_gain_path.resolve())
        if any(isinstance(row.get("portfolio_restart_count"), int | float) for row in raw_rows):
            multistart_gap_path = study_dir / "plot_multistart_vs_repair_gap.png"
            multistart_gap_series = []
            for variant in sorted(
                {str(row.get("study_variant") or row.get("variant_label")) for row in raw_rows}
            ):
                points = [
                    (
                        float(
                            row.get(
                                "total_actual_evaluations_used",
                                row.get("actual_evaluations_used"),
                            )
                        ),
                        float(row["regret_vs_repair_only_reference"]),
                    )
                    for row in raw_rows
                    if str(row.get("study_variant") or row.get("variant_label")) == variant
                    and isinstance(
                        row.get(
                            "total_actual_evaluations_used",
                            row.get("actual_evaluations_used"),
                        ),
                        int | float,
                    )
                    and isinstance(row.get("regret_vs_repair_only_reference"), int | float)
                ]
                if points:
                    multistart_gap_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: total evaluations vs regret to repair_only",
                x_label="total_actual_evaluations_used",
                y_label="regret_vs_repair_only_reference",
                series=multistart_gap_series,
                output_path=multistart_gap_path,
            ):
                plots["plot_multistart_vs_repair_gap"] = str(multistart_gap_path.resolve())
        rerun_regret_metric = "regret_vs_greedy_local_search"
        rerun_regret_title = "regret to greedy"
        if not any(
            isinstance(row.get(rerun_regret_metric), int | float) for row in raw_rows
        ) and any(isinstance(row.get("regret_vs_repair_only"), int | float) for row in raw_rows):
            rerun_regret_metric = "regret_vs_repair_only"
            rerun_regret_title = "regret to repair_only"
        rerun_regret_path = study_dir / "plot_rerun_gate_vs_regret.png"
        rerun_regret_series = []
        for variant in sorted(
            {
                str(row.get("study_variant") or row.get("variant_label"))
                for row in raw_rows
                if isinstance(
                    row.get("total_actual_evaluations_used", row.get("actual_evaluations_used")),
                    int | float,
                )
                and isinstance(row.get(rerun_regret_metric), int | float)
            }
        ):
            points = [
                (
                    float(
                        row.get("total_actual_evaluations_used", row.get("actual_evaluations_used"))
                    ),
                    float(row[rerun_regret_metric]),
                )
                for row in raw_rows
                if str(row.get("study_variant") or row.get("variant_label")) == variant
                and isinstance(
                    row.get("total_actual_evaluations_used", row.get("actual_evaluations_used")),
                    int | float,
                )
                and isinstance(row.get(rerun_regret_metric), int | float)
            ]
            if points:
                rerun_regret_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: total evaluations vs {rerun_regret_title}",
            x_label="total_actual_evaluations_used",
            y_label=rerun_regret_metric,
            series=rerun_regret_series,
            output_path=rerun_regret_path,
        ):
            plots["plot_rerun_gate_vs_regret"] = str(rerun_regret_path.resolve())
        rerun_value_path = study_dir / "plot_initial_feasible_fraction_vs_rerun_value.png"
        rerun_value_series = []
        for variant in sorted(
            {
                str(row.get("study_variant") or row.get("variant_label"))
                for row in raw_rows
                if isinstance(row.get("initial_feasible_fraction"), int | float)
                and isinstance(row.get("regret_vs_repair_only"), int | float)
            }
        ):
            points = [
                (
                    float(row["initial_feasible_fraction"]),
                    -float(row["regret_vs_repair_only"]),
                )
                for row in raw_rows
                if str(row.get("study_variant") or row.get("variant_label")) == variant
                and isinstance(row.get("initial_feasible_fraction"), int | float)
                and isinstance(row.get("regret_vs_repair_only"), int | float)
            ]
            if points:
                rerun_value_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: initial feasible fraction vs gain over repair_only",
            x_label="initial_feasible_fraction",
            y_label="gain_over_repair_only",
            series=rerun_value_series,
            output_path=rerun_value_path,
        ):
            plots["plot_initial_feasible_fraction_vs_rerun_value"] = str(
                rerun_value_path.resolve()
            )
        capacity_gain_path = study_dir / "plot_capacity_tightness_vs_gain.png"
        capacity_gain_series = []
        for variant in sorted(
            {
                str(row.get("study_variant") or row.get("variant_label"))
                for row in raw_rows
                if isinstance(row.get("capacity_ratio"), int | float)
                and isinstance(row.get("regret_vs_none"), int | float)
            }
        ):
            points = [
                (
                    float(row["capacity_ratio"]),
                    -float(row["regret_vs_none"]),
                )
                for row in raw_rows
                if str(row.get("study_variant") or row.get("variant_label")) == variant
                and isinstance(row.get("capacity_ratio"), int | float)
                and isinstance(row.get("regret_vs_none"), int | float)
            ]
            if points:
                capacity_gain_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: capacity tightness vs gain over plain GA",
            x_label="capacity_ratio",
            y_label="gain_over_none",
            series=capacity_gain_series,
            output_path=capacity_gain_path,
        ):
            plots["plot_capacity_tightness_vs_gain"] = str(capacity_gain_path.resolve())

    if study.problem == "tsp":
        route_path = study_dir / "plot_route_distance.png"
        if _plot_history_lines(
            title=f"{study.study_name}: best_route_distance vs generation",
            x_label="Generation",
            y_label="best_route_distance",
            series=_history_series(history_rows, "best_route_distance"),
            output_path=route_path,
        ):
            plots["plot_route_distance"] = str(route_path.resolve())
        diversity_distance_path = study_dir / "plot_diversity_vs_distance.png"
        tsp_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["final_edge_diversity_ratio"]),
                    float(row["best_route_distance"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("final_edge_diversity_ratio"), int | float)
                and isinstance(row.get("best_route_distance"), int | float)
            ]
            if points:
                tsp_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: final edge diversity vs route distance",
            x_label="final_edge_diversity_ratio",
            y_label="best_route_distance",
            series=tsp_series,
            output_path=diversity_distance_path,
        ):
            plots["plot_diversity_vs_distance"] = str(diversity_distance_path.resolve())
        if ranking_detail_rows:
            rank_scatter_path = study_dir / "plot_fast_vs_canonical_rank.png"
            rank_scatter_series = []
            for case_group in sorted(
                {
                    str(row.get("case_group") or "overall")
                    for row in ranking_detail_rows
                    if isinstance(row.get("canonical_rank"), int | float)
                    and isinstance(row.get("fast_rank"), int | float)
                }
            ):
                points = [
                    (
                        float(row["fast_rank"]),
                        float(row["canonical_rank"]),
                    )
                    for row in ranking_detail_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("canonical_rank"), int | float)
                    and isinstance(row.get("fast_rank"), int | float)
                ]
                if points:
                    rank_scatter_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: fast rank vs canonical rank",
                x_label="fast_rank",
                y_label="canonical_rank",
                series=rank_scatter_series,
                output_path=rank_scatter_path,
            ):
                plots["plot_fast_vs_canonical_rank"] = str(rank_scatter_path.resolve())
        if triage_workflow_rows:
            recall_budget_path = study_dir / "plot_topk_recall_vs_budget.png"
            recall_budget_series = []
            for case_group in sorted(
                {
                    str(row.get("case_group") or "overall")
                    for row in triage_workflow_rows
                    if str(row.get("scope")) != "case"
                    and isinstance(row.get("total_actual_evaluations_used"), int | float)
                    and isinstance(row.get("oracle_hit_rate"), int | float)
                }
            ):
                points = [
                    (
                        float(row["total_actual_evaluations_used"]),
                        float(row["oracle_hit_rate"]),
                    )
                    for row in triage_workflow_rows
                    if str(row.get("scope")) != "case"
                    and str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("total_actual_evaluations_used"), int | float)
                    and isinstance(row.get("oracle_hit_rate"), int | float)
                ]
                if points:
                    recall_budget_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: triage recall vs total evaluations",
                x_label="total_actual_evaluations_used",
                y_label="oracle_hit_rate",
                series=recall_budget_series,
                output_path=recall_budget_path,
            ):
                plots["plot_topk_recall_vs_budget"] = str(recall_budget_path.resolve())
            triage_regret_path = study_dir / "plot_triage_cost_vs_regret.png"
            triage_regret_series = []
            for workflow in sorted(
                {
                    str(row.get("workflow") or "")
                    for row in triage_workflow_rows
                    if str(row.get("scope")) != "case"
                    and isinstance(row.get("total_actual_evaluations_used"), int | float)
                    and isinstance(row.get("final_regret_vs_full_canonical"), int | float)
                }
            ):
                points = [
                    (
                        float(row["total_actual_evaluations_used"]),
                        float(row["final_regret_vs_full_canonical"]),
                    )
                    for row in triage_workflow_rows
                    if str(row.get("scope")) != "case"
                    and str(row.get("workflow") or "") == workflow
                    and isinstance(row.get("total_actual_evaluations_used"), int | float)
                    and isinstance(row.get("final_regret_vs_full_canonical"), int | float)
                ]
                if points:
                    triage_regret_series.append((workflow, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: triage cost vs regret",
                x_label="total_actual_evaluations_used",
                y_label="final_regret_vs_full_canonical",
                series=triage_regret_series,
                output_path=triage_regret_path,
            ):
                plots["plot_triage_cost_vs_regret"] = str(triage_regret_path.resolve())
        if ranking_fidelity_rows:
            fidelity_group_path = study_dir / "plot_rescue_target_vs_anticase_rank_fidelity.png"
            fidelity_group_series = []
            for case_group in sorted(
                {
                    str(row.get("case_group") or "overall")
                    for row in ranking_fidelity_rows
                    if str(row.get("scope")) == "case_group"
                    and isinstance(row.get("spearman_rank_correlation"), int | float)
                    and isinstance(row.get("fast_top1_regret_vs_canonical_best"), int | float)
                }
            ):
                points = [
                    (
                        float(row["spearman_rank_correlation"]),
                        float(row["fast_top1_regret_vs_canonical_best"]),
                    )
                    for row in ranking_fidelity_rows
                    if str(row.get("scope")) == "case_group"
                    and str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("spearman_rank_correlation"), int | float)
                    and isinstance(row.get("fast_top1_regret_vs_canonical_best"), int | float)
                ]
                if points:
                    fidelity_group_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: rank fidelity by case group",
                x_label="spearman_rank_correlation",
                y_label="fast_top1_regret_vs_canonical_best",
                series=fidelity_group_series,
                output_path=fidelity_group_path,
            ):
                plots["plot_rescue_target_vs_anticase_rank_fidelity"] = str(
                    fidelity_group_path.resolve()
                )
        if qf_pair_rows:
            qf_config = _qf_tolerance_config(study) or {}
            qf_plot_suffix = (
                str(qf_config.get("plot_name_suffix") or "").strip()
                if isinstance(qf_config.get("plot_name_suffix"), str)
                else ""
            )
            loss_distribution_path = study_dir / _plot_file_name(
                "plot_q_vs_f_loss_distribution.png",
                qf_plot_suffix,
            )
            loss_labels = ["overall"]
            loss_values = [
                [float(row["route_distance_loss_pct"]) for row in qf_pair_rows]
            ]
            for case_group in ("rescue_target", "anti_case"):
                values = [
                    float(row["route_distance_loss_pct"])
                    for row in qf_pair_rows
                    if str(row.get("case_group") or "overall") == case_group
                ]
                if values:
                    loss_labels.append(case_group)
                    loss_values.append(values)
            if _plot_box_comparison(
                title=f"{study.study_name}: Q vs F route-distance loss",
                x_labels=loss_labels,
                grouped_values=loss_values,
                y_label="route_distance_loss_pct",
                output_path=loss_distribution_path,
            ):
                plots["plot_q_vs_f_loss_distribution"] = str(
                    loss_distribution_path.resolve()
                )

            accept_rate_path = study_dir / _plot_file_name(
                "plot_tolerance_accept_rate.png",
                qf_plot_suffix,
            )
            accept_rate_series = []
            for case_group in sorted(
                {
                    str(row.get("case_group") or "overall")
                    for row in tolerance_rows
                    if isinstance(row.get("tolerance_bin_pct"), int | float)
                    and isinstance(row.get("acceptable_rate"), int | float)
                }
            ):
                points = [
                    (float(row["tolerance_bin_pct"]), float(row["acceptable_rate"]))
                    for row in tolerance_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("tolerance_bin_pct"), int | float)
                    and isinstance(row.get("acceptable_rate"), int | float)
                ]
                if points:
                    accept_rate_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: tolerance bin vs F acceptable rate",
                x_label="tolerance_bin_pct",
                y_label="acceptable_rate",
                series=accept_rate_series,
                output_path=accept_rate_path,
            ):
                plots["plot_tolerance_accept_rate"] = str(accept_rate_path.resolve())

            split_loss_path = study_dir / _plot_file_name(
                "plot_rescue_vs_anticase_loss.png",
                qf_plot_suffix,
            )
            split_labels = []
            split_values = []
            for case_group in ("rescue_target", "anti_case"):
                values = [
                    float(row["route_distance_loss_pct"])
                    for row in qf_pair_rows
                    if str(row.get("case_group") or "overall") == case_group
                ]
                if values:
                    split_labels.append(case_group)
                    split_values.append(values)
            if split_values and _plot_box_comparison(
                title=f"{study.study_name}: rescue-target vs anti-case F loss",
                x_labels=split_labels,
                grouped_values=split_values,
                y_label="route_distance_loss_pct",
                output_path=split_loss_path,
            ):
                plots["plot_rescue_vs_anticase_loss"] = str(split_loss_path.resolve())

            savings_loss_path = study_dir / _plot_file_name(
                "plot_budget_savings_vs_quality_loss.png",
                qf_plot_suffix,
            )
            savings_loss_series = []
            for case_group in sorted(
                {str(row.get("case_group") or "overall") for row in qf_pair_rows}
            ):
                points = [
                    (
                        float(row["actual_eval_savings_pct"]),
                        float(row["route_distance_loss_pct"]),
                    )
                    for row in qf_pair_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("actual_eval_savings_pct"), int | float)
                    and isinstance(row.get("route_distance_loss_pct"), int | float)
                ]
                if points:
                    savings_loss_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: budget savings vs quality loss",
                x_label="actual_eval_savings_pct",
                y_label="route_distance_loss_pct",
                series=savings_loss_series,
                output_path=savings_loss_path,
            ):
                plots["plot_budget_savings_vs_quality_loss"] = str(
                    savings_loss_path.resolve()
                )
        if tsp_fast_tail_pair_rows and tsp_fast_tail_summary_rows:
            tail_config = _tsp_fast_tail_config(study) or {}
            tail_distribution_path = study_dir / _stress_plot_file_name(
                tail_config,
                "tail_distribution",
                "plot_q_vs_f_tail_distribution.png",
            )
            candidate_labels = []
            candidate_values = []
            for variant in sorted({str(row.get("study_variant") or "") for row in tsp_fast_tail_pair_rows}):
                values = [
                    float(row["route_distance_loss_pct_vs_quality"])
                    for row in tsp_fast_tail_pair_rows
                    if str(row.get("study_variant") or "") == variant
                    and isinstance(row.get("route_distance_loss_pct_vs_quality"), int | float)
                ]
                if values:
                    candidate_labels.append(variant)
                    candidate_values.append(values)
            if candidate_values and _plot_box_comparison(
                title=f"{study.study_name}: candidate loss vs Q",
                x_labels=candidate_labels,
                grouped_values=candidate_values,
                y_label="route_distance_loss_pct_vs_quality",
                output_path=tail_distribution_path,
            ):
                plots["plot_q_vs_f_tail_distribution"] = str(tail_distribution_path.resolve())

            overall_rows = [
                row
                for row in tsp_fast_tail_summary_rows
                if str(row.get("scope")) == "overall"
            ]
            anti_p90_path = study_dir / _stress_plot_file_name(
                tail_config,
                "candidate_vs_anti_case_p90",
                "plot_candidate_vs_anti_case_p90.png",
            )
            anti_p90_labels = [
                str(row.get("study_variant") or "")
                for row in overall_rows
                if isinstance(row.get("anti_case_p90_loss_pct"), int | float)
            ]
            anti_p90_values = [
                float(row["anti_case_p90_loss_pct"])
                for row in overall_rows
                if isinstance(row.get("anti_case_p90_loss_pct"), int | float)
            ]
            if _plot_categorical_bars(
                title=f"{study.study_name}: anti-case p90 loss by candidate",
                x_labels=anti_p90_labels,
                values=anti_p90_values,
                y_label="anti_case_p90_loss_pct",
                output_path=anti_p90_path,
            ):
                plots["plot_candidate_vs_anti_case_p90"] = str(anti_p90_path.resolve())
                plots["plot_currentF_vs_candidate_tail"] = str(anti_p90_path.resolve())

            anti_p90_p95_path = study_dir / _stress_plot_file_name(
                tail_config,
                "anticase_p90_p95_reduction",
                "plot_anticase_p90_p95_reduction.png",
            )
            anti_p90_p95_series = []
            anti_p90_points = [
                (float(index), float(row["anti_case_p90_loss_pct"]))
                for index, row in enumerate(overall_rows)
                if isinstance(row.get("anti_case_p90_loss_pct"), int | float)
            ]
            anti_p95_points = [
                (float(index), float(row["anti_case_p95_loss_pct"]))
                for index, row in enumerate(overall_rows)
                if isinstance(row.get("anti_case_p95_loss_pct"), int | float)
            ]
            if anti_p90_points:
                anti_p90_p95_series.append(("anti_case_p90", anti_p90_points))
            if anti_p95_points:
                anti_p90_p95_series.append(("anti_case_p95", anti_p95_points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: anti-case p90/p95 contour comparison",
                x_label="candidate_index",
                y_label="loss_pct",
                series=anti_p90_p95_series,
                output_path=anti_p90_p95_path,
            ):
                plots["plot_anticase_p90_p95_reduction"] = str(anti_p90_p95_path.resolve())

            anti_p95_max_path = study_dir / _stress_plot_file_name(
                tail_config,
                "anticase_p95_max_reduction",
                "plot_anticase_p95_max_reduction.png",
            )
            anti_p95_max_series = []
            anti_p95_max_points = [
                (
                    float(row["anti_case_p95_loss_pct"]),
                    float(row["anti_case_max_loss_pct"]),
                )
                for row in overall_rows
                if isinstance(row.get("anti_case_p95_loss_pct"), int | float)
                and isinstance(row.get("anti_case_max_loss_pct"), int | float)
            ]
            if anti_p95_max_points:
                anti_p95_max_series.append(("candidate", anti_p95_max_points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: anti-case p95 vs max contour comparison",
                x_label="anti_case_p95_loss_pct",
                y_label="anti_case_max_loss_pct",
                series=anti_p95_max_series,
                output_path=anti_p95_max_path,
            ):
                plots["plot_anticase_p95_max_reduction"] = str(
                    anti_p95_max_path.resolve()
                )

            rescue_mean_path = study_dir / _stress_plot_file_name(
                tail_config,
                "candidate_vs_rescue_mean",
                "plot_candidate_vs_rescue_mean.png",
            )
            rescue_mean_labels = [
                str(row.get("study_variant") or "")
                for row in overall_rows
                if isinstance(row.get("rescue_target_mean_loss_pct"), int | float)
            ]
            rescue_mean_values = [
                float(row["rescue_target_mean_loss_pct"])
                for row in overall_rows
                if isinstance(row.get("rescue_target_mean_loss_pct"), int | float)
            ]
            if _plot_categorical_bars(
                title=f"{study.study_name}: rescue-target mean loss by candidate",
                x_labels=rescue_mean_labels,
                values=rescue_mean_values,
                y_label="rescue_target_mean_loss_pct",
                output_path=rescue_mean_path,
            ):
                plots["plot_candidate_vs_rescue_mean"] = str(rescue_mean_path.resolve())

            rescue_anti_tail_path = study_dir / _stress_plot_file_name(
                tail_config,
                "rescue_vs_anticase_tail",
                "plot_rescue_vs_anticase_tail.png",
            )
            rescue_anti_series = [
                (
                    "candidate",
                    [
                        (
                            float(row["rescue_target_mean_loss_pct"]),
                            float(row["anti_case_p90_loss_pct"]),
                        )
                        for row in overall_rows
                        if isinstance(row.get("rescue_target_mean_loss_pct"), int | float)
                        and isinstance(row.get("anti_case_p90_loss_pct"), int | float)
                    ],
                )
            ]
            rescue_anti_series = [(label, points) for label, points in rescue_anti_series if points]
            if _plot_metric_scatter(
                title=f"{study.study_name}: rescue-target mean vs anti-case p90 tail",
                x_label="rescue_target_mean_loss_pct",
                y_label="anti_case_p90_loss_pct",
                series=rescue_anti_series,
                output_path=rescue_anti_tail_path,
            ):
                plots["plot_rescue_vs_anticase_tail"] = str(rescue_anti_tail_path.resolve())

            budget_tail_path = study_dir / _stress_plot_file_name(
                tail_config,
                "budget_vs_tail_loss",
                "plot_budget_vs_tail_loss.png",
            )
            budget_tail_series = []
            budget_tail_points = [
                (
                    float(row["configured_budget_mean"]),
                    float(row["anti_case_p90_loss_pct"]),
                )
                for row in overall_rows
                if isinstance(row.get("configured_budget_mean"), int | float)
                and isinstance(row.get("anti_case_p90_loss_pct"), int | float)
            ]
            if budget_tail_points:
                budget_tail_series.append(("overall", budget_tail_points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: configured budget vs anti-case p90 loss",
                x_label="configured_budget_mean",
                y_label="anti_case_p90_loss_pct",
                series=budget_tail_series,
                output_path=budget_tail_path,
            ):
                plots["plot_budget_vs_tail_loss"] = str(budget_tail_path.resolve())

            seed_tail_path = study_dir / _stress_plot_file_name(
                tail_config,
                "seed_fraction_vs_tail",
                "plot_seed_fraction_vs_tail.png",
            )
            seed_tail_series: dict[str, list[tuple[float, float]]] = {}
            for row in overall_rows:
                seed_fraction = row.get("algorithm_options.seed_fraction")
                anti_p90 = row.get("anti_case_p90_loss_pct")
                if not isinstance(seed_fraction, int | float) or not isinstance(anti_p90, int | float):
                    continue
                mutation_name = str(row.get("mutation") or "unknown")
                seed_tail_series.setdefault(mutation_name, []).append(
                    (float(seed_fraction), float(anti_p90))
                )
            if _plot_metric_scatter(
                title=f"{study.study_name}: seed fraction vs anti-case p90 loss",
                x_label="seed_fraction",
                y_label="anti_case_p90_loss_pct",
                series=sorted(seed_tail_series.items()),
                output_path=seed_tail_path,
            ):
                plots["plot_seed_fraction_vs_tail"] = str(seed_tail_path.resolve())

            operator_tail_path = study_dir / _stress_plot_file_name(
                tail_config,
                "operator_vs_tail",
                "plot_operator_vs_tail.png",
            )
            operator_labels = []
            operator_values = []
            for mutation_name in sorted(
                {
                    str(row.get("mutation") or "unknown")
                    for row in tsp_fast_tail_pair_rows
                    if str(row.get("case_group") or "overall") == "anti_case"
                }
            ):
                values = [
                    float(row["route_distance_loss_pct_vs_quality"])
                    for row in tsp_fast_tail_pair_rows
                    if str(row.get("case_group") or "overall") == "anti_case"
                    and str(row.get("mutation") or "unknown") == mutation_name
                    and isinstance(row.get("route_distance_loss_pct_vs_quality"), int | float)
                ]
                if values:
                    operator_labels.append(mutation_name)
                    operator_values.append(values)
            if operator_values and _plot_box_comparison(
                title=f"{study.study_name}: anti-case tail by mutation operator",
                x_labels=operator_labels,
                grouped_values=operator_values,
                y_label="route_distance_loss_pct_vs_quality",
                output_path=operator_tail_path,
            ):
                plots["plot_operator_vs_tail"] = str(operator_tail_path.resolve())

            tail_freeze_config = _tsp_tail_freeze_config(study)
            if tail_freeze_config is not None:
                freeze_summary_path = study_dir / "plot_tsp_tail_freeze_summary.png"
                freeze_labels: list[str] = []
                freeze_values: list[float] = []
                current_fast_overall = next(
                    (
                        row
                        for row in overall_rows
                        if str(row.get("study_variant") or "") == "current_fast"
                    ),
                    None,
                )
                if current_fast_overall is not None:
                    for label, key in (
                        ("anti_p95", "anti_case_p95_loss_pct"),
                        ("anti_max", "anti_case_max_loss_pct"),
                        ("rescue_mean", "rescue_target_mean_loss_pct"),
                    ):
                        value = current_fast_overall.get(key)
                        if isinstance(value, int | float):
                            freeze_labels.append(label)
                            freeze_values.append(float(value))
                if freeze_values and _plot_categorical_bars(
                    title=f"{study.study_name}: frozen TSP fast tail readout",
                    x_labels=freeze_labels,
                    values=freeze_values,
                    y_label="loss_pct",
                    output_path=freeze_summary_path,
                ):
                    plots["plot_tsp_tail_freeze_summary"] = str(
                        freeze_summary_path.resolve()
                    )

                anticase_qf_path = study_dir / "plot_anticase_q_vs_f_tail.png"
                anticase_fast_values = [
                    float(row["route_distance_loss_pct_vs_quality"])
                    for row in tsp_fast_tail_pair_rows
                    if str(row.get("study_variant") or "") == "current_fast"
                    and str(row.get("case_group") or "") == "anti_case"
                    and isinstance(row.get("route_distance_loss_pct_vs_quality"), int | float)
                ]
                q_reference_values = [0.0 for _ in anticase_fast_values]
                if anticase_fast_values and _plot_box_comparison(
                    title=f"{study.study_name}: anti-case Q vs current F tail",
                    x_labels=["quality_first", "current_fast"],
                    grouped_values=[q_reference_values, anticase_fast_values],
                    y_label="route_distance_loss_pct_vs_quality",
                    output_path=anticase_qf_path,
                ):
                    plots["plot_anticase_q_vs_f_tail"] = str(anticase_qf_path.resolve())

            legacy_plot_name = tail_config.get("legacy_reference_plot_name")
            if isinstance(legacy_plot_name, str) and legacy_plot_name.strip():
                legacy_focus_case_group = str(
                    tail_config.get("legacy_reference_case_group") or "anti_case"
                )
                legacy_variant_order = tail_config.get("variant_order")
                ordered_variants: list[str] = []
                if isinstance(legacy_variant_order, list):
                    ordered_variants.extend(
                        str(value).strip()
                        for value in legacy_variant_order
                        if isinstance(value, str) and str(value).strip()
                    )
                for variant in sorted(
                    {
                        str(row.get("study_variant") or "")
                        for row in tsp_fast_tail_pair_rows
                        if str(row.get("case_group") or "overall") == legacy_focus_case_group
                    }
                ):
                    if variant and variant not in ordered_variants:
                        ordered_variants.append(variant)
                legacy_labels = []
                legacy_values = []
                for variant in ordered_variants:
                    values = [
                        float(row["route_distance_loss_pct_vs_quality"])
                        for row in tsp_fast_tail_pair_rows
                        if str(row.get("case_group") or "overall") == legacy_focus_case_group
                        and str(row.get("study_variant") or "") == variant
                        and isinstance(row.get("route_distance_loss_pct_vs_quality"), int | float)
                    ]
                    if values:
                        legacy_labels.append(variant)
                        legacy_values.append(values)
                legacy_plot_path = study_dir / legacy_plot_name.strip()
                if legacy_values and _plot_box_comparison(
                title=f"{study.study_name}: {legacy_focus_case_group} old-fast vs new-fast tail",
                x_labels=legacy_labels,
                grouped_values=legacy_values,
                y_label="route_distance_loss_pct_vs_quality",
                output_path=legacy_plot_path,
                ):
                    plots["plot_old_fast_vs_new_fast_tail"] = str(legacy_plot_path.resolve())
        seed_fraction_gap_path = study_dir / "plot_seed_fraction_vs_gap.png"
        seed_fraction_series: dict[str, list[tuple[float, float]]] = {}
        for row in raw_rows:
            seed_fraction = row.get("algorithm_options.seed_fraction")
            regret_value = _tsp_regret_value(row)
            if not isinstance(seed_fraction, int | float) or not isinstance(
                regret_value,
                int | float,
            ):
                continue
            mutation_name = row.get("mutation", "unknown")
            case_group = row.get("case_group") or "overall"
            budget_value = row.get("configured_budget")
            budget_label = (
                f"budget={int(budget_value)}"
                if isinstance(budget_value, int | float)
                else "budget=unknown"
            )
            series_label = f"{case_group} | {mutation_name} | {budget_label}"
            seed_fraction_series.setdefault(series_label, []).append(
                (float(seed_fraction), float(regret_value))
            )
        if _plot_metric_scatter(
            title=f"{study.study_name}: seed fraction vs regret",
            x_label="seed_fraction",
            y_label="regret_vs_current_preferred_profile",
            series=sorted(seed_fraction_series.items()),
            output_path=seed_fraction_gap_path,
        ):
            plots["plot_seed_fraction_vs_gap"] = str(seed_fraction_gap_path.resolve())
        seed_source_gap_path = study_dir / "plot_seed_source_vs_gap.png"
        seed_source_values: list[list[float]] = []
        seed_source_labels: list[str] = []
        for init_strategy in sorted(
            {
                str(row.get("algorithm_options.init_strategy") or "none")
                for row in raw_rows
                if _tsp_regret_value(row) is not None
            }
        ):
            values = [
                float(gap_value)
                for row in raw_rows
                for gap_value in [_tsp_regret_value(row)]
                if gap_value is not None
                and str(row.get("algorithm_options.init_strategy") or "none") == init_strategy
            ]
            if values:
                seed_source_values.append(values)
                seed_source_labels.append(init_strategy)
        if seed_source_values and _plot_box_comparison(
            title=f"{study.study_name}: seed source vs regret",
            x_labels=seed_source_labels,
            grouped_values=seed_source_values,
            y_label="regret_vs_current_preferred_profile",
            output_path=seed_source_gap_path,
        ):
            plots["plot_seed_source_vs_gap"] = str(seed_source_gap_path.resolve())
        mutation_gap_path = study_dir / "plot_mutation_operator_vs_gap.png"
        mutation_values: list[list[float]] = []
        mutation_labels: list[str] = []
        for mutation_name in sorted(
            {
                str(row["mutation"])
                for row in raw_rows
                if isinstance(row.get("mutation"), str)
                and _tsp_regret_value(row) is not None
            }
        ):
            values = [
                float(gap_value)
                for row in raw_rows
                for gap_value in [_tsp_regret_value(row)]
                if row.get("mutation") == mutation_name
                and gap_value is not None
            ]
            if values:
                mutation_values.append(values)
                mutation_labels.append(mutation_name)
        if mutation_values and _plot_box_comparison(
            title=f"{study.study_name}: mutation operator vs regret",
            x_labels=mutation_labels,
            grouped_values=mutation_values,
            y_label="regret_vs_current_preferred_profile",
            output_path=mutation_gap_path,
        ):
            plots["plot_mutation_operator_vs_gap"] = str(mutation_gap_path.resolve())
        initial_quality_gap_path = study_dir / "plot_initial_quality_vs_final_gap.png"
        initial_quality_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["initial_best_route_distance"]),
                    float(gap_value),
                )
                for row in raw_rows
                for gap_value in [_tsp_regret_value(row)]
                if row["variant_label"] == variant
                and isinstance(row.get("initial_best_route_distance"), int | float)
                and gap_value is not None
            ]
            if points:
                initial_quality_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: initial best route vs final gap",
            x_label="initial_best_route_distance",
            y_label="regret_vs_current_preferred_profile",
            series=initial_quality_series,
            output_path=initial_quality_gap_path,
        ):
            plots["plot_initial_quality_vs_final_gap"] = str(
                initial_quality_gap_path.resolve()
            )
        post_trigger_path = study_dir / "plot_post_trigger_gain.png"
        if _plot_summary_metric_boxes(
            title=f"{study.study_name}: post_trigger_improvement by variant",
            y_label="post_trigger_improvement",
            raw_rows=raw_rows,
            metric_name="post_trigger_improvement",
            output_path=post_trigger_path,
        ):
            plots["plot_post_trigger_gain"] = str(post_trigger_path.resolve())
        refresh_gain_path = study_dir / "plot_refresh_schedule_vs_gain.png"
        refresh_gain_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["total_refresh_fraction_realized"]),
                    float(row["post_trigger_improvement"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("total_refresh_fraction_realized"), int | float)
                and isinstance(row.get("post_trigger_improvement"), int | float)
            ]
            if points:
                refresh_gain_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: refresh schedule vs post-trigger gain",
            x_label="total_refresh_fraction_realized",
            y_label="post_trigger_improvement",
            series=refresh_gain_series,
            output_path=refresh_gain_path,
        ):
            plots["plot_refresh_schedule_vs_gain"] = str(refresh_gain_path.resolve())
        refresh_volume_path = study_dir / "plot_refresh_volume_vs_gain.png"
        if _plot_metric_scatter(
            title=f"{study.study_name}: realized refresh volume vs post-trigger gain",
            x_label="realized_refresh_volume",
            y_label="post_trigger_improvement",
            series=refresh_gain_series,
            output_path=refresh_volume_path,
        ):
            plots["plot_refresh_volume_vs_gain"] = str(refresh_volume_path.resolve())
        collapse_trigger_path = study_dir / "plot_collapse_onset_vs_trigger.png"
        collapse_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["collapse_onset_generation"]),
                    float(row["first_trigger_generation"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("collapse_onset_generation"), int | float)
                and isinstance(row.get("first_trigger_generation"), int | float)
            ]
            if points:
                collapse_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: collapse onset vs first trigger",
            x_label="collapse_onset_generation",
            y_label="first_trigger_generation",
            series=collapse_series,
            output_path=collapse_trigger_path,
        ):
            plots["plot_collapse_onset_vs_trigger"] = str(collapse_trigger_path.resolve())
        budget_gap_path = study_dir / "plot_budget_band_vs_gap.png"
        budget_gap_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["configured_budget"]),
                    float(row["best_route_distance"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("configured_budget"), int | float)
                and isinstance(row.get("best_route_distance"), int | float)
            ]
            if points:
                budget_gap_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: configured budget vs route distance",
            x_label="configured_budget",
            y_label="best_route_distance",
            series=budget_gap_series,
            output_path=budget_gap_path,
        ):
            plots["plot_budget_band_vs_gap"] = str(budget_gap_path.resolve())
            plots["plot_budget_vs_distance"] = str(budget_gap_path.resolve())
        budget_runtime_path = study_dir / "plot_budget_vs_runtime.png"
        budget_runtime_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["configured_budget"]),
                    float(row["runtime_seconds"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("configured_budget"), int | float)
                and isinstance(row.get("runtime_seconds"), int | float)
            ]
            if points:
                budget_runtime_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: configured budget vs runtime",
            x_label="configured_budget",
            y_label="runtime_seconds",
            series=budget_runtime_series,
            output_path=budget_runtime_path,
        ):
            plots["plot_budget_vs_runtime"] = str(budget_runtime_path.resolve())
        mode_timeline_path = study_dir / "plot_mode_switch_timeline.png"
        mode_switch_series: list[tuple[str, list[tuple[float, float]]]] = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = []
            for row in raw_rows:
                if row["variant_label"] != variant:
                    continue
                generations = row.get("mode_switch_generations", [])
                if not isinstance(generations, list):
                    continue
                for generation in generations:
                    if isinstance(generation, int | float):
                        points.append((float(generation), float(row["seed"])))
            if points:
                mode_switch_series.append((variant, points))
        if _plot_trigger_events(
            title=f"{study.study_name}: mode switches by generation",
            series=mode_switch_series,
            output_path=mode_timeline_path,
        ):
            plots["plot_mode_switch_timeline"] = str(mode_timeline_path.resolve())
        diversity_mode_path = study_dir / "plot_diversity_vs_mode.png"
        diversity_mode_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["time_in_trigger_mode"]),
                    float(row["final_edge_diversity_ratio"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("time_in_trigger_mode"), int | float)
                and isinstance(row.get("final_edge_diversity_ratio"), int | float)
            ]
            if points:
                diversity_mode_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: time in trigger mode vs final edge diversity",
            x_label="time_in_trigger_mode",
            y_label="final_edge_diversity_ratio",
            series=diversity_mode_series,
            output_path=diversity_mode_path,
        ):
            plots["plot_diversity_vs_mode"] = str(diversity_mode_path.resolve())
        collapse_switch_path = study_dir / "plot_collapse_onset_vs_switch.png"
        collapse_switch_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["collapse_onset_generation"]),
                    float(row["first_switch_generation"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("collapse_onset_generation"), int | float)
                and isinstance(row.get("first_switch_generation"), int | float)
            ]
            if points:
                collapse_switch_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: collapse onset vs first switch",
            x_label="collapse_onset_generation",
            y_label="first_switch_generation",
            series=collapse_switch_series,
            output_path=collapse_switch_path,
        ):
            plots["plot_collapse_onset_vs_switch"] = str(collapse_switch_path.resolve())
        regret_path = study_dir / "plot_regret_vs_policy.png"
        if _plot_summary_metric_boxes(
            title=f"{study.study_name}: regret vs oracle fixed policy by variant",
            y_label="regret_vs_oracle_fixed_policy",
            raw_rows=raw_rows,
            metric_name="regret_vs_oracle_fixed_policy",
            output_path=regret_path,
        ):
            plots["plot_regret_vs_policy"] = str(regret_path.resolve())
        budget_regret_path = study_dir / "plot_budget_band_vs_regret.png"
        budget_regret_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["configured_budget"]),
                    float(row["regret_vs_oracle_fixed_policy"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("configured_budget"), int | float)
                and isinstance(row.get("regret_vs_oracle_fixed_policy"), int | float)
            ]
            if points:
                budget_regret_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: configured budget vs regret",
            x_label="configured_budget",
            y_label="regret_vs_oracle_fixed_policy",
            series=budget_regret_series,
            output_path=budget_regret_path,
        ):
            plots["plot_budget_band_vs_regret"] = str(budget_regret_path.resolve())
        budget_vs_current_path = study_dir / "plot_budget_vs_regret.png"
        budget_vs_current_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["configured_budget"]),
                    float(row["regret_vs_current_preferred_profile"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("configured_budget"), int | float)
                and isinstance(row.get("regret_vs_current_preferred_profile"), int | float)
            ]
            if points:
                budget_vs_current_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: configured budget vs regret to current default",
            x_label="configured_budget",
            y_label="regret_vs_current_preferred_profile",
            series=budget_vs_current_series,
            output_path=budget_vs_current_path,
        ):
            plots["plot_budget_vs_regret"] = str(budget_vs_current_path.resolve())
        early_stop_quality_path = study_dir / "plot_early_stop_vs_quality.png"
        early_stop_quality_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["actual_evaluations_used"]),
                    float(row["best_route_distance"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("actual_evaluations_used"), int | float)
                and isinstance(row.get("best_route_distance"), int | float)
            ]
            if points:
                early_stop_quality_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: actual evaluations vs route distance",
            x_label="actual_evaluations_used",
            y_label="best_route_distance",
            series=early_stop_quality_series,
            output_path=early_stop_quality_path,
        ):
            plots["plot_early_stop_vs_quality"] = str(early_stop_quality_path.resolve())
        if any(isinstance(row.get("portfolio_restart_count"), int | float) for row in raw_rows):
            total_budget_path = study_dir / "plot_total_budget_vs_best_of_k.png"
            total_budget_series = []
            for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
                points = [
                    (
                        float(row["configured_budget"]),
                        float(row["best_route_distance"]),
                    )
                    for row in raw_rows
                    if row["variant_label"] == variant
                    and isinstance(row.get("configured_budget"), int | float)
                    and isinstance(row.get("best_route_distance"), int | float)
                ]
                if points:
                    total_budget_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: total configured budget vs best route distance",
                x_label="configured_budget",
                y_label="best_route_distance",
                series=total_budget_series,
                output_path=total_budget_path,
            ):
                plots["plot_total_budget_vs_best_of_k"] = str(total_budget_path.resolve())

            restart_regret_path = study_dir / "plot_restart_count_vs_regret.png"
            restart_regret_series = []
            for variant in sorted({str(row["variant_label"]) for row in summary_rows}):
                points = [
                    (
                        float(row["portfolio_restart_count_mean"]),
                        float(row["regret_vs_canonical_once_mean"]),
                    )
                    for row in summary_rows
                    if str(row["variant_label"]) == variant
                    and isinstance(row.get("portfolio_restart_count_mean"), int | float)
                    and isinstance(row.get("regret_vs_canonical_once_mean"), int | float)
                ]
                if points:
                    restart_regret_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: restart count vs regret to canonical",
                x_label="portfolio_restart_count",
                y_label="regret_vs_canonical_once_mean",
                series=restart_regret_series,
                output_path=restart_regret_path,
            ):
                plots["plot_restart_count_vs_regret"] = str(restart_regret_path.resolve())

            multistart_gap_path = study_dir / "plot_multistart_vs_single_gap.png"
            multistart_gap_series = []
            for variant in sorted({str(row["variant_label"]) for row in summary_rows}):
                points = [
                    (
                        float(row["regret_vs_fast_once_mean"]),
                        float(row["regret_vs_canonical_once_mean"]),
                    )
                    for row in summary_rows
                    if str(row["variant_label"]) == variant
                    and isinstance(row.get("regret_vs_fast_once_mean"), int | float)
                    and isinstance(row.get("regret_vs_canonical_once_mean"), int | float)
                ]
                if points:
                    multistart_gap_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: regret to fast vs regret to canonical",
                x_label="regret_vs_fast_once_mean",
                y_label="regret_vs_canonical_once_mean",
                series=multistart_gap_series,
                output_path=multistart_gap_path,
            ):
                plots["plot_multistart_vs_single_gap"] = str(multistart_gap_path.resolve())
        if any(isinstance(row.get("pilot_budget_fraction"), int | float) for row in raw_rows):
            pilot_fraction_path = study_dir / "plot_pilot_fraction_vs_regret.png"
            pilot_fraction_series = []
            for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
                points = [
                    (
                        float(row["pilot_budget_fraction"]),
                        float(row["regret_vs_canonical_profile"]),
                    )
                    for row in raw_rows
                    if row["variant_label"] == variant
                    and isinstance(row.get("pilot_budget_fraction"), int | float)
                    and isinstance(row.get("regret_vs_canonical_profile"), int | float)
                ]
                if points:
                    pilot_fraction_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: pilot fraction vs regret to canonical",
                x_label="pilot_budget_fraction",
                y_label="regret_vs_canonical_profile",
                series=pilot_fraction_series,
                output_path=pilot_fraction_path,
            ):
                plots["plot_pilot_fraction_vs_regret"] = str(pilot_fraction_path.resolve())
            escalation_budget_path = study_dir / "plot_escalation_rate_vs_budget.png"
            escalation_budget_series = []
            for variant in sorted({str(row["variant_label"]) for row in summary_rows}):
                points = [
                    (
                        float(row["actual_evaluations_used_mean"]),
                        float(row["escalation_rate"]),
                    )
                    for row in summary_rows
                    if str(row["variant_label"]) == variant
                    and isinstance(row.get("actual_evaluations_used_mean"), int | float)
                    and isinstance(row.get("escalation_rate"), int | float)
                ]
                if points:
                    escalation_budget_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: actual evaluations vs escalation rate",
                x_label="actual_evaluations_used_mean",
                y_label="escalation_rate",
                series=escalation_budget_series,
                output_path=escalation_budget_path,
            ):
                plots["plot_escalation_rate_vs_budget"] = str(
                    escalation_budget_path.resolve()
                )
            gate_error_path = study_dir / "plot_false_keep_vs_false_escalate.png"
            gate_error_series = []
            for variant in sorted({str(row["variant_label"]) for row in summary_rows}):
                points = [
                    (
                        float(row["false_keep_rate"]),
                        float(row["false_escalation_rate"]),
                    )
                    for row in summary_rows
                    if str(row["variant_label"]) == variant
                    and isinstance(row.get("false_keep_rate"), int | float)
                    and isinstance(row.get("false_escalation_rate"), int | float)
                ]
                if points:
                    gate_error_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: false keep vs false escalation",
                x_label="false_keep_rate",
                y_label="false_escalation_rate",
                series=gate_error_series,
                output_path=gate_error_path,
            ):
                plots["plot_false_keep_vs_false_escalate"] = str(gate_error_path.resolve())
            actual_eval_quality_path = study_dir / "plot_actual_eval_vs_quality.png"
            actual_eval_quality_series = []
            for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
                points = [
                    (
                        float(
                            row.get(
                                "total_actual_evaluations_used",
                                row.get("actual_evaluations_used"),
                            )
                        ),
                        float(row["best_route_distance"]),
                    )
                    for row in raw_rows
                    if row["variant_label"] == variant
                    and isinstance(
                        row.get(
                            "total_actual_evaluations_used",
                            row.get("actual_evaluations_used"),
                        ),
                        int | float,
                    )
                    and isinstance(row.get("best_route_distance"), int | float)
                ]
                if points:
                    actual_eval_quality_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: total evaluations vs route distance",
                x_label="total_actual_evaluations_used",
                y_label="best_route_distance",
                series=actual_eval_quality_series,
                output_path=actual_eval_quality_path,
            ):
                plots["plot_actual_eval_vs_quality"] = str(actual_eval_quality_path.resolve())
        if case_group_summary_rows:
            grouped_lookup = {
                (
                    str(row.get("case_group", "")),
                    str(row.get("combo_label", "")),
                ): row
                for row in case_group_summary_rows
            }
            rescue_anti_points: dict[str, list[tuple[float, float]]] = {}
            combo_labels = sorted(
                {
                    str(row.get("combo_label", ""))
                    for row in case_group_summary_rows
                    if row.get("case_group") == "rescue_target"
                }
            )
            for combo_label in combo_labels:
                rescue_row = grouped_lookup.get(("rescue_target", combo_label))
                anti_row = grouped_lookup.get(("anti_case", combo_label))
                if not rescue_row or not anti_row:
                    continue
                rescue_regret = rescue_row.get("regret_vs_oracle_fixed_policy_mean")
                anti_damage = anti_row.get("anti_case_damage_mean")
                policy_label = str(
                    rescue_row.get("algorithm_options.adaptive_policy", combo_label)
                )
                if not isinstance(rescue_regret, int | float) or not isinstance(
                    anti_damage,
                    int | float,
                ):
                    continue
                rescue_anti_points.setdefault(policy_label, []).append(
                    (float(rescue_regret), float(anti_damage))
                )
            rescue_anti_series = sorted(rescue_anti_points.items())
            rescue_anti_path = study_dir / "plot_rescue_target_vs_anticase_gap.png"
            if _plot_metric_scatter(
                title=f"{study.study_name}: rescue-target regret vs anti-case damage",
                x_label="rescue_target_regret_vs_oracle",
                y_label="anti_case_damage_vs_decay",
                series=rescue_anti_series,
                output_path=rescue_anti_path,
            ):
                plots["plot_rescue_target_vs_anticase_gap"] = str(
                    rescue_anti_path.resolve()
                )

    if study.problem == "zdt1":
        hypervolume_path = study_dir / "plot_hypervolume.png"
        if _plot_history_lines(
            title=f"{study.study_name}: hypervolume vs generation",
            x_label="Generation",
            y_label="hypervolume",
            series=_history_series(history_rows, "hypervolume"),
            output_path=hypervolume_path,
        ):
            plots["plot_hypervolume"] = str(hypervolume_path.resolve())
        hv_spread_path = study_dir / "plot_hv_vs_spread.png"
        zdt_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["spread"]),
                    float(row["hypervolume"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("spread"), int | float)
                and isinstance(row.get("hypervolume"), int | float)
            ]
            if points:
                zdt_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: spread vs hypervolume",
            x_label="spread",
            y_label="hypervolume",
            series=zdt_series,
            output_path=hv_spread_path,
        ):
            plots["plot_hv_vs_spread"] = str(hv_spread_path.resolve())
        if ranking_detail_rows:
            hv_rank_path = study_dir / "plot_fast_vs_canonical_hv_rank.png"
            hv_rank_series = []
            points = [
                (
                    float(row["fast_rank"]),
                    float(row["canonical_rank"]),
                )
                for row in ranking_detail_rows
                if isinstance(row.get("canonical_rank"), int | float)
                and isinstance(row.get("fast_rank"), int | float)
            ]
            if points:
                hv_rank_series.append(("overall", points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: fast HV rank vs canonical rank",
                x_label="fast_rank",
                y_label="canonical_rank",
                series=hv_rank_series,
                output_path=hv_rank_path,
            ):
                plots["plot_fast_vs_canonical_hv_rank"] = str(hv_rank_path.resolve())
        if triage_workflow_rows:
            topk_budget_path = study_dir / "plot_topk_recall_vs_budget.png"
            topk_budget_series = []
            workflow_points = [
                (
                    float(row["total_actual_evaluations_used"]),
                    float(row["oracle_hit_rate"]),
                )
                for row in triage_workflow_rows
                if str(row.get("scope")) != "case"
                and isinstance(row.get("total_actual_evaluations_used"), int | float)
                and isinstance(row.get("oracle_hit_rate"), int | float)
            ]
            if workflow_points:
                topk_budget_series.append(("overall", workflow_points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: triage recall vs total evaluations",
                x_label="total_actual_evaluations_used",
                y_label="oracle_hit_rate",
                series=topk_budget_series,
                output_path=topk_budget_path,
            ):
                plots["plot_topk_recall_vs_budget"] = str(topk_budget_path.resolve())
            triage_hv_path = study_dir / "plot_triage_cost_vs_hv_regret.png"
            triage_hv_series = []
            hv_regret_points = [
                (
                    float(row["total_actual_evaluations_used"]),
                    float(row["final_regret_vs_full_canonical"]),
                )
                for row in triage_workflow_rows
                if str(row.get("scope")) != "case"
                and isinstance(row.get("total_actual_evaluations_used"), int | float)
                and isinstance(row.get("final_regret_vs_full_canonical"), int | float)
            ]
            if hv_regret_points:
                triage_hv_series.append(("overall", hv_regret_points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: triage cost vs HV regret",
                x_label="total_actual_evaluations_used",
                y_label="final_regret_vs_full_canonical",
                series=triage_hv_series,
                output_path=triage_hv_path,
            ):
                plots["plot_triage_cost_vs_hv_regret"] = str(triage_hv_path.resolve())
            spread_fail_path = study_dir / "plot_spread_safety_failures.png"
            spread_fail_series = []
            fail_points = [
                (
                    float(row["total_actual_evaluations_used"]),
                    float(row.get("spread_safety_fail", 0.0)),
                )
                for row in triage_workflow_rows
                if str(row.get("scope")) != "case"
                and isinstance(row.get("total_actual_evaluations_used"), int | float)
                and isinstance(row.get("spread_safety_fail", 0.0), int | float)
            ]
            if fail_points:
                spread_fail_series.append(("spread_safety_fail", fail_points))
            pareto_fail_points = [
                (
                    float(row["total_actual_evaluations_used"]),
                    float(row.get("pareto_ratio_safety_fail", 0.0)),
                )
                for row in triage_workflow_rows
                if str(row.get("scope")) != "case"
                and isinstance(row.get("total_actual_evaluations_used"), int | float)
                and isinstance(row.get("pareto_ratio_safety_fail", 0.0), int | float)
            ]
            if pareto_fail_points:
                spread_fail_series.append(("pareto_ratio_safety_fail", pareto_fail_points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: safety failures vs total evaluations",
                x_label="total_actual_evaluations_used",
                y_label="safety_fail_rate",
                series=spread_fail_series,
                output_path=spread_fail_path,
            ):
                plots["plot_spread_safety_failures"] = str(spread_fail_path.resolve())
        if qf_pair_rows:
            qf_config = _qf_tolerance_config(study) or {}
            qf_plot_suffix = (
                str(qf_config.get("plot_name_suffix") or "").strip()
                if isinstance(qf_config.get("plot_name_suffix"), str)
                else ""
            )
            hv_loss_distribution_path = study_dir / _plot_file_name(
                "plot_q_vs_f_hv_loss_distribution.png",
                qf_plot_suffix,
            )
            hv_loss_values = [[float(row["hv_loss_pct"]) for row in qf_pair_rows]]
            if _plot_box_comparison(
                title=f"{study.study_name}: Q vs F hypervolume loss",
                x_labels=["overall"],
                grouped_values=hv_loss_values,
                y_label="hv_loss_pct",
                output_path=hv_loss_distribution_path,
            ):
                plots["plot_q_vs_f_hv_loss_distribution"] = str(
                    hv_loss_distribution_path.resolve()
                )

            tolerance_accept_path = study_dir / _plot_file_name(
                "plot_tolerance_accept_rate.png",
                qf_plot_suffix,
            )
            tolerance_accept_series = []
            tolerance_points = [
                (
                    float(row["tolerance_bin_pct"]),
                    float(row["acceptable_rate"]),
                )
                for row in tolerance_rows
                if isinstance(row.get("tolerance_bin_pct"), int | float)
                and isinstance(row.get("acceptable_rate"), int | float)
            ]
            if tolerance_points:
                tolerance_accept_series.append(("safety_gated", tolerance_points))
            hv_only_points = [
                (
                    float(row["tolerance_bin_pct"]),
                    float(row["hv_only_accept_rate"]),
                )
                for row in tolerance_rows
                if isinstance(row.get("tolerance_bin_pct"), int | float)
                and isinstance(row.get("hv_only_accept_rate"), int | float)
            ]
            if hv_only_points:
                tolerance_accept_series.append(("hv_only", hv_only_points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: tolerance bin vs F acceptable rate",
                x_label="tolerance_bin_pct",
                y_label="acceptable_rate",
                series=tolerance_accept_series,
                output_path=tolerance_accept_path,
            ):
                plots["plot_tolerance_accept_rate"] = str(tolerance_accept_path.resolve())

            tolerance_fail_path = study_dir / _plot_file_name(
                "plot_spread_safety_failures.png",
                qf_plot_suffix,
            )
            tolerance_fail_series = []
            pareto_fail_points = [
                (
                    float(row["tolerance_bin_pct"]),
                    float(row["pareto_ratio_fail_rate"]),
                )
                for row in tolerance_rows
                if isinstance(row.get("tolerance_bin_pct"), int | float)
                and isinstance(row.get("pareto_ratio_fail_rate"), int | float)
            ]
            if pareto_fail_points:
                tolerance_fail_series.append(("pareto_ratio_fail_rate", pareto_fail_points))
            spread_fail_points = [
                (
                    float(row["tolerance_bin_pct"]),
                    float(row["spread_fail_rate"]),
                )
                for row in tolerance_rows
                if isinstance(row.get("tolerance_bin_pct"), int | float)
                and isinstance(row.get("spread_fail_rate"), int | float)
            ]
            if spread_fail_points:
                tolerance_fail_series.append(("spread_fail_rate", spread_fail_points))
            joint_fail_points = [
                (
                    float(row["tolerance_bin_pct"]),
                    float(row["joint_safety_fail_rate"]),
                )
                for row in tolerance_rows
                if isinstance(row.get("tolerance_bin_pct"), int | float)
                and isinstance(row.get("joint_safety_fail_rate"), int | float)
            ]
            if joint_fail_points:
                tolerance_fail_series.append(("joint_safety_fail_rate", joint_fail_points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: safety failure rate by HV tolerance",
                x_label="tolerance_bin_pct",
                y_label="fail_rate",
                series=tolerance_fail_series,
                output_path=tolerance_fail_path,
            ):
                plots["plot_spread_safety_failures"] = str(tolerance_fail_path.resolve())

            savings_hv_path = study_dir / _plot_file_name(
                "plot_budget_savings_vs_hv_loss.png",
                qf_plot_suffix,
            )
            savings_hv_series = []
            savings_hv_points = [
                (
                    float(row["actual_eval_savings_pct"]),
                    float(row["hv_loss_pct"]),
                )
                for row in qf_pair_rows
                if isinstance(row.get("actual_eval_savings_pct"), int | float)
                and isinstance(row.get("hv_loss_pct"), int | float)
            ]
            if savings_hv_points:
                savings_hv_series.append(("overall", savings_hv_points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: budget savings vs HV loss",
                x_label="actual_eval_savings_pct",
                y_label="hv_loss_pct",
                series=savings_hv_series,
                output_path=savings_hv_path,
            ):
                plots["plot_budget_savings_vs_hv_loss"] = str(savings_hv_path.resolve())
        if any("study_variant" in row for row in tail_risk_summary_rows):
            hardening_config = _zdt1_fast_hardening_config(study) or {}
            overall_tail_rows = [
                row
                for row in tail_risk_summary_rows
                if str(row.get("scope") or "overall") == "overall"
                and isinstance(row.get("mean_loss_pct"), int | float)
            ]
            if overall_tail_rows:
                hv_loss_path = study_dir / _stress_plot_file_name(
                    hardening_config,
                    "candidate_hv_loss",
                    "plot_currentF_vs_candidate_hv_loss.png",
                )
                hv_loss_labels = [str(row.get("study_variant") or "") for row in overall_tail_rows]
                hv_loss_values = [float(row["mean_loss_pct"]) for row in overall_tail_rows]
                if _plot_categorical_bars(
                    title=f"{study.study_name}: mean HV loss by candidate",
                    x_labels=hv_loss_labels,
                    values=hv_loss_values,
                    y_label="mean_hv_loss_pct",
                    output_path=hv_loss_path,
                ):
                    plots["plot_currentF_vs_candidate_hv_loss"] = str(hv_loss_path.resolve())

                spread_fail_path = study_dir / _stress_plot_file_name(
                    hardening_config,
                    "spread_safety_failures",
                    "plot_spread_safety_failures.png",
                )
                spread_fail_labels = [
                    str(row.get("study_variant") or "")
                    for row in overall_tail_rows
                    if isinstance(row.get("joint_safety_fail_rate"), int | float)
                ]
                spread_fail_values = [
                    float(row["joint_safety_fail_rate"])
                    for row in overall_tail_rows
                    if isinstance(row.get("joint_safety_fail_rate"), int | float)
                ]
                if _plot_categorical_bars(
                    title=f"{study.study_name}: joint safety fail rate by candidate",
                    x_labels=spread_fail_labels,
                    values=spread_fail_values,
                    y_label="joint_safety_fail_rate",
                    output_path=spread_fail_path,
                ):
                    plots["plot_spread_safety_failures"] = str(spread_fail_path.resolve())

                budget_hv_loss_path = study_dir / _stress_plot_file_name(
                    hardening_config,
                    "budget_vs_hv_loss",
                    "plot_budget_vs_hv_loss.png",
                )
                budget_hv_series = [
                    (
                        "candidate",
                        [
                            (
                                float(row["configured_budget_mean"]),
                                float(row["mean_loss_pct"]),
                            )
                            for row in overall_tail_rows
                            if isinstance(row.get("configured_budget_mean"), int | float)
                            and isinstance(row.get("mean_loss_pct"), int | float)
                        ],
                    )
                ]
                budget_hv_series = [
                    (label, points) for label, points in budget_hv_series if points
                ]
                if _plot_metric_scatter(
                    title=f"{study.study_name}: configured budget vs mean HV loss",
                    x_label="configured_budget_mean",
                    y_label="mean_hv_loss_pct",
                    series=budget_hv_series,
                    output_path=budget_hv_loss_path,
                ):
                    plots["plot_budget_vs_hv_loss"] = str(budget_hv_loss_path.resolve())

                spread_tail_path = study_dir / _stress_plot_file_name(
                    hardening_config,
                    "population_generation_vs_spread_tail",
                    "plot_population_generation_vs_spread_tail.png",
                )
                spread_tail_series = []
                spread_tail_points = [
                    (
                        float(row["population_size"]),
                        float(row["spread_delta_p90"]),
                    )
                    for row in overall_tail_rows
                    if isinstance(row.get("population_size"), int | float)
                    and isinstance(row.get("spread_delta_p90"), int | float)
                ]
                if spread_tail_points:
                    spread_tail_series.append(("spread_delta_p90", spread_tail_points))
                spread_tail_p95_points = [
                    (
                        float(row["population_size"]),
                        float(row["spread_delta_p95"]),
                    )
                    for row in overall_tail_rows
                    if isinstance(row.get("population_size"), int | float)
                    and isinstance(row.get("spread_delta_p95"), int | float)
                ]
                if spread_tail_p95_points:
                    spread_tail_series.append(("spread_delta_p95", spread_tail_p95_points))
                if _plot_metric_scatter(
                    title=f"{study.study_name}: population size vs spread tail",
                    x_label="population_size",
                    y_label="spread_delta_tail",
                    series=spread_tail_series,
                    output_path=spread_tail_path,
                ):
                    plots["plot_population_generation_vs_spread_tail"] = str(
                        spread_tail_path.resolve()
                    )

                timing_joint_path = study_dir / _stress_plot_file_name(
                    hardening_config,
                    "timing_vs_joint_fail_rate",
                    "plot_timing_vs_joint_fail_rate.png",
                )
                timing_joint_labels = [
                    str(row.get("study_variant") or "")
                    for row in overall_tail_rows
                    if isinstance(row.get("joint_safety_fail_rate"), int | float)
                ]
                timing_joint_values = [
                    float(row["joint_safety_fail_rate"])
                    for row in overall_tail_rows
                    if isinstance(row.get("joint_safety_fail_rate"), int | float)
                ]
                if _plot_categorical_bars(
                    title=f"{study.study_name}: joint fail rate by timing candidate",
                    x_labels=timing_joint_labels,
                    values=timing_joint_values,
                    y_label="joint_safety_fail_rate",
                    output_path=timing_joint_path,
                ):
                    plots["plot_timing_vs_joint_fail_rate"] = str(timing_joint_path.resolve())
        budget_hv_path = study_dir / "plot_budget_vs_hv.png"
        budget_series = []
        for threshold_label in sorted(
            {
                str(row.get("algorithm_options.diversity_threshold", row["variant_label"]))
                for row in raw_rows
                if isinstance(row.get("configured_budget"), int | float)
                and isinstance(row.get("hypervolume"), int | float)
            }
        ):
            points = [
                (float(row["configured_budget"]), float(row["hypervolume"]))
                for row in raw_rows
                if str(
                    row.get("algorithm_options.diversity_threshold", row["variant_label"])
                )
                == threshold_label
                and isinstance(row.get("configured_budget"), int | float)
                and isinstance(row.get("hypervolume"), int | float)
            ]
            if points:
                budget_series.append((f"threshold={threshold_label}", points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: configured budget vs hypervolume",
            x_label="configured_budget",
            y_label="hypervolume",
            series=budget_series,
            output_path=budget_hv_path,
        ):
            plots["plot_budget_vs_hv"] = str(budget_hv_path.resolve())
        budget_spread_path = study_dir / "plot_budget_vs_spread.png"
        budget_spread_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["configured_budget"]),
                    float(row["spread"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("configured_budget"), int | float)
                and isinstance(row.get("spread"), int | float)
            ]
            if points:
                budget_spread_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: configured budget vs spread",
            x_label="configured_budget",
            y_label="spread",
            series=budget_spread_series,
            output_path=budget_spread_path,
        ):
            plots["plot_budget_vs_spread"] = str(budget_spread_path.resolve())
        early_stop_hv_path = study_dir / "plot_early_stop_vs_hv.png"
        early_stop_hv_series = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = [
                (
                    float(row["actual_evaluations_used"]),
                    float(row["hypervolume"]),
                )
                for row in raw_rows
                if row["variant_label"] == variant
                and isinstance(row.get("actual_evaluations_used"), int | float)
                and isinstance(row.get("hypervolume"), int | float)
            ]
            if points:
                early_stop_hv_series.append((variant, points))
        if _plot_metric_scatter(
            title=f"{study.study_name}: actual evaluations vs hypervolume",
            x_label="actual_evaluations_used",
            y_label="hypervolume",
            series=early_stop_hv_series,
            output_path=early_stop_hv_path,
        ):
            plots["plot_early_stop_vs_hv"] = str(early_stop_hv_path.resolve())
        if any(isinstance(row.get("portfolio_restart_count"), int | float) for row in raw_rows):
            merged_budget_path = study_dir / "plot_total_budget_vs_merged_hv.png"
            merged_budget_series = []
            for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
                points = [
                    (
                        float(row["configured_budget"]),
                        float(row["hypervolume"]),
                    )
                    for row in raw_rows
                    if row["variant_label"] == variant
                    and isinstance(row.get("configured_budget"), int | float)
                    and isinstance(row.get("hypervolume"), int | float)
                ]
                if points:
                    merged_budget_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: total configured budget vs merged archive hypervolume",
                x_label="configured_budget",
                y_label="hypervolume",
                series=merged_budget_series,
                output_path=merged_budget_path,
            ):
                plots["plot_total_budget_vs_merged_hv"] = str(merged_budget_path.resolve())

            restart_hv_path = study_dir / "plot_restart_count_vs_hv.png"
            restart_hv_series = []
            for variant in sorted({str(row["variant_label"]) for row in summary_rows}):
                points = [
                    (
                        float(row["portfolio_restart_count_mean"]),
                        float(row["hypervolume_mean"]),
                    )
                    for row in summary_rows
                    if str(row["variant_label"]) == variant
                    and isinstance(row.get("portfolio_restart_count_mean"), int | float)
                    and isinstance(row.get("hypervolume_mean"), int | float)
                ]
                if points:
                    restart_hv_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: restart count vs hypervolume",
                x_label="portfolio_restart_count",
                y_label="hypervolume_mean",
                series=restart_hv_series,
                output_path=restart_hv_path,
            ):
                plots["plot_restart_count_vs_hv"] = str(restart_hv_path.resolve())

            merged_single_path = study_dir / "plot_merged_archive_vs_single_run.png"
            merged_single_series = []
            for variant in sorted({str(row["variant_label"]) for row in summary_rows}):
                points = [
                    (
                        float(row["regret_vs_fast_once_mean"]),
                        float(row["hypervolume_mean"]),
                    )
                    for row in summary_rows
                    if str(row["variant_label"]) == variant
                    and isinstance(row.get("regret_vs_fast_once_mean"), int | float)
                    and isinstance(row.get("hypervolume_mean"), int | float)
                ]
                if points:
                    merged_single_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: regret to fast vs merged archive hypervolume",
                x_label="regret_vs_fast_once_mean",
                y_label="hypervolume_mean",
                series=merged_single_series,
                output_path=merged_single_path,
            ):
                plots["plot_merged_archive_vs_single_run"] = str(
                    merged_single_path.resolve()
                )
        if any(isinstance(row.get("pilot_budget_fraction"), int | float) for row in raw_rows):
            pilot_fraction_hv_path = study_dir / "plot_pilot_fraction_vs_hv.png"
            pilot_fraction_hv_series = []
            for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
                points = [
                    (
                        float(row["pilot_budget_fraction"]),
                        float(row["hypervolume"]),
                    )
                    for row in raw_rows
                    if row["variant_label"] == variant
                    and isinstance(row.get("pilot_budget_fraction"), int | float)
                    and isinstance(row.get("hypervolume"), int | float)
                ]
                if points:
                    pilot_fraction_hv_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: pilot fraction vs hypervolume",
                x_label="pilot_budget_fraction",
                y_label="hypervolume",
                series=pilot_fraction_hv_series,
                output_path=pilot_fraction_hv_path,
            ):
                plots["plot_pilot_fraction_vs_hv"] = str(pilot_fraction_hv_path.resolve())
            escalation_hv_path = study_dir / "plot_escalation_rate_vs_hv.png"
            escalation_hv_series = []
            for variant in sorted({str(row["variant_label"]) for row in summary_rows}):
                points = [
                    (
                        float(row["escalation_rate"]),
                        float(row["hypervolume_mean"]),
                    )
                    for row in summary_rows
                    if str(row["variant_label"]) == variant
                    and isinstance(row.get("escalation_rate"), int | float)
                    and isinstance(row.get("hypervolume_mean"), int | float)
                ]
                if points:
                    escalation_hv_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: escalation rate vs hypervolume",
                x_label="escalation_rate",
                y_label="hypervolume_mean",
                series=escalation_hv_series,
                output_path=escalation_hv_path,
            ):
                plots["plot_escalation_rate_vs_hv"] = str(escalation_hv_path.resolve())
            actual_eval_hv_path = study_dir / "plot_actual_eval_vs_hv.png"
            actual_eval_hv_series = []
            for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
                points = [
                    (
                        float(
                            row.get(
                                "total_actual_evaluations_used",
                                row.get("actual_evaluations_used"),
                            )
                        ),
                        float(row["hypervolume"]),
                    )
                    for row in raw_rows
                    if row["variant_label"] == variant
                    and isinstance(
                        row.get(
                            "total_actual_evaluations_used",
                            row.get("actual_evaluations_used"),
                        ),
                        int | float,
                    )
                    and isinstance(row.get("hypervolume"), int | float)
                ]
                if points:
                    actual_eval_hv_series.append((variant, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: total evaluations vs hypervolume",
                x_label="total_actual_evaluations_used",
                y_label="hypervolume",
                series=actual_eval_hv_series,
                output_path=actual_eval_hv_path,
            ):
                plots["plot_actual_eval_vs_hv"] = str(actual_eval_hv_path.resolve())
        def _adaptive_only(row: dict[str, Any]) -> bool:
            return row.get("algorithm_options.adaptive_policy") == "low_diversity_injection"

        threshold_hv_path = study_dir / "plot_threshold_vs_hv.png"
        if _plot_summary_axis_vs_metric(
            summary_rows,
            axis_column="algorithm_options.diversity_threshold",
            metric_column="hypervolume_mean",
            title=f"{study.study_name}: threshold vs hypervolume",
            x_label="diversity_threshold",
            y_label="hypervolume",
            output_path=threshold_hv_path,
            include_row=_adaptive_only,
        ):
            plots["plot_threshold_vs_hv"] = str(threshold_hv_path.resolve())
        refresh_hv_path = study_dir / "plot_refresh_vs_hv.png"
        if _plot_summary_axis_vs_metric(
            summary_rows,
            axis_column="algorithm_options.refresh_fraction",
            metric_column="hypervolume_mean",
            title=f"{study.study_name}: refresh fraction vs hypervolume",
            x_label="refresh_fraction",
            y_label="hypervolume",
            output_path=refresh_hv_path,
            include_row=_adaptive_only,
        ):
            plots["plot_refresh_vs_hv"] = str(refresh_hv_path.resolve())
        cooldown_hv_path = study_dir / "plot_cooldown_vs_hv.png"
        if _plot_summary_axis_vs_metric(
            summary_rows,
            axis_column="algorithm_options.adaptation_cooldown",
            metric_column="hypervolume_mean",
            title=f"{study.study_name}: cooldown vs hypervolume",
            x_label="adaptation_cooldown",
            y_label="hypervolume",
            output_path=cooldown_hv_path,
            include_row=_adaptive_only,
        ):
            plots["plot_cooldown_vs_hv"] = str(cooldown_hv_path.resolve())
    if len(study.sweep) == 1:
        axis_name = next(iter(study.sweep))
        sweep_points = []
        for row in summary_rows:
            axis_value = row.get(axis_name)
            mean_value = row.get(f"{study.primary_metric}_mean")
            if isinstance(axis_value, int | float) and isinstance(mean_value, int | float):
                sweep_points.append(
                    (
                        float(axis_value),
                        float(mean_value),
                        (
                            float(row["primary_metric_ci_low"])
                            if isinstance(row.get("primary_metric_ci_low"), int | float)
                            else None
                        ),
                        (
                            float(row["primary_metric_ci_high"])
                            if isinstance(row.get("primary_metric_ci_high"), int | float)
                            else None
                        ),
                    )
                )
        parameter_sweep_path = study_dir / "plot_parameter_sweep.png"
        if _plot_parameter_sweep(
            title=f"{study.study_name}: {study.primary_metric} by {axis_name}",
            x_label=axis_name,
            y_label=study.primary_metric,
            points=sweep_points,
            output_path=parameter_sweep_path,
        ):
            plots["plot_parameter_sweep"] = str(parameter_sweep_path.resolve())

    if seed_budget_rows:
        seed_budget_config = _seed_budget_config(study) or {}
        decision_tolerance = float(seed_budget_config.get("decision_tolerance_pct", 0.5))
        decision_accept_column = str(decision_tolerance).replace(".", "_")
        accept_column = f"accept_rate_at_{decision_accept_column}_pct"

        if study.problem == "tsp":
            ci_path = study_dir / "plot_seed_count_vs_loss_ci.png"
            ci_series = []
            for case_group in ("overall", "rescue_target", "anti_case"):
                points = [
                    (float(row["seed_count"]), float(row["ci_width_pct"]))
                    for row in seed_budget_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("ci_width_pct"), int | float)
                ]
                if points:
                    ci_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: seed count vs TSP loss CI width",
                x_label="seed_count",
                y_label="ci_width_pct",
                series=ci_series,
                output_path=ci_path,
            ):
                plots["plot_seed_count_vs_loss_ci"] = str(ci_path.resolve())

            flip_path = study_dir / "plot_seed_count_vs_decision_flip.png"
            flip_series = []
            for case_group in ("overall", "rescue_target", "anti_case"):
                points = [
                    (float(row["seed_count"]), float(row["decision_flip_rate_vs_full"]))
                    for row in seed_budget_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("decision_flip_rate_vs_full"), int | float)
                ]
                if points:
                    flip_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: seed count vs TSP decision flips",
                x_label="seed_count",
                y_label="decision_flip_rate_vs_full",
                series=flip_series,
                output_path=flip_path,
            ):
                plots["plot_seed_count_vs_decision_flip"] = str(flip_path.resolve())

            accept_path = study_dir / "plot_seed_count_vs_accept_rate.png"
            accept_series = []
            for case_group in ("overall", "rescue_target", "anti_case"):
                points = [
                    (float(row["seed_count"]), float(row[accept_column]))
                    for row in seed_budget_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get(accept_column), int | float)
                ]
                if points:
                    accept_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: seed count vs TSP acceptable rate",
                x_label="seed_count",
                y_label=accept_column,
                series=accept_series,
                output_path=accept_path,
            ):
                plots["plot_seed_count_vs_accept_rate"] = str(accept_path.resolve())

            split_path = study_dir / "plot_rescue_vs_anticase_seed_stability.png"
            split_series = []
            for case_group in ("rescue_target", "anti_case"):
                points = [
                    (float(row["seed_count"]), float(row["decision_stability_score"]))
                    for row in seed_budget_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("decision_stability_score"), int | float)
                ]
                if points:
                    split_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: rescue-target vs anti-case seed stability",
                x_label="seed_count",
                y_label="decision_stability_score",
                series=split_series,
                output_path=split_path,
            ):
                plots["plot_rescue_vs_anticase_seed_stability"] = str(split_path.resolve())

        elif study.problem == "zdt1":
            hv_ci_path = study_dir / "plot_seed_count_vs_hv_ci.png"
            hv_ci_series = [
                (
                    "overall",
                    [
                        (float(row["seed_count"]), float(row["ci_width_pct"]))
                        for row in seed_budget_rows
                        if isinstance(row.get("seed_count"), int | float)
                        and isinstance(row.get("ci_width_pct"), int | float)
                    ],
                )
            ]
            hv_ci_series = [(label, points) for label, points in hv_ci_series if points]
            if _plot_metric_scatter(
                title=f"{study.study_name}: seed count vs HV CI width",
                x_label="seed_count",
                y_label="ci_width_pct",
                series=hv_ci_series,
                output_path=hv_ci_path,
            ):
                plots["plot_seed_count_vs_hv_ci"] = str(hv_ci_path.resolve())

            safety_path = study_dir / "plot_seed_count_vs_safety_fail_rate.png"
            safety_series = [
                (
                    "overall",
                    [
                        (float(row["seed_count"]), float(row["joint_safety_fail_rate"]))
                        for row in seed_budget_rows
                        if isinstance(row.get("seed_count"), int | float)
                        and isinstance(row.get("joint_safety_fail_rate"), int | float)
                    ],
                )
            ]
            safety_series = [(label, points) for label, points in safety_series if points]
            if _plot_metric_scatter(
                title=f"{study.study_name}: seed count vs ZDT1 safety fail rate",
                x_label="seed_count",
                y_label="joint_safety_fail_rate",
                series=safety_series,
                output_path=safety_path,
            ):
                plots["plot_seed_count_vs_safety_fail_rate"] = str(safety_path.resolve())

            flip_path = study_dir / "plot_seed_count_vs_decision_flip.png"
            flip_series = [
                (
                    "overall",
                    [
                        (float(row["seed_count"]), float(row["decision_flip_rate_vs_full"]))
                        for row in seed_budget_rows
                        if isinstance(row.get("seed_count"), int | float)
                        and isinstance(row.get("decision_flip_rate_vs_full"), int | float)
                    ],
                )
            ]
            flip_series = [(label, points) for label, points in flip_series if points]
            if _plot_metric_scatter(
                title=f"{study.study_name}: seed count vs ZDT1 decision flips",
                x_label="seed_count",
                y_label="decision_flip_rate_vs_full",
                series=flip_series,
                output_path=flip_path,
            ):
                plots["plot_seed_count_vs_decision_flip"] = str(flip_path.resolve())

        elif study.problem == "knapsack":
            stability_path = study_dir / "plot_seed_count_vs_repair_note_stability.png"
            stability_series = []
            for case_group in sorted(
                {str(row.get("case_group") or "overall") for row in seed_budget_rows}
            ):
                points = [
                    (float(row["seed_count"]), float(row["decision_stability_score"]))
                    for row in seed_budget_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("decision_stability_score"), int | float)
                ]
                if points:
                    stability_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: seed count vs repair note stability",
                x_label="seed_count",
                y_label="decision_stability_score",
                series=stability_series,
                output_path=stability_path,
            ):
                plots["plot_seed_count_vs_repair_note_stability"] = str(
                    stability_path.resolve()
                )

        elif study.problem == "onemax":
            stability_path = study_dir / "plot_seed_count_vs_control_stability.png"
            stability_series = [
                (
                    "overall",
                    [
                        (float(row["seed_count"]), float(row["decision_stability_score"]))
                        for row in seed_budget_rows
                        if isinstance(row.get("seed_count"), int | float)
                        and isinstance(row.get("decision_stability_score"), int | float)
                    ],
                )
            ]
            stability_series = [(label, points) for label, points in stability_series if points]
            if _plot_metric_scatter(
                title=f"{study.study_name}: seed count vs control stability",
                x_label="seed_count",
                y_label="decision_stability_score",
                series=stability_series,
                output_path=stability_path,
            ):
                plots["plot_seed_count_vs_control_stability"] = str(stability_path.resolve())

    if sequential_decision_rows:
        if study.problem == "tsp":
            ci_path = study_dir / "plot_seed_stage_vs_ci_width.png"
            ci_series = []
            for label in sorted(
                {
                    f"{row.get('mode')}|{row.get('case_group')}"
                    for row in sequential_decision_rows
                    if isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("bootstrap_ci_width"), int | float)
                }
            ):
                mode, case_group = label.split("|", 1)
                points = [
                    (float(row["seed_count"]), float(row["bootstrap_ci_width"]))
                    for row in sequential_decision_rows
                    if str(row.get("mode")) == mode
                    and str(row.get("case_group")) == case_group
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("bootstrap_ci_width"), int | float)
                ]
                if points:
                    ci_series.append((label, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: sequential seed stage vs CI width",
                x_label="seed_count",
                y_label="bootstrap_ci_width",
                series=ci_series,
                output_path=ci_path,
            ):
                plots["plot_seed_stage_vs_ci_width"] = str(ci_path.resolve())

            flip_path = study_dir / "plot_seed_stage_vs_decision_flip.png"
            flip_series = []
            for label in sorted(
                {
                    f"{row.get('mode')}|{row.get('case_group')}"
                    for row in sequential_decision_rows
                    if isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("decision_flip_rate_vs_full"), int | float)
                }
            ):
                mode, case_group = label.split("|", 1)
                points = [
                    (float(row["seed_count"]), float(row["decision_flip_rate_vs_full"]))
                    for row in sequential_decision_rows
                    if str(row.get("mode")) == mode
                    and str(row.get("case_group")) == case_group
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("decision_flip_rate_vs_full"), int | float)
                ]
                if points:
                    flip_series.append((label, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: sequential seed stage vs decision flip",
                x_label="seed_count",
                y_label="decision_flip_rate_vs_full",
                series=flip_series,
                output_path=flip_path,
            ):
                plots["plot_seed_stage_vs_decision_flip"] = str(flip_path.resolve())

            escalation_path = study_dir / "plot_rescue_vs_anticase_escalation_rate.png"
            escalation_series = []
            for case_group in ("rescue_target", "anti_case"):
                points = [
                    (float(row["seed_count"]), float(row["should_escalate"]))
                    for row in sequential_decision_rows
                    if str(row.get("case_group")) == case_group
                    and str(row.get("mode")) == "quality_sensitive"
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("should_escalate"), int | float)
                ]
                if points:
                    escalation_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: rescue-target vs anti-case escalation",
                x_label="seed_count",
                y_label="should_escalate",
                series=escalation_series,
                output_path=escalation_path,
            ):
                plots["plot_rescue_vs_anticase_escalation_rate"] = str(
                    escalation_path.resolve()
                )

            cost_path = study_dir / "plot_cost_savings_vs_false_decision.png"
            cost_series = []
            for mode in sorted(
                {
                    str(row.get("mode"))
                    for row in sequential_decision_rows
                    if isinstance(row.get("actual_eval_savings_vs_full_pct"), int | float)
                    and isinstance(row.get("decision_flip_rate_vs_full"), int | float)
                }
            ):
                points = [
                    (
                        float(row["actual_eval_savings_vs_full_pct"]),
                        float(row["decision_flip_rate_vs_full"]),
                    )
                    for row in sequential_decision_rows
                    if str(row.get("mode")) == mode
                    and isinstance(row.get("actual_eval_savings_vs_full_pct"), int | float)
                    and isinstance(row.get("decision_flip_rate_vs_full"), int | float)
                ]
                if points:
                    cost_series.append((mode, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: cost savings vs false decision proxy",
                x_label="actual_eval_savings_vs_full_pct",
                y_label="decision_flip_rate_vs_full",
                series=cost_series,
                output_path=cost_path,
            ):
                plots["plot_cost_savings_vs_false_decision"] = str(cost_path.resolve())

        elif study.problem == "zdt1":
            hv_ci_path = study_dir / "plot_seed_stage_vs_hv_ci.png"
            hv_ci_series = []
            for mode in sorted(
                {
                    str(row.get("mode"))
                    for row in sequential_decision_rows
                    if isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("bootstrap_ci_width"), int | float)
                }
            ):
                points = [
                    (float(row["seed_count"]), float(row["bootstrap_ci_width"]))
                    for row in sequential_decision_rows
                    if str(row.get("mode")) == mode
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("bootstrap_ci_width"), int | float)
                ]
                if points:
                    hv_ci_series.append((mode, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: sequential seed stage vs HV CI width",
                x_label="seed_count",
                y_label="bootstrap_ci_width",
                series=hv_ci_series,
                output_path=hv_ci_path,
            ):
                plots["plot_seed_stage_vs_hv_ci"] = str(hv_ci_path.resolve())

            safety_path = study_dir / "plot_seed_stage_vs_safety_fail_rate.png"
            safety_series = []
            for mode in sorted(
                {
                    str(row.get("mode"))
                    for row in sequential_decision_rows
                    if isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("joint_safety_fail_rate"), int | float)
                }
            ):
                points = [
                    (float(row["seed_count"]), float(row["joint_safety_fail_rate"]))
                    for row in sequential_decision_rows
                    if str(row.get("mode")) == mode
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("joint_safety_fail_rate"), int | float)
                ]
                if points:
                    safety_series.append((mode, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: sequential seed stage vs safety fail rate",
                x_label="seed_count",
                y_label="joint_safety_fail_rate",
                series=safety_series,
                output_path=safety_path,
            ):
                plots["plot_seed_stage_vs_safety_fail_rate"] = str(safety_path.resolve())

            flip_path = study_dir / "plot_seed_stage_vs_decision_flip.png"
            flip_series = []
            for mode in sorted(
                {
                    str(row.get("mode"))
                    for row in sequential_decision_rows
                    if isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("decision_flip_rate_vs_full"), int | float)
                }
            ):
                points = [
                    (float(row["seed_count"]), float(row["decision_flip_rate_vs_full"]))
                    for row in sequential_decision_rows
                    if str(row.get("mode")) == mode
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("decision_flip_rate_vs_full"), int | float)
                ]
                if points:
                    flip_series.append((mode, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: sequential seed stage vs decision flip",
                x_label="seed_count",
                y_label="decision_flip_rate_vs_full",
                series=flip_series,
                output_path=flip_path,
            ):
                plots["plot_seed_stage_vs_decision_flip"] = str(flip_path.resolve())

        elif study.problem == "knapsack":
            stability_path = study_dir / "plot_seed_stage_vs_repair_note_stability.png"
            stability_series = []
            for case_group in sorted(
                {str(row.get("case_group") or "overall") for row in sequential_decision_rows}
            ):
                points = [
                    (float(row["seed_count"]), float(row["decision_stability_score"]))
                    for row in sequential_decision_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("seed_count"), int | float)
                    and isinstance(row.get("decision_stability_score"), int | float)
                ]
                if points:
                    stability_series.append((case_group, points))
            if _plot_metric_scatter(
                title=f"{study.study_name}: sequential seed stage vs repair-note stability",
                x_label="seed_count",
                y_label="decision_stability_score",
                series=stability_series,
                output_path=stability_path,
            ):
                plots["plot_seed_stage_vs_repair_note_stability"] = str(
                    stability_path.resolve()
                )

    if zdt1_spread_validation_rows:
        overall_validation_rows = [
            row
            for row in zdt1_spread_validation_rows
            if str(row.get("scope") or "") == "overall"
        ]
        if overall_validation_rows:
            candidate_compare_path = study_dir / "plot_spread_candidate_vs_currentF.png"
            compare_labels = [
                str(row.get("study_variant") or "")
                for row in overall_validation_rows
                if isinstance(row.get("spread_fail_rate"), int | float)
            ]
            compare_values = [
                float(row["spread_fail_rate"])
                for row in overall_validation_rows
                if isinstance(row.get("spread_fail_rate"), int | float)
            ]
            if _plot_categorical_bars(
                title=f"{study.study_name}: spread fail rate by fast candidate",
                x_labels=compare_labels,
                values=compare_values,
                y_label="spread_fail_rate",
                output_path=candidate_compare_path,
            ):
                plots["plot_spread_candidate_vs_currentF"] = str(
                    candidate_compare_path.resolve()
                )

            hv_tradeoff_path = study_dir / "plot_hv_preservation_vs_spread_gain.png"
            hv_tradeoff_series = [
                (
                    "candidate",
                    [
                        (
                            float(row["spread_tail_reduction_score"]),
                            float(row["hv_preservation_score"]),
                        )
                        for row in zdt1_spread_validation_rows
                        if str(row.get("study_variant") or "") != "current_fast"
                        and isinstance(row.get("spread_tail_reduction_score"), int | float)
                        and isinstance(row.get("hv_preservation_score"), int | float)
                    ],
                )
            ]
            hv_tradeoff_series = [
                (label, points) for label, points in hv_tradeoff_series if points
            ]
            if _plot_metric_scatter(
                title=f"{study.study_name}: HV preservation vs spread gain",
                x_label="spread_tail_reduction_score",
                y_label="hv_preservation_score",
                series=hv_tradeoff_series,
                output_path=hv_tradeoff_path,
            ):
                plots["plot_hv_preservation_vs_spread_gain"] = str(
                    hv_tradeoff_path.resolve()
                )

        slice_validation_rows = [
            row
            for row in zdt1_spread_validation_rows
            if str(row.get("scope") or "") == "slice"
        ]
        if slice_validation_rows:
            spread_tail_path = study_dir / "plot_spread_tail_validation.png"
            spread_tail_labels = [
                f"{row.get('slice')}:{row.get('study_variant')}"
                for row in slice_validation_rows
                if isinstance(row.get("spread_delta_p90"), int | float)
            ]
            spread_tail_values = [
                float(row["spread_delta_p90"])
                for row in slice_validation_rows
                if isinstance(row.get("spread_delta_p90"), int | float)
            ]
            if _plot_categorical_bars(
                title=f"{study.study_name}: slice-level spread tail validation",
                x_labels=spread_tail_labels,
                values=spread_tail_values,
                y_label="spread_delta_p90",
                output_path=spread_tail_path,
            ):
                plots["plot_spread_tail_validation"] = str(spread_tail_path.resolve())

            joint_non_regression_path = study_dir / "plot_joint_non_regression.png"
            joint_labels = [
                f"{row.get('slice')}:{row.get('study_variant')}"
                for row in slice_validation_rows
                if isinstance(row.get("joint_safety_fail_rate"), int | float)
            ]
            joint_values = [
                float(row["joint_safety_fail_rate"])
                for row in slice_validation_rows
                if isinstance(row.get("joint_safety_fail_rate"), int | float)
            ]
            if _plot_categorical_bars(
                title=f"{study.study_name}: joint non-regression check by slice",
                x_labels=joint_labels,
                values=joint_values,
                y_label="joint_safety_fail_rate",
                output_path=joint_non_regression_path,
            ):
                plots["plot_joint_non_regression"] = str(
                    joint_non_regression_path.resolve()
                )

            stable_normal_non_regression_path = (
                study_dir / "plot_stable_normal_non_regression.png"
            )
            stable_normal_labels = [
                f"{row.get('slice')}:{row.get('study_variant')}:{metric}"
                for row in slice_validation_rows
                if str(row.get("slice") or "") in {"stable_contrast", "normal_holdout"}
                for metric, key in (
                    ("spread", "spread_fail_rate"),
                    ("joint", "joint_safety_fail_rate"),
                )
                if isinstance(row.get(key), int | float)
            ]
            stable_normal_values = [
                float(row[key])
                for row in slice_validation_rows
                if str(row.get("slice") or "") in {"stable_contrast", "normal_holdout"}
                for _, key in (
                    ("spread", "spread_fail_rate"),
                    ("joint", "joint_safety_fail_rate"),
                )
                if isinstance(row.get(key), int | float)
            ]
            if stable_normal_values and _plot_categorical_bars(
                title=f"{study.study_name}: stable/normal non-regression check",
                x_labels=stable_normal_labels,
                values=stable_normal_values,
                y_label="fail_rate",
                output_path=stable_normal_non_regression_path,
            ):
                plots["plot_stable_normal_non_regression"] = str(
                    stable_normal_non_regression_path.resolve()
                )

    if _stress_suite_config(study) is not None and tail_risk_summary_rows:
        stress_config = _stress_suite_config(study) or {}
        if study.problem == "tsp" and qf_pair_rows:
            tail_distribution_path = study_dir / _stress_plot_file_name(
                stress_config,
                "tail_distribution",
                "plot_tail_loss_distribution.png",
            )
            group_labels = sorted({str(row.get("case_group") or "overall") for row in qf_pair_rows})
            grouped_values = [
                [
                    float(row["route_distance_loss_pct"])
                    for row in qf_pair_rows
                    if str(row.get("case_group") or "overall") == group
                    and isinstance(row.get("route_distance_loss_pct"), int | float)
                ]
                for group in group_labels
            ]
            grouped_pairs = [
                (label, values)
                for label, values in zip(group_labels, grouped_values, strict=False)
                if values
            ]
            if grouped_pairs and _plot_box_comparison(
                title=f"{study.study_name}: tail loss distribution by case group",
                x_labels=[label for label, _ in grouped_pairs],
                grouped_values=[values for _, values in grouped_pairs],
                y_label="route_distance_loss_pct",
                output_path=tail_distribution_path,
            ):
                plots["plot_tail_loss_distribution"] = str(tail_distribution_path.resolve())

            group_loss_path = study_dir / _stress_plot_file_name(
                stress_config,
                "case_group_loss",
                "plot_case_group_vs_loss.png",
            )
            loss_rows = [
                row for row in tail_risk_summary_rows if str(row.get("scope")) != "overall"
            ]
            if loss_rows and _plot_categorical_bars(
                title=f"{study.study_name}: case group vs mean route-distance loss",
                x_labels=[str(row.get("case_group")) for row in loss_rows],
                values=[
                    float(row["mean_loss_pct"])
                    for row in loss_rows
                    if isinstance(row.get("mean_loss_pct"), int | float)
                ],
                y_label="mean_loss_pct",
                output_path=group_loss_path,
            ):
                plots["plot_case_group_vs_loss"] = str(group_loss_path.resolve())

            flip_path = study_dir / _stress_plot_file_name(
                stress_config,
                "decision_flip",
                "plot_decision_flip_vs_case_group.png",
            )
            flip_rows = [
                row
                for row in tail_risk_summary_rows
                if str(row.get("scope")) != "overall"
                and isinstance(row.get("decision_flip_rate"), int | float)
            ]
            if flip_rows and _plot_categorical_bars(
                title=f"{study.study_name}: decision flip rate by case group",
                x_labels=[str(row.get("case_group")) for row in flip_rows],
                values=[float(row["decision_flip_rate"]) for row in flip_rows],
                y_label="decision_flip_rate",
                output_path=flip_path,
            ):
                plots["plot_decision_flip_vs_case_group"] = str(flip_path.resolve())

        elif study.problem == "zdt1" and qf_pair_rows:
            hv_distribution_path = study_dir / _stress_plot_file_name(
                stress_config,
                "hv_tail_distribution",
                "plot_hv_tail_distribution.png",
            )
            hv_values = [[float(row["hv_loss_pct"]) for row in qf_pair_rows]]
            if _plot_box_comparison(
                title=f"{study.study_name}: HV tail distribution",
                x_labels=["overall"],
                grouped_values=hv_values,
                y_label="hv_loss_pct",
                output_path=hv_distribution_path,
            ):
                plots["plot_hv_tail_distribution"] = str(hv_distribution_path.resolve())

            spread_tail_path = study_dir / _stress_plot_file_name(
                stress_config,
                "spread_safety_tail",
                "plot_spread_safety_tail.png",
            )
            spread_series = []
            clean_points = [
                (float(row["hv_loss_pct"]), float(row["spread_delta"]))
                for row in qf_pair_rows
                if isinstance(row.get("hv_loss_pct"), int | float)
                and isinstance(row.get("spread_delta"), int | float)
                and float(row.get("joint_safety_fail", 0.0) or 0.0) == 0.0
            ]
            fail_points = [
                (float(row["hv_loss_pct"]), float(row["spread_delta"]))
                for row in qf_pair_rows
                if isinstance(row.get("hv_loss_pct"), int | float)
                and isinstance(row.get("spread_delta"), int | float)
                and float(row.get("joint_safety_fail", 0.0) or 0.0) > 0.0
            ]
            if clean_points:
                spread_series.append(("clean", clean_points))
            if fail_points:
                spread_series.append(("safety_fail", fail_points))
            if spread_series and _plot_metric_scatter(
                title=f"{study.study_name}: spread safety tail",
                x_label="hv_loss_pct",
                y_label="spread_delta",
                series=spread_series,
                output_path=spread_tail_path,
            ):
                plots["plot_spread_safety_tail"] = str(spread_tail_path.resolve())

            case_group_hv_path = study_dir / _stress_plot_file_name(
                stress_config,
                "case_group_hv_loss",
                "plot_case_group_vs_hv_loss.png",
            )
            hv_rows = [
                row for row in tail_risk_summary_rows if str(row.get("scope")) != "overall"
            ]
            if hv_rows and _plot_categorical_bars(
                title=f"{study.study_name}: stress group vs mean HV loss",
                x_labels=[str(row.get("case_group")) for row in hv_rows],
                values=[
                    float(row["mean_loss_pct"])
                    for row in hv_rows
                    if isinstance(row.get("mean_loss_pct"), int | float)
                ],
                y_label="mean_loss_pct",
                output_path=case_group_hv_path,
            ):
                plots["plot_case_group_vs_hv_loss"] = str(case_group_hv_path.resolve())

        elif study.problem == "knapsack" and raw_rows:
            grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
            default_variant = str(stress_config.get("default_variant") or "greedy_local_search")
            baseline_variant = str(stress_config.get("baseline_variant") or "none")
            repair_variant = str(stress_config.get("repair_variant") or "repair_only")
            for row in raw_rows:
                variant = _study_variant_key(row)
                case_id = str(row.get("case_id") or "")
                seed = row.get("seed")
                if variant not in {default_variant, baseline_variant, repair_variant}:
                    continue
                if not case_id or not isinstance(seed, int):
                    continue
                grouped.setdefault((case_id, seed), {})[variant] = row
            repair_rows: list[dict[str, Any]] = []
            for (case_id, _seed), variants in grouped.items():
                default_row = variants.get(default_variant)
                baseline_row = variants.get(baseline_variant)
                repair_row = variants.get(repair_variant)
                if default_row is None or baseline_row is None or repair_row is None:
                    continue
                default_metric = _stress_numeric(default_row.get("best_feasible_fitness"))
                baseline_metric = _stress_numeric(baseline_row.get("best_feasible_fitness"))
                repair_metric = _stress_numeric(repair_row.get("best_feasible_fitness"))
                if default_metric is None or baseline_metric is None or repair_metric is None:
                    continue
                repair_rows.append(
                    {
                        "case_id": case_id,
                        "case_group": str(repair_row.get("case_group") or "overall"),
                        "repair_gain_vs_none": repair_metric - baseline_metric,
                        "repair_gap_vs_greedy": repair_metric - default_metric,
                    }
                )
            family_gain_path = study_dir / _stress_plot_file_name(
                stress_config,
                "family_vs_repair_gain",
                "plot_family_vs_repair_gain.png",
            )
            family_labels = sorted({str(row.get("case_group") or "overall") for row in repair_rows})
            family_values = []
            family_plot_labels = []
            for label in family_labels:
                values = [
                    float(row["repair_gain_vs_none"])
                    for row in repair_rows
                    if str(row.get("case_group") or "overall") == label
                    and isinstance(row.get("repair_gain_vs_none"), int | float)
                ]
                if values:
                    family_plot_labels.append(label)
                    family_values.append(values)
            if family_values and _plot_box_comparison(
                title=f"{study.study_name}: repair gain by family",
                x_labels=family_plot_labels,
                grouped_values=family_values,
                y_label="repair_gain_vs_none",
                output_path=family_gain_path,
            ):
                plots["plot_family_vs_repair_gain"] = str(family_gain_path.resolve())

            borderline_path = study_dir / _stress_plot_file_name(
                stress_config,
                "borderline_case_stability",
                "plot_borderline_case_stability.png",
            )
            borderline_rows = [
                row
                for row in tail_risk_summary_rows
                if str(row.get("scope")) != "overall"
                and isinstance(row.get("repair_gap_vs_greedy_max_abs"), int | float)
            ]
            if borderline_rows and _plot_categorical_bars(
                title=f"{study.study_name}: borderline family stability",
                x_labels=[str(row.get("case_group")) for row in borderline_rows],
                values=[float(row["repair_gap_vs_greedy_max_abs"]) for row in borderline_rows],
                y_label="repair_gap_vs_greedy_max_abs",
                output_path=borderline_path,
            ):
                plots["plot_borderline_case_stability"] = str(borderline_path.resolve())

        elif study.problem == "onemax" and tail_risk_summary_rows:
            control_rows = [
                row
                for row in tail_risk_summary_rows
                if isinstance(row.get("control_delta_vs_reference_max_abs"), int | float)
            ]
            control_path = study_dir / _stress_plot_file_name(
                stress_config,
                "control_stability",
                "plot_control_stability.png",
            )
            if control_rows and _plot_categorical_bars(
                title=f"{study.study_name}: control stability",
                x_labels=[str(row.get("case_group")) for row in control_rows],
                values=[float(row["control_delta_vs_reference_max_abs"]) for row in control_rows],
                y_label="control_delta_vs_reference_max_abs",
                output_path=control_path,
            ):
                plots["plot_control_stability"] = str(control_path.resolve())

    if failure_trace_rows:
        if study.problem == "tsp":
            diversity_path = study_dir / "plot_trace_distance_vs_diversity.png"
            diversity_series: list[tuple[str, list[tuple[float, float]]]] = []
            for case_group in sorted({str(row.get("case_group") or "overall") for row in failure_trace_rows}):
                points = [
                    (
                        float(row["final_edge_diversity_ratio"]),
                        float(row["route_distance_loss_pct_vs_quality"]),
                    )
                    for row in failure_trace_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("final_edge_diversity_ratio"), int | float)
                    and isinstance(row.get("route_distance_loss_pct_vs_quality"), int | float)
                ]
                if points:
                    diversity_series.append((case_group, points))
            if diversity_series and _plot_metric_scatter(
                title=f"{study.study_name}: final diversity vs loss",
                x_label="final_edge_diversity_ratio",
                y_label="route_distance_loss_pct_vs_quality",
                series=diversity_series,
                output_path=diversity_path,
            ):
                plots["plot_trace_distance_vs_diversity"] = str(diversity_path.resolve())

            headstart_path = study_dir / "plot_trace_headstart_vs_final_gap.png"
            headstart_series: list[tuple[str, list[tuple[float, float]]]] = []
            for variant in sorted({str(row.get("study_variant") or "") for row in failure_trace_rows}):
                points = [
                    (
                        float(row["initial_best_headstart_vs_q"]),
                        float(row["route_distance_delta_vs_quality"]),
                    )
                    for row in failure_trace_rows
                    if str(row.get("study_variant") or "") == variant
                    and isinstance(row.get("initial_best_headstart_vs_q"), int | float)
                    and isinstance(row.get("route_distance_delta_vs_quality"), int | float)
                ]
                if points:
                    headstart_series.append((variant, points))
            if headstart_series and _plot_metric_scatter(
                title=f"{study.study_name}: initial head-start vs final gap",
                x_label="initial_best_headstart_vs_q",
                y_label="route_distance_delta_vs_quality",
                series=headstart_series,
                output_path=headstart_path,
            ):
                plots["plot_trace_headstart_vs_final_gap"] = str(headstart_path.resolve())

            slope_path = study_dir / "plot_trace_improvement_slope.png"
            slope_series: list[tuple[str, list[tuple[float, float]]]] = []
            for case_group in sorted({str(row.get("case_group") or "overall") for row in failure_trace_rows}):
                points = [
                    (
                        float(row["last_improvement_generation"]),
                        float(row["recent_window_slope"]),
                    )
                    for row in failure_trace_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("last_improvement_generation"), int | float)
                    and isinstance(row.get("recent_window_slope"), int | float)
                ]
                if points:
                    slope_series.append((case_group, points))
            if slope_series and _plot_metric_scatter(
                title=f"{study.study_name}: last improvement vs recent slope",
                x_label="last_improvement_generation",
                y_label="recent_window_slope",
                series=slope_series,
                output_path=slope_path,
            ):
                plots["plot_trace_improvement_slope"] = str(slope_path.resolve())

            late_refine_path = study_dir / "plot_late_refinement_gap.png"
            late_refine_series: list[tuple[str, list[tuple[float, float]]]] = []
            for case_group in sorted({str(row.get("case_group") or "overall") for row in failure_trace_rows}):
                points = [
                    (
                        float(row["late_refinement_score"]),
                        float(row["route_distance_loss_pct_vs_quality"]),
                    )
                    for row in failure_trace_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("late_refinement_score"), int | float)
                    and isinstance(row.get("route_distance_loss_pct_vs_quality"), int | float)
                ]
                if points:
                    late_refine_series.append((case_group, points))
            if late_refine_series and _plot_metric_scatter(
                title=f"{study.study_name}: late refinement vs loss",
                x_label="late_refinement_score",
                y_label="route_distance_loss_pct_vs_quality",
                series=late_refine_series,
                output_path=late_refine_path,
            ):
                plots["plot_late_refinement_gap"] = str(late_refine_path.resolve())

            collapse_gap_path = study_dir / "plot_collapse_onset_vs_last_improvement.png"
            collapse_gap_series: list[tuple[str, list[tuple[float, float]]]] = []
            for case_group in sorted({str(row.get("case_group") or "overall") for row in failure_trace_rows}):
                points = [
                    (
                        float(row["collapse_onset_generation"]),
                        float(row["last_improvement_generation"]),
                    )
                    for row in failure_trace_rows
                    if str(row.get("case_group") or "overall") == case_group
                    and isinstance(row.get("collapse_onset_generation"), int | float)
                    and isinstance(row.get("last_improvement_generation"), int | float)
                ]
                if points:
                    collapse_gap_series.append((case_group, points))
            if collapse_gap_series and _plot_metric_scatter(
                title=f"{study.study_name}: collapse onset vs last improvement",
                x_label="collapse_onset_generation",
                y_label="last_improvement_generation",
                series=collapse_gap_series,
                output_path=collapse_gap_path,
            ):
                plots["plot_collapse_onset_vs_last_improvement"] = str(collapse_gap_path.resolve())

            overlay_path = study_dir / "plot_anticase_vs_rescue_overlay.png"
            overlay_labels = sorted({str(row.get("case_group") or "overall") for row in failure_trace_rows})
            overlay_values = []
            overlay_names = []
            for label in overlay_labels:
                values = [
                    float(row["route_distance_loss_pct_vs_quality"])
                    for row in failure_trace_rows
                    if str(row.get("case_group") or "overall") == label
                    and isinstance(row.get("route_distance_loss_pct_vs_quality"), int | float)
                ]
                if values:
                    overlay_names.append(label)
                    overlay_values.append(values)
            if overlay_values and _plot_box_comparison(
                title=f"{study.study_name}: anti-case vs rescue loss overlay",
                x_labels=overlay_names,
                grouped_values=overlay_values,
                y_label="route_distance_loss_pct_vs_quality",
                output_path=overlay_path,
            ):
                plots["plot_anticase_vs_rescue_overlay"] = str(overlay_path.resolve())

            tradeoff_path = study_dir / "plot_population_generation_tradeoff_vs_tail.png"
            tradeoff_names: list[str] = []
            tradeoff_values: list[list[float]] = []
            for variant in sorted({str(row.get("study_variant") or "") for row in failure_trace_rows}):
                values = [
                    float(row["route_distance_loss_pct_vs_quality"])
                    for row in failure_trace_rows
                    if str(row.get("study_variant") or "") == variant
                    and str(row.get("case_group") or "") == "anti_case"
                    and isinstance(row.get("route_distance_loss_pct_vs_quality"), int | float)
                ]
                if values:
                    tradeoff_names.append(variant)
                    tradeoff_values.append(values)
            if tradeoff_values and _plot_box_comparison(
                title=f"{study.study_name}: population/generation tradeoff vs anti-case tail",
                x_labels=tradeoff_names,
                grouped_values=tradeoff_values,
                y_label="route_distance_loss_pct_vs_quality",
                output_path=tradeoff_path,
            ):
                plots["plot_population_generation_tradeoff_vs_tail"] = str(tradeoff_path.resolve())

        elif study.problem == "zdt1":
            hv_spread_path = study_dir / "plot_trace_hv_vs_spread.png"
            hv_spread_series: list[tuple[str, list[tuple[float, float]]]] = []
            for label, flag_key in (("safe", "joint_safety_fail"), ("fail", "joint_safety_fail")):
                points = [
                    (
                        float(row["hv_loss_pct_vs_quality"]),
                        float(row["spread_delta_vs_quality"]),
                    )
                    for row in failure_trace_rows
                    if ((float(row.get(flag_key, 0.0) or 0.0) > 0.0) == (label == "fail"))
                    and isinstance(row.get("hv_loss_pct_vs_quality"), int | float)
                    and isinstance(row.get("spread_delta_vs_quality"), int | float)
                ]
                if points:
                    hv_spread_series.append((label, points))
            if hv_spread_series and _plot_metric_scatter(
                title=f"{study.study_name}: HV loss vs spread delta",
                x_label="hv_loss_pct_vs_quality",
                y_label="spread_delta_vs_quality",
                series=hv_spread_series,
                output_path=hv_spread_path,
            ):
                plots["plot_trace_hv_vs_spread"] = str(hv_spread_path.resolve())

            front_hv_path = study_dir / "plot_trace_frontsize_vs_hv.png"
            front_hv_series: list[tuple[str, list[tuple[float, float]]]] = []
            for variant in sorted({str(row.get("study_variant") or "") for row in failure_trace_rows}):
                points = [
                    (
                        float(row["final_front_size"]),
                        float(row["hv_loss_pct_vs_quality"]),
                    )
                    for row in failure_trace_rows
                    if str(row.get("study_variant") or "") == variant
                    and isinstance(row.get("final_front_size"), int | float)
                    and isinstance(row.get("hv_loss_pct_vs_quality"), int | float)
                ]
                if points:
                    front_hv_series.append((variant, points))
            if front_hv_series and _plot_metric_scatter(
                title=f"{study.study_name}: front size vs HV loss",
                x_label="final_front_size",
                y_label="hv_loss_pct_vs_quality",
                series=front_hv_series,
                output_path=front_hv_path,
            ):
                plots["plot_trace_frontsize_vs_hv"] = str(front_hv_path.resolve())

            onset_path = study_dir / "plot_trace_safety_fail_onset.png"
            onset_series: list[tuple[str, list[tuple[float, float]]]] = []
            for case_group in sorted({str(row.get("target_id") or "overall") for row in failure_trace_rows}):
                points = [
                    (
                        float(row["safety_fail_onset_generation"]),
                        float(row["spread_delta_vs_quality"]),
                    )
                    for row in failure_trace_rows
                    if str(row.get("target_id") or "overall") == case_group
                    and isinstance(row.get("safety_fail_onset_generation"), int | float)
                    and isinstance(row.get("spread_delta_vs_quality"), int | float)
                ]
                if points:
                    onset_series.append((case_group, points))
            if onset_series and _plot_metric_scatter(
                title=f"{study.study_name}: safety fail onset vs spread delta",
                x_label="safety_fail_onset_generation",
                y_label="spread_delta_vs_quality",
                series=onset_series,
                output_path=onset_path,
            ):
                plots["plot_trace_safety_fail_onset"] = str(onset_path.resolve())
                plots["plot_safety_fail_onset"] = str(onset_path.resolve())

            overlay_path = study_dir / "plot_safe_vs_fail_overlay.png"
            overlay_labels = ["safe", "fail"]
            overlay_values = [
                [
                    float(row["hv_loss_pct_vs_quality"])
                    for row in failure_trace_rows
                    if float(row.get("joint_safety_fail", 0.0) or 0.0) == 0.0
                    and isinstance(row.get("hv_loss_pct_vs_quality"), int | float)
                ],
                [
                    float(row["hv_loss_pct_vs_quality"])
                    for row in failure_trace_rows
                    if float(row.get("joint_safety_fail", 0.0) or 0.0) > 0.0
                    and isinstance(row.get("hv_loss_pct_vs_quality"), int | float)
                ],
            ]
            filtered_labels = [label for label, values in zip(overlay_labels, overlay_values, strict=False) if values]
            filtered_values = [values for values in overlay_values if values]
            if filtered_values and _plot_box_comparison(
                title=f"{study.study_name}: safe vs fail HV overlay",
                x_labels=filtered_labels,
                grouped_values=filtered_values,
                y_label="hv_loss_pct_vs_quality",
                output_path=overlay_path,
            ):
                plots["plot_safe_vs_fail_overlay"] = str(overlay_path.resolve())

            split_path = study_dir / "plot_spread_vs_joint_fail_split.png"
            split_labels = ["safe", "spread_only", "joint_fail"]
            split_values = [
                [
                    float(row["spread_delta_vs_quality"])
                    for row in failure_trace_rows
                    if float(row.get("spread_safety_fail", 0.0) or 0.0) == 0.0
                    and float(row.get("joint_safety_fail", 0.0) or 0.0) == 0.0
                    and isinstance(row.get("spread_delta_vs_quality"), int | float)
                ],
                [
                    float(row["spread_delta_vs_quality"])
                    for row in failure_trace_rows
                    if float(row.get("spread_safety_fail", 0.0) or 0.0) > 0.0
                    and float(row.get("joint_safety_fail", 0.0) or 0.0) == 0.0
                    and isinstance(row.get("spread_delta_vs_quality"), int | float)
                ],
                [
                    float(row["spread_delta_vs_quality"])
                    for row in failure_trace_rows
                    if float(row.get("joint_safety_fail", 0.0) or 0.0) > 0.0
                    and isinstance(row.get("spread_delta_vs_quality"), int | float)
                ],
            ]
            filtered_split_labels = [
                label for label, values in zip(split_labels, split_values, strict=False) if values
            ]
            filtered_split_values = [values for values in split_values if values]
            if filtered_split_values and _plot_box_comparison(
                title=f"{study.study_name}: spread vs joint fail split",
                x_labels=filtered_split_labels,
                grouped_values=filtered_split_values,
                y_label="spread_delta_vs_quality",
                output_path=split_path,
            ):
                plots["plot_spread_vs_joint_fail_split"] = str(split_path.resolve())

    if study.problem == "zdt1":
        best_variant = _best_variant_row(study, summary_rows)
        if best_variant is not None:
            points: list[tuple[float, float]] = []
            for row in raw_rows:
                if row["variant_label"] != best_variant["variant_label"]:
                    continue
                for vector in row.get("pareto_front_vectors", []):
                    if (
                        isinstance(vector, list)
                        and len(vector) == 2
                        and isinstance(vector[0], int | float)
                        and isinstance(vector[1], int | float)
                    ):
                        points.append((float(vector[0]), float(vector[1])))
            pareto_path = study_dir / "plot_final_pareto_front.png"
            if _plot_pareto_scatter(
                title=f"{study.study_name}: final Pareto front",
                points=points,
                output_path=pareto_path,
            ):
                plots["plot_final_pareto_front"] = str(pareto_path.resolve())

    if study.problem in {"tsp", "zdt1", "knapsack"}:
        trigger_path = study_dir / "plot_trigger_events.png"
        trigger_series: list[tuple[str, list[tuple[float, float]]]] = []
        for variant in sorted({str(row["variant_label"]) for row in raw_rows}):
            points = []
            for row in raw_rows:
                if row["variant_label"] != variant:
                    continue
                generations = row.get("trigger_event_generations", [])
                if not isinstance(generations, list):
                    continue
                for generation in generations:
                    if isinstance(generation, int | float):
                        points.append((float(generation), float(row["seed"])))
            if points:
                trigger_series.append((variant, points))
        if _plot_trigger_events(
            title=f"{study.study_name}: trigger events by generation",
            series=trigger_series,
            output_path=trigger_path,
        ):
            plots["plot_trigger_events"] = str(trigger_path.resolve())

    return plots


def _single_run_plot_outputs(payload: dict[str, Any], local_dir: Path) -> dict[str, str]:
    plots: dict[str, str] = {}
    history_rows = payload.get("history_rows", [])
    if isinstance(history_rows, list) and history_rows:
        metric_name = _problem_history_metric(str(payload["problem"]), {})
        points = []
        for row in history_rows:
            generation = row.get("generation")
            metric_value = _coerce_metric_value(str(payload["problem"]), metric_name, row)
            if isinstance(generation, int) and metric_value is not None:
                points.append((float(generation), metric_value))
        history_path = local_dir / "plot_convergence.png"
        if _plot_history_lines(
            title=f"{payload['label']}: {metric_name} vs generation",
            x_label="Generation",
            y_label=metric_name,
            series=[(str(payload["label"]), points)],
            output_path=history_path,
        ):
            plots["plot_convergence"] = str(history_path.resolve())

        diversity_column = _problem_diversity_column(str(payload["problem"]))
        if diversity_column is not None:
            diversity_points = [
                (float(row["generation"]), float(row[diversity_column]))
                for row in history_rows
                if isinstance(row.get("generation"), int)
                and isinstance(row.get(diversity_column), int | float)
            ]
            diversity_path = local_dir / "plot_diversity.png"
            if _plot_history_lines(
                title=f"{payload['label']}: {diversity_column} vs generation",
                x_label="Generation",
                y_label=diversity_column,
                series=[(str(payload["label"]), diversity_points)],
                output_path=diversity_path,
            ):
                plots["plot_diversity"] = str(diversity_path.resolve())

        stagnation_points = [
            (float(row["generation"]), float(row["generations_since_last_improvement"]))
            for row in history_rows
            if isinstance(row.get("generation"), int)
            and isinstance(row.get("generations_since_last_improvement"), int | float)
        ]
        stagnation_path = local_dir / "plot_stagnation.png"
        if _plot_history_lines(
            title=f"{payload['label']}: generations_since_last_improvement",
            x_label="Generation",
            y_label="generations_since_last_improvement",
            series=[(str(payload["label"]), stagnation_points)],
            output_path=stagnation_path,
        ):
            plots["plot_stagnation"] = str(stagnation_path.resolve())

        if payload.get("problem") == "tsp":
            route_points = [
                (float(row["generation"]), float(row["best_route_distance"]))
                for row in history_rows
                if isinstance(row.get("generation"), int)
                and isinstance(row.get("best_route_distance"), int | float)
            ]
            route_path = local_dir / "plot_route_distance.png"
            if _plot_history_lines(
                title=f"{payload['label']}: best_route_distance vs generation",
                x_label="Generation",
                y_label="best_route_distance",
                series=[(str(payload["label"]), route_points)],
                output_path=route_path,
            ):
                plots["plot_route_distance"] = str(route_path.resolve())

        if payload.get("problem") == "knapsack":
            feasibility_points = [
                (float(row["generation"]), float(row["feasible_ratio"]))
                for row in history_rows
                if isinstance(row.get("generation"), int)
                and isinstance(row.get("feasible_ratio"), int | float)
            ]
            feasibility_path = local_dir / "plot_feasibility.png"
            if _plot_history_lines(
                title=f"{payload['label']}: feasible_ratio vs generation",
                x_label="Generation",
                y_label="feasible_ratio",
                series=[(str(payload["label"]), feasibility_points)],
                output_path=feasibility_path,
            ):
                plots["plot_feasibility"] = str(feasibility_path.resolve())
            mutation_rate_points = [
                (float(row["generation"]), float(row["adaptive_mutation_rate"]))
                for row in history_rows
                if isinstance(row.get("generation"), int)
                and isinstance(row.get("adaptive_mutation_rate"), int | float)
            ]
            mutation_rate_path = local_dir / "plot_mutation_rate.png"
            if _plot_history_lines(
                title=f"{payload['label']}: adaptive_mutation_rate vs generation",
                x_label="Generation",
                y_label="adaptive_mutation_rate",
                series=[(str(payload["label"]), mutation_rate_points)],
                output_path=mutation_rate_path,
            ):
                plots["plot_mutation_rate"] = str(mutation_rate_path.resolve())

        if payload.get("problem") == "zdt1":
            hv_points = [
                (float(row["generation"]), float(row["hypervolume"]))
                for row in history_rows
                if isinstance(row.get("generation"), int)
                and isinstance(row.get("hypervolume"), int | float)
            ]
            hv_path = local_dir / "plot_hypervolume.png"
            if _plot_history_lines(
                title=f"{payload['label']}: hypervolume vs generation",
                x_label="Generation",
                y_label="hypervolume",
                series=[(str(payload["label"]), hv_points)],
                output_path=hv_path,
            ):
                plots["plot_hypervolume"] = str(hv_path.resolve())
    if payload.get("problem") in {"tsp", "zdt1", "knapsack"}:
        trigger_generations = payload["metrics"].get("trigger_event_generations", [])
        if isinstance(trigger_generations, list) and trigger_generations:
            trigger_points = [
                (float(generation), 1.0)
                for generation in trigger_generations
                if isinstance(generation, int | float)
            ]
            trigger_path = local_dir / "plot_trigger_events.png"
            if _plot_trigger_events(
                title=f"{payload['label']}: trigger events by generation",
                series=[(str(payload["label"]), trigger_points)],
                output_path=trigger_path,
            ):
                plots["plot_trigger_events"] = str(trigger_path.resolve())

    if payload.get("problem") == "zdt1":
        points = []
        for vector in payload["metrics"].get("pareto_front_vectors", []):
            if (
                isinstance(vector, list)
                and len(vector) == 2
                and isinstance(vector[0], int | float)
                and isinstance(vector[1], int | float)
            ):
                points.append((float(vector[0]), float(vector[1])))
        pareto_path = local_dir / "plot_final_pareto_front.png"
        if _plot_pareto_scatter(
            title=f"{payload['label']}: final Pareto front",
            points=points,
            output_path=pareto_path,
        ):
            plots["plot_final_pareto_front"] = str(pareto_path.resolve())
    return plots


def run_local_experiment(
    *,
    preset: str | None = None,
    demo: str | None = None,
    config_path: str | Path | None = None,
    output_root: str | Path = "outputs/local_runs",
    seed: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    selected = [preset is not None, demo is not None, config_path is not None]
    if sum(selected) != 1:
        raise ValueError("Provide exactly one of preset, demo, or config_path")

    source_name = preset or demo or Path(config_path or "").stem
    kind = "preset" if preset is not None else "demo" if demo is not None else "config"
    local_dir = Path(output_root) / f"{_timestamp()}_{kind}_{_safe_name(str(source_name))}"
    artifact_root = local_dir / "artifacts"
    local_dir.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, Any] = {}
    if seed is not None:
        overrides["seed"] = seed
    if run_name is not None:
        overrides["run_name"] = run_name

    if preset is not None:
        result = run_preset(preset, output_root=artifact_root, overrides=overrides or None)
        problem = result.raw_summary.get("problem")
    elif demo is not None:
        result = run_demo(demo, output_root=artifact_root)
        problem = _DEMO_PROBLEMS.get(demo, result.raw_summary.get("problem"))
    else:
        config_data = load_config(config_path).to_dict()
        config_data.update(overrides)
        result = run_config(config_data, output_root=artifact_root)
        problem = result.raw_summary.get("problem")

    if not isinstance(problem, str):
        raise ValueError("Could not resolve problem for local experiment")

    summary_path = Path(result.summary_path) if result.summary_path else None
    history_rows = (
        _read_history_csv(summary_path.parent / "history.csv")
        if summary_path is not None
        else []
    )
    payload = {
        "schema_version": LOCAL_STUDY_SCHEMA_VERSION,
        "kind": kind,
        "label": str(source_name),
        "problem": problem,
        "source_name": str(source_name),
        "artifact_dir": str(artifact_root.resolve()),
        "run_output_dir": result.output_dir,
        "summary_path": result.summary_path,
        "metrics": _extract_problem_metrics(problem, result.raw_summary),
        "raw_summary": result.raw_summary,
        "history_rows": history_rows,
    }

    raw_path = local_dir / "raw_result.json"
    summary_csv_path = local_dir / "summary.csv"
    summary_md_path = local_dir / "summary.md"
    _write_json(raw_path, payload)
    _write_rows_csv(
        summary_csv_path,
        [
            {
                "kind": kind,
                "label": str(source_name),
                "problem": problem,
                **{
                    key: value
                    for key, value in payload["metrics"].items()
                    if key != "pareto_front_vectors"
                },
            }
        ],
    )
    summary_md_path.write_text(_render_single_run_markdown(payload), encoding="utf-8")
    payload["plots"] = _single_run_plot_outputs(payload, local_dir)
    _write_json(raw_path, payload)
    return {
        "local_dir": str(local_dir.resolve()),
        "raw_result_json": str(raw_path.resolve()),
        "summary_csv": str(summary_csv_path.resolve()),
        "summary_md": str(summary_md_path.resolve()),
        "plots": payload["plots"],
    }


def run_local_study(
    study_ref: str | Path,
    *,
    output_root: str | Path = "outputs/local_studies",
) -> dict[str, Any]:
    study = load_local_study(study_ref)
    study_dir = Path(output_root) / f"{_timestamp()}_{_safe_name(study.study_name)}"
    runs_root = study_dir / "runs"
    study_dir.mkdir(parents=True, exist_ok=True)

    base_data = apply_overrides(_base_config_data(study), study.shared_overrides)
    axes = list(study.sweep.keys())
    cases = study.cases or (
        LocalStudyCase(case_id="default", overrides={}, note=""),
    )
    include_case_label = bool(study.cases)
    raw_rows: list[dict[str, Any]] = []
    history_payloads: list[dict[str, Any]] = []

    variant_index = 0
    for case in cases:
        case_base_data = apply_overrides(base_data, case.overrides)
        for values in product(*(study.sweep[axis] for axis in axes)):
            variant_index += 1
            combo = {axis: value for axis, value in zip(axes, values, strict=True)}
            combo_label = _combo_label(combo)
            label = (
                f"case={case.case_id} | {combo_label}" if include_case_label else combo_label
            )
            variant_payload, config_combo = _variant_payload(study, combo)
            config_template = apply_overrides(
                apply_overrides(case_base_data, variant_payload),
                config_combo,
            )
            for seed in study.seeds:
                config_data = apply_overrides(
                    config_template,
                    {"seed": seed, "run_name": _run_name(study.study_name, label, seed)},
                )
                local_baseline = config_data.pop("__local_baseline__", None)
                study_metadata = config_data.pop("__study_metadata__", {})
                if not isinstance(study_metadata, dict):
                    study_metadata = {}
                config = GAConfig.from_dict(config_data)
                if local_baseline == "knapsack_greedy_local_search":
                    result = _knapsack_greedy_local_search_result(config, output_root=runs_root)
                elif local_baseline == "knapsack_repair_rerun_gate":
                    result = _knapsack_rerun_gate_result(config, output_root=runs_root)
                elif local_baseline == "tsp_restart_portfolio":
                    result = _single_objective_restart_portfolio_result(
                        config,
                        output_root=runs_root,
                        problem="tsp",
                    )
                elif local_baseline == "knapsack_restart_portfolio":
                    result = _single_objective_restart_portfolio_result(
                        config,
                        output_root=runs_root,
                        problem="knapsack",
                    )
                elif local_baseline == "zdt1_restart_portfolio":
                    result = _zdt1_restart_portfolio_result(config, output_root=runs_root)
                elif local_baseline == "tsp_two_stage_gate":
                    result = _two_stage_escalation_result(
                        config,
                        output_root=runs_root,
                        problem="tsp",
                    )
                elif local_baseline == "zdt1_two_stage_gate":
                    result = _two_stage_escalation_result(
                        config,
                        output_root=runs_root,
                        problem="zdt1",
                    )
                else:
                    result = run_experiment(config, output_root=runs_root)
                summary_path = result.output_dir / "summary.json"
                history_path = result.output_dir / "history.csv"
                history_rows = _read_history_csv(history_path)
                resolved_columns = _resolved_config_columns(config_data)
                if isinstance(local_baseline, str) and local_baseline.strip():
                    resolved_columns["algorithm"] = local_baseline.strip()
                row = {
                    "study_name": study.study_name,
                    "problem": study.problem,
                    "variant_index": variant_index,
                    "variant_label": label,
                    "parameter_values": combo,
                    "seed": seed,
                    "summary_path": str(summary_path.resolve()),
                    "history_path": str(history_path.resolve()),
                    "output_dir": str(result.output_dir.resolve()),
                    "case_id": case.case_id if include_case_label else None,
                    "case_note": case.note if include_case_label else "",
                    "case_group": case.group if include_case_label else "",
                    "family_label": case.group if include_case_label and case.group else "",
                    **_parameter_columns(combo),
                    **resolved_columns,
                    **_parameter_columns(resolved_columns),
                    **_study_metadata_fields(study_metadata),
                    **_extract_problem_metrics(study.problem, result.summary),
                    **_extract_initial_history_metrics(study.problem, history_rows),
                    **_extract_final_history_metrics(study.problem, history_rows),
                    **_trigger_selectivity_metrics(
                        study.problem,
                        history_rows,
                        result.summary,
                        config_data,
                    ),
                }
                seeded_count = row.get("hybrid_seeded_individuals")
                population_size = config_data.get("population_size")
                if (
                    isinstance(seeded_count, int | float)
                    and isinstance(population_size, int | float)
                    and float(population_size) > 0.0
                ):
                    row["seeded_population_fraction_realized"] = float(seeded_count) / float(
                        population_size
                    )
                raw_rows.append(row)
                history_payloads.append(
                    {
                        "variant_label": label,
                        "parameter_values": combo,
                        "seed": seed,
                        "history": history_rows,
                    }
                )

    if study.problem == "tsp":
        _annotate_tsp_regret_rows(raw_rows)
        _annotate_restart_portfolio_rows("tsp", raw_rows)
        _annotate_two_stage_gate_rows("tsp", raw_rows)
    if study.problem == "knapsack":
        _annotate_knapsack_regret_rows(raw_rows)
        _annotate_restart_portfolio_rows("knapsack", raw_rows)
    if study.problem == "zdt1":
        _annotate_restart_portfolio_rows("zdt1", raw_rows)
        _annotate_two_stage_gate_rows("zdt1", raw_rows)
    summary_rows = _variant_summary_rows(study, raw_rows)
    case_group_summary_rows: list[dict[str, Any]] = []
    if study.problem == "tsp":
        _annotate_tsp_regret_summary(summary_rows, raw_rows)
        case_group_summary_rows = _tsp_case_group_summary_rows(study, raw_rows)
    if study.problem == "knapsack":
        case_group_summary_rows = _knapsack_case_group_summary_rows(study, raw_rows)
    if study.problem == "zdt1":
        _annotate_zdt1_regret_summary(summary_rows)
    ranking_detail_rows = _triage_detail_rows(study, raw_rows)
    ranking_fidelity_rows = _triage_ranking_fidelity_rows(study, ranking_detail_rows)
    triage_workflow_rows = _triage_workflow_rows(study, ranking_detail_rows)
    qf_pair_rows = _qf_pair_rows(study, raw_rows)
    tolerance_rows = _qf_tolerance_rows(study, qf_pair_rows)
    seed_budget_rows = _seed_budget_rows(study, raw_rows, qf_pair_rows)
    sequential_decision_rows = _sequential_decision_rows(study, raw_rows, qf_pair_rows)
    tsp_fast_tail_pair_rows = _tsp_fast_tail_pair_rows(study, raw_rows)
    tsp_fast_tail_summary_rows = _tsp_fast_tail_summary_rows(study, tsp_fast_tail_pair_rows)
    zdt1_fast_hardening_pair_rows = _zdt1_fast_hardening_pair_rows(study, raw_rows)
    zdt1_fast_hardening_summary_rows = _zdt1_fast_hardening_summary_rows(
        study,
        zdt1_fast_hardening_pair_rows,
    )
    zdt1_spread_validation_rows = _zdt1_spread_candidate_validation_rows(
        study,
        zdt1_fast_hardening_pair_rows,
    )
    zdt1_spread_boundary_rows = _zdt1_spread_boundary_rows(
        study,
        zdt1_spread_validation_rows,
    )
    stress_case_catalog_rows = _stress_case_catalog_rows(study, raw_rows, qf_pair_rows)
    tail_risk_summary_rows = _stress_tail_summary_rows(study, raw_rows, qf_pair_rows)
    if not tail_risk_summary_rows:
        if tsp_fast_tail_summary_rows:
            tail_risk_summary_rows = list(tsp_fast_tail_summary_rows)
        elif zdt1_spread_validation_rows:
            tail_risk_summary_rows = list(zdt1_spread_validation_rows)
        elif zdt1_fast_hardening_summary_rows:
            tail_risk_summary_rows = list(zdt1_fast_hardening_summary_rows)
    failure_trace_rows = build_failure_trace_rows(
        study.problem,
        _failure_trace_config(study),
        raw_rows,
    )
    failure_hypothesis_rows = build_failure_hypothesis_rows(
        study.problem,
        _failure_trace_config(study),
        failure_trace_rows,
    )
    history_rows = _history_summary_rows(study, history_payloads)
    best_variant = _best_variant_row(study, summary_rows)
    plots = _study_plot_outputs(
        study,
        raw_rows,
        summary_rows,
        history_rows,
        case_group_summary_rows,
        ranking_detail_rows,
        ranking_fidelity_rows,
        triage_workflow_rows,
        qf_pair_rows,
        tolerance_rows,
        seed_budget_rows,
        sequential_decision_rows,
        tsp_fast_tail_pair_rows,
        tsp_fast_tail_summary_rows,
        zdt1_spread_validation_rows,
        stress_case_catalog_rows,
        tail_risk_summary_rows,
        failure_trace_rows,
        study_dir,
    )

    bundle = {
        "schema_version": LOCAL_STUDY_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "study": study.to_dict(),
        "study_dir": str(study_dir.resolve()),
        "raw_runs": raw_rows,
        "summary_rows": summary_rows,
        "case_group_summary_rows": case_group_summary_rows,
        "ranking_detail_rows": ranking_detail_rows,
        "ranking_fidelity_rows": ranking_fidelity_rows,
        "triage_workflow_rows": triage_workflow_rows,
        "qf_pair_rows": qf_pair_rows,
        "tolerance_rows": tolerance_rows,
        "seed_budget_rows": seed_budget_rows,
        "sequential_decision_rows": sequential_decision_rows,
        "tsp_fast_tail_pair_rows": tsp_fast_tail_pair_rows,
        "tsp_fast_tail_summary_rows": tsp_fast_tail_summary_rows,
        "zdt1_fast_hardening_pair_rows": zdt1_fast_hardening_pair_rows,
        "zdt1_fast_hardening_summary_rows": zdt1_fast_hardening_summary_rows,
        "zdt1_spread_validation_rows": zdt1_spread_validation_rows,
        "zdt1_spread_boundary_rows": zdt1_spread_boundary_rows,
        "stress_case_catalog_rows": stress_case_catalog_rows,
        "tail_risk_summary_rows": tail_risk_summary_rows,
        "failure_trace_rows": failure_trace_rows,
        "failure_hypothesis_rows": failure_hypothesis_rows,
        "history_rows": history_rows,
        "best_variant": best_variant,
        "plots": plots,
    }

    _write_json(study_dir / "study.json", study.to_dict())
    _write_rows_csv(study_dir / "raw_results.csv", raw_rows)
    _write_json(study_dir / "raw_results.json", {"rows": raw_rows})
    _write_rows_csv(study_dir / "summary.csv", summary_rows)
    if case_group_summary_rows:
        _write_rows_csv(study_dir / "case_group_summary.csv", case_group_summary_rows)
    if ranking_detail_rows:
        _write_rows_csv(study_dir / "ranking_detail.csv", ranking_detail_rows)
    if ranking_fidelity_rows:
        _write_rows_csv(study_dir / "ranking_fidelity.csv", ranking_fidelity_rows)
    if triage_workflow_rows:
        _write_rows_csv(study_dir / "triage_workflow_summary.csv", triage_workflow_rows)
    if qf_pair_rows:
        _write_rows_csv(study_dir / "qf_pair_rows.csv", qf_pair_rows)
    if tolerance_rows:
        _write_rows_csv(study_dir / "tolerance_table.csv", tolerance_rows)
    if seed_budget_rows:
        _write_rows_csv(study_dir / "seed_budget_table.csv", seed_budget_rows)
    if sequential_decision_rows:
        _write_rows_csv(study_dir / "sequential_decision_table.csv", sequential_decision_rows)
    if tsp_fast_tail_pair_rows:
        _write_rows_csv(study_dir / "tsp_fast_tail_rows.csv", tsp_fast_tail_pair_rows)
    if tsp_fast_tail_summary_rows:
        _write_rows_csv(study_dir / "tsp_fast_tail_summary.csv", tsp_fast_tail_summary_rows)
    if zdt1_fast_hardening_pair_rows:
        _write_rows_csv(
            study_dir / "zdt1_fast_hardening_rows.csv",
            zdt1_fast_hardening_pair_rows,
        )
    if zdt1_fast_hardening_summary_rows:
        _write_rows_csv(
            study_dir / "zdt1_fast_hardening_summary.csv",
            zdt1_fast_hardening_summary_rows,
        )
    if zdt1_spread_validation_rows:
        _write_rows_csv(
            study_dir / "zdt1_spread_candidate_validation_summary.csv",
            zdt1_spread_validation_rows,
        )
    if zdt1_spread_boundary_rows:
        _write_rows_csv(
            study_dir / "zdt1_spread_candidate_boundary_table.csv",
            zdt1_spread_boundary_rows,
        )
        (study_dir / "zdt1_spread_candidate_boundary_notes.md").write_text(
            _render_zdt1_spread_boundary_notes(
                study,
                zdt1_spread_boundary_rows,
            ),
            encoding="utf-8",
        )
    if stress_case_catalog_rows:
        _write_rows_csv(study_dir / "stress_case_catalog.csv", stress_case_catalog_rows)
        (study_dir / "stress_case_catalog.md").write_text(
            _render_stress_case_catalog_markdown(study, stress_case_catalog_rows),
            encoding="utf-8",
        )
    if tail_risk_summary_rows:
        _write_rows_csv(study_dir / "tail_risk_summary.csv", tail_risk_summary_rows)
        if _stress_target_reduction_config(study) is not None:
            _write_rows_csv(study_dir / "tail_risk_reduction_summary.csv", tail_risk_summary_rows)
    if failure_trace_rows:
        _write_rows_csv(study_dir / "failure_trace_table.csv", failure_trace_rows)
    if failure_hypothesis_rows:
        _write_rows_csv(study_dir / "failure_hypotheses.csv", failure_hypothesis_rows)
    if (
        _tsp_tail_freeze_config(study) is not None
        and tsp_fast_tail_summary_rows
    ):
        tail_freeze_summary = _render_tsp_tail_freeze_summary(
            study,
            tsp_fast_tail_summary_rows,
            failure_hypothesis_rows,
        )
        (study_dir / "tsp_irreducible_tail_freeze_summary.md").write_text(
            tail_freeze_summary,
            encoding="utf-8",
        )
        (study_dir / "tsp_protocol_limitation_freeze_summary.md").write_text(
            tail_freeze_summary,
            encoding="utf-8",
        )
    stress_notes_enabled = _stress_suite_config(study) is not None
    if stress_notes_enabled and (stress_case_catalog_rows or tail_risk_summary_rows):
        (study_dir / "stress_suite_notes.md").write_text(
            _render_stress_suite_notes(study, stress_case_catalog_rows, tail_risk_summary_rows),
            encoding="utf-8",
        )
    _write_rows_csv(study_dir / "history_summary.csv", history_rows)
    (study_dir / "summary.md").write_text(
        _render_study_markdown(
            study,
            summary_rows,
            best_variant,
            case_group_summary_rows,
            ranking_fidelity_rows,
            triage_workflow_rows,
            tolerance_rows,
            seed_budget_rows,
            sequential_decision_rows,
            tsp_fast_tail_summary_rows,
            zdt1_spread_validation_rows,
            stress_case_catalog_rows,
            tail_risk_summary_rows,
            failure_trace_rows,
            failure_hypothesis_rows,
        ),
        encoding="utf-8",
    )
    _write_json(study_dir / "study_results.json", bundle)

    return {
        "study_dir": str(study_dir.resolve()),
        "summary_csv": str((study_dir / "summary.csv").resolve()),
        "summary_md": str((study_dir / "summary.md").resolve()),
        "case_group_summary_csv": str((study_dir / "case_group_summary.csv").resolve())
        if case_group_summary_rows
        else None,
        "ranking_detail_csv": str((study_dir / "ranking_detail.csv").resolve())
        if ranking_detail_rows
        else None,
        "ranking_fidelity_csv": str((study_dir / "ranking_fidelity.csv").resolve())
        if ranking_fidelity_rows
        else None,
        "triage_workflow_summary_csv": str(
            (study_dir / "triage_workflow_summary.csv").resolve()
        )
        if triage_workflow_rows
        else None,
        "qf_pair_rows_csv": str((study_dir / "qf_pair_rows.csv").resolve())
        if qf_pair_rows
        else None,
        "tolerance_table_csv": str((study_dir / "tolerance_table.csv").resolve())
        if tolerance_rows
        else None,
        "seed_budget_table_csv": str((study_dir / "seed_budget_table.csv").resolve())
        if seed_budget_rows
        else None,
        "sequential_decision_table_csv": str(
            (study_dir / "sequential_decision_table.csv").resolve()
        )
        if sequential_decision_rows
        else None,
        "tsp_fast_tail_rows_csv": str((study_dir / "tsp_fast_tail_rows.csv").resolve())
        if tsp_fast_tail_pair_rows
        else None,
        "tsp_fast_tail_summary_csv": str(
            (study_dir / "tsp_fast_tail_summary.csv").resolve()
        )
        if tsp_fast_tail_summary_rows
        else None,
        "zdt1_fast_hardening_rows_csv": str(
            (study_dir / "zdt1_fast_hardening_rows.csv").resolve()
        )
        if zdt1_fast_hardening_pair_rows
        else None,
        "zdt1_fast_hardening_summary_csv": str(
            (study_dir / "zdt1_fast_hardening_summary.csv").resolve()
        )
        if zdt1_fast_hardening_summary_rows
        else None,
        "zdt1_spread_candidate_validation_summary_csv": str(
            (study_dir / "zdt1_spread_candidate_validation_summary.csv").resolve()
        )
        if zdt1_spread_validation_rows
        else None,
        "zdt1_spread_candidate_boundary_table_csv": str(
            (study_dir / "zdt1_spread_candidate_boundary_table.csv").resolve()
        )
        if zdt1_spread_boundary_rows
        else None,
        "zdt1_spread_candidate_boundary_notes_md": str(
            (study_dir / "zdt1_spread_candidate_boundary_notes.md").resolve()
        )
        if zdt1_spread_boundary_rows
        else None,
        "stress_case_catalog_csv": str((study_dir / "stress_case_catalog.csv").resolve())
        if stress_case_catalog_rows
        else None,
        "stress_case_catalog_md": str((study_dir / "stress_case_catalog.md").resolve())
        if stress_case_catalog_rows
        else None,
        "tail_risk_summary_csv": str((study_dir / "tail_risk_summary.csv").resolve())
        if tail_risk_summary_rows
        else None,
        "failure_trace_table_csv": str((study_dir / "failure_trace_table.csv").resolve())
        if failure_trace_rows
        else None,
        "failure_hypotheses_csv": str((study_dir / "failure_hypotheses.csv").resolve())
        if failure_hypothesis_rows
        else None,
        "tsp_irreducible_tail_freeze_summary_md": str(
            (study_dir / "tsp_irreducible_tail_freeze_summary.md").resolve()
        )
        if _tsp_tail_freeze_config(study) is not None and tsp_fast_tail_summary_rows
        else None,
        "tsp_protocol_limitation_freeze_summary_md": str(
            (study_dir / "tsp_protocol_limitation_freeze_summary.md").resolve()
        )
        if _tsp_tail_freeze_config(study) is not None and tsp_fast_tail_summary_rows
        else None,
        "tail_risk_reduction_summary_csv": str(
            (study_dir / "tail_risk_reduction_summary.csv").resolve()
        )
        if tail_risk_summary_rows and _stress_target_reduction_config(study) is not None
        else None,
        "stress_suite_notes_md": str((study_dir / "stress_suite_notes.md").resolve())
        if stress_notes_enabled and (stress_case_catalog_rows or tail_risk_summary_rows)
        else None,
        "raw_results_csv": str((study_dir / "raw_results.csv").resolve()),
        "history_summary_csv": str((study_dir / "history_summary.csv").resolve()),
        "plots": plots,
        "best_variant": best_variant,
    }


def rerender_local_study(study_dir: str | Path) -> dict[str, Any]:
    root = Path(study_dir).resolve()
    bundle = _load_json_dict(root / "study_results.json")
    study = load_local_study(bundle["study"]["source_path"])
    raw_rows = list(bundle.get("raw_runs", []))
    summary_rows = list(bundle.get("summary_rows", []))
    case_group_summary_rows = list(bundle.get("case_group_summary_rows", []))
    ranking_detail_rows = list(bundle.get("ranking_detail_rows", []))
    ranking_fidelity_rows = list(bundle.get("ranking_fidelity_rows", []))
    triage_workflow_rows = list(bundle.get("triage_workflow_rows", []))
    qf_pair_rows = list(bundle.get("qf_pair_rows", []))
    tolerance_rows = list(bundle.get("tolerance_rows", []))
    seed_budget_rows = list(bundle.get("seed_budget_rows", []))
    sequential_decision_rows = list(bundle.get("sequential_decision_rows", []))
    tsp_fast_tail_pair_rows = list(bundle.get("tsp_fast_tail_pair_rows", []))
    tsp_fast_tail_summary_rows = list(bundle.get("tsp_fast_tail_summary_rows", []))
    zdt1_fast_hardening_pair_rows = list(bundle.get("zdt1_fast_hardening_pair_rows", []))
    zdt1_fast_hardening_summary_rows = list(bundle.get("zdt1_fast_hardening_summary_rows", []))
    zdt1_spread_validation_rows = list(bundle.get("zdt1_spread_validation_rows", []))
    zdt1_spread_boundary_rows = list(bundle.get("zdt1_spread_boundary_rows", []))
    if not zdt1_spread_boundary_rows and zdt1_spread_validation_rows:
        zdt1_spread_boundary_rows = _zdt1_spread_boundary_rows(
            study,
            zdt1_spread_validation_rows,
        )
        bundle["zdt1_spread_boundary_rows"] = zdt1_spread_boundary_rows
    stress_case_catalog_rows = list(bundle.get("stress_case_catalog_rows", []))
    tail_risk_summary_rows = list(bundle.get("tail_risk_summary_rows", []))
    failure_trace_rows = list(bundle.get("failure_trace_rows", []))
    failure_hypothesis_rows = list(bundle.get("failure_hypothesis_rows", []))
    history_rows = list(bundle.get("history_rows", []))
    if not tail_risk_summary_rows:
        if tsp_fast_tail_summary_rows:
            tail_risk_summary_rows = list(tsp_fast_tail_summary_rows)
        elif zdt1_spread_validation_rows:
            tail_risk_summary_rows = list(zdt1_spread_validation_rows)
        elif zdt1_fast_hardening_summary_rows:
            tail_risk_summary_rows = list(zdt1_fast_hardening_summary_rows)
    plots = _study_plot_outputs(
        study,
        raw_rows,
        summary_rows,
        history_rows,
        case_group_summary_rows,
        ranking_detail_rows,
        ranking_fidelity_rows,
        triage_workflow_rows,
        qf_pair_rows,
        tolerance_rows,
        seed_budget_rows,
        sequential_decision_rows,
        tsp_fast_tail_pair_rows,
        tsp_fast_tail_summary_rows,
        zdt1_spread_validation_rows,
        stress_case_catalog_rows,
        tail_risk_summary_rows,
        failure_trace_rows,
        root,
    )
    bundle["plots"] = plots
    (root / "summary.md").write_text(
        _render_study_markdown(
            study,
            summary_rows,
            bundle.get("best_variant"),
            case_group_summary_rows,
            ranking_fidelity_rows,
            triage_workflow_rows,
            tolerance_rows,
            seed_budget_rows,
            sequential_decision_rows,
            tsp_fast_tail_summary_rows,
            zdt1_spread_validation_rows,
            stress_case_catalog_rows,
            tail_risk_summary_rows,
            failure_trace_rows,
            failure_hypothesis_rows,
        ),
        encoding="utf-8",
    )
    if _tsp_tail_freeze_config(study) is not None and tsp_fast_tail_summary_rows:
        tail_freeze_summary = _render_tsp_tail_freeze_summary(
            study,
            tsp_fast_tail_summary_rows,
            failure_hypothesis_rows,
        )
        (root / "tsp_irreducible_tail_freeze_summary.md").write_text(
            tail_freeze_summary,
            encoding="utf-8",
        )
        (root / "tsp_protocol_limitation_freeze_summary.md").write_text(
            tail_freeze_summary,
            encoding="utf-8",
        )
    if zdt1_spread_boundary_rows:
        _write_rows_csv(
            root / "zdt1_spread_candidate_boundary_table.csv",
            zdt1_spread_boundary_rows,
        )
        (root / "zdt1_spread_candidate_boundary_notes.md").write_text(
            _render_zdt1_spread_boundary_notes(
                study,
                zdt1_spread_boundary_rows,
            ),
            encoding="utf-8",
        )
    if stress_case_catalog_rows:
        (root / "stress_case_catalog.md").write_text(
            _render_stress_case_catalog_markdown(study, stress_case_catalog_rows),
            encoding="utf-8",
        )
    stress_notes_enabled = _stress_suite_config(study) is not None
    if stress_notes_enabled and (stress_case_catalog_rows or tail_risk_summary_rows):
        (root / "stress_suite_notes.md").write_text(
            _render_stress_suite_notes(study, stress_case_catalog_rows, tail_risk_summary_rows),
            encoding="utf-8",
        )
    _write_json(root / "study_results.json", bundle)
    return {"study_dir": str(root), "plots": plots}
