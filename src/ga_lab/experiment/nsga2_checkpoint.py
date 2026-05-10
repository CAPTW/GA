from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # NumPy remains optional for checkpoint metadata.
    import numpy as _np
except Exception:  # pragma: no cover - depends on local environment
    _np = None

from ga_lab.config import GAConfig
from ga_lab.experiment.algorithm_checkpoint import (
    build_config_hash,
    summarize_operator_signature,
)

NSGA2_SCHEMA_VERSION = "nsga2_checkpoint_phase2a_v1"
NSGA2_CHECKPOINT_TYPE = "nsga2_checkpoint"
NSGA2_ALGORITHM_NAME = "nsga2"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(slots=True)
class Nsga2CheckpointMetadata:
    schema_version: str
    checkpoint_type: str
    algorithm: str
    problem: str
    seed: int
    created_at: str
    generation_index: int
    actual_evaluations: int
    requested_budget: int
    config_hash: str
    operator_signature_summary: dict[str, Any]
    problem_signature_summary: dict[str, Any]
    objective_count: int
    objective_directions: list[bool]
    completed: bool
    warnings: list[str] = field(default_factory=list)
    rank_crowding_policy: str = "cache_recompute_on_resume"


@dataclass(slots=True)
class Nsga2PopulationCheckpoint:
    decision_vectors: list[list[float]]
    objective_values: list[list[float]]
    population_size: int
    objective_count: int
    ranks: list[int] | None = None
    crowding_distances: list[float | str] | None = None
    front_indices: list[list[int]] | None = None
    feasible_mask: list[bool] | None = None
    finite_validation_status: str = "unknown"


@dataclass(slots=True)
class Nsga2SelectionState:
    last_parent_indices: list[int] | None = None
    last_offspring_count: int | None = None
    survival_fronts: list[list[int]] | None = None
    boundary_points: list[int] | None = None


@dataclass(slots=True)
class Nsga2RNGCheckpoint:
    python_random_state: Any | None
    numpy_random_state: Any | None = None
    algorithm_rng_state: Any | None = None
    rng_capture_complete: bool = True
    rng_warning: str | None = None


@dataclass(slots=True)
class Nsga2EvaluationBudgetState:
    requested_budget: int
    actual_evaluations: int
    remaining_budget: int
    generation_index: int
    evaluations_per_generation: int | None = None
    budget_policy: str = "configured_evaluation_budget"


@dataclass(slots=True)
class Nsga2CheckpointState:
    metadata: Nsga2CheckpointMetadata
    population: Nsga2PopulationCheckpoint
    rng: Nsga2RNGCheckpoint
    budget_state: Nsga2EvaluationBudgetState
    selection_state: Nsga2SelectionState = field(default_factory=Nsga2SelectionState)
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Nsga2ResumeCompatibilityReport:
    schema_version_compatible: bool
    checkpoint_type_match: bool
    algorithm_match: bool
    problem_match: bool
    dimension_match: bool
    bounds_match: bool
    objective_count_match: bool
    objective_directions_match: bool
    config_hash_match: bool
    operator_signature_match: bool
    population_size_compatible: bool
    budget_compatible: bool
    rng_state_available: bool
    checkpoint_finite: bool
    cached_rank_crowding_policy_present: bool
    decision: str
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Nsga2CachedFrontValidationReport:
    objective_shape_valid: bool
    rank_shape_valid: bool
    crowding_shape_valid: bool
    crowding_values_valid: bool
    front_indices_valid: bool
    decision: str
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _tupleize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    if isinstance(value, dict):
        return {key: _tupleize(item) for key, item in value.items()}
    return value


def build_nsga2_config_hash(config: GAConfig) -> str:
    return build_config_hash(config)


def summarize_nsga2_operator_signature(config: GAConfig) -> dict[str, Any]:
    signature = summarize_operator_signature(config)
    signature["algorithm"] = NSGA2_ALGORITHM_NAME
    return signature


def summarize_nsga2_problem_signature(
    config: GAConfig,
    *,
    problem: object | None = None,
    objective_count: int,
    objective_directions: list[bool],
) -> dict[str, Any]:
    return {
        "problem": str(getattr(problem, "name", config.problem)),
        "configured_problem": config.problem,
        "dimension": config.genome_length,
        "representation": config.representation,
        "representation_options": dict(config.representation_options),
        "problem_options": dict(config.problem_options),
        "objective_count": int(objective_count),
        "objective_directions": [bool(value) for value in objective_directions],
    }


def capture_nsga2_rng_state(rng: Any) -> Nsga2RNGCheckpoint:
    warnings: list[str] = []
    python_state = None
    if hasattr(rng, "getstate"):
        python_state = _jsonable(rng.getstate())
    else:
        warnings.append("Python RNG state is unavailable")

    numpy_state = None
    if _np is not None:
        try:
            numpy_state = _jsonable(_np.random.get_state())
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"NumPy RNG state capture failed: {exc}")

    return Nsga2RNGCheckpoint(
        python_random_state=python_state,
        numpy_random_state=numpy_state,
        algorithm_rng_state=None,
        rng_capture_complete=not warnings,
        rng_warning="; ".join(warnings) if warnings else None,
    )


def restore_nsga2_rng_state(rng: Any, checkpoint: Nsga2RNGCheckpoint) -> list[str]:
    warnings: list[str] = []
    if checkpoint.python_random_state is None:
        warnings.append("RNG state missing; deterministic true resume unavailable")
    elif hasattr(rng, "setstate"):
        rng.setstate(_tupleize(checkpoint.python_random_state))
    else:
        warnings.append("Target RNG cannot restore state")

    if _np is not None and checkpoint.numpy_random_state is not None:
        try:
            numpy_state = list(checkpoint.numpy_random_state)
            if len(numpy_state) >= 2:
                numpy_state[1] = _np.array(numpy_state[1], dtype="uint32")
            _np.random.set_state(tuple(numpy_state))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"NumPy RNG state restore failed: {exc}")
    return warnings


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_allowed_crowding_value(value: Any) -> bool:
    if isinstance(value, str):
        return value in {"inf", "-inf"}
    return isinstance(value, int | float) and not isinstance(value, bool) and (
        math.isfinite(float(value)) or math.isinf(float(value))
    )


def validate_nsga2_checkpoint_finite(checkpoint: Nsga2CheckpointState) -> None:
    for row in checkpoint.population.objective_values:
        for value in row:
            if not _is_finite_number(value):
                raise ValueError("NSGA-II checkpoint contains non-finite objective values")


def validate_cached_rank_crowding(
    *,
    objective_values: list[list[float]],
    cached_ranks: list[int] | None,
    cached_crowding_distances: list[float | str] | None,
    objective_directions: list[bool],
    front_indices: list[list[int]] | None = None,
) -> Nsga2CachedFrontValidationReport:
    failures: list[str] = []
    warnings: list[str] = []
    population_size = len(objective_values)
    objective_count = len(objective_directions)

    objective_shape_valid = bool(objective_values) and all(
        len(row) == objective_count and all(_is_finite_number(value) for value in row)
        for row in objective_values
    )
    if not objective_shape_valid:
        failures.append("objective shape/non-finite mismatch")

    rank_shape_valid = cached_ranks is None or len(cached_ranks) == population_size
    if not rank_shape_valid:
        failures.append("rank length mismatch")
    if cached_ranks is not None and any(rank < 0 for rank in cached_ranks):
        failures.append("negative rank value")

    crowding_shape_valid = (
        cached_crowding_distances is None or len(cached_crowding_distances) == population_size
    )
    if not crowding_shape_valid:
        failures.append("crowding length mismatch")
    crowding_values_valid = cached_crowding_distances is None or all(
        _is_allowed_crowding_value(value) for value in cached_crowding_distances
    )
    if not crowding_values_valid:
        failures.append("crowding value invalid")

    front_indices_valid = True
    if front_indices is not None:
        seen: set[int] = set()
        for front in front_indices:
            for index in front:
                if index < 0 or index >= population_size:
                    front_indices_valid = False
                if index in seen:
                    warnings.append("front index repeated")
                seen.add(index)
    if not front_indices_valid:
        failures.append("front index out of range")

    decision = "fail" if failures else ("warning" if warnings else "pass")
    return Nsga2CachedFrontValidationReport(
        objective_shape_valid=objective_shape_valid,
        rank_shape_valid=rank_shape_valid,
        crowding_shape_valid=crowding_shape_valid,
        crowding_values_valid=crowding_values_valid,
        front_indices_valid=front_indices_valid,
        decision=decision,
        warnings=warnings,
        failures=failures,
    )


def checkpoint_to_dict(checkpoint: Nsga2CheckpointState) -> dict[str, Any]:
    validate_nsga2_checkpoint_finite(checkpoint)
    return {
        "schema_version": checkpoint.metadata.schema_version,
        "metadata": _jsonable(asdict(checkpoint.metadata)),
        "population": _jsonable(asdict(checkpoint.population)),
        "rng": _jsonable(asdict(checkpoint.rng)),
        "budget_state": _jsonable(asdict(checkpoint.budget_state)),
        "selection_state": _jsonable(asdict(checkpoint.selection_state)),
        "history": _jsonable(checkpoint.history),
    }


def build_nsga2_checkpoint_state_from_run_state(
    *,
    config: GAConfig,
    problem: object,
    rng: Any,
    generation_index: int,
    actual_evaluations: int,
    requested_budget: int | None,
    population: list[list[float]],
    objective_values: list[list[float]],
    objective_directions: list[bool],
    ranks: list[int] | None,
    crowding_distances: list[float] | None,
    front_indices: list[list[int]] | None,
    history: list[dict[str, Any]] | None = None,
    completed: bool = False,
) -> Nsga2CheckpointState:
    objective_count = len(objective_directions)
    budget_value = int(requested_budget) if requested_budget is not None else int(actual_evaluations)
    metadata = Nsga2CheckpointMetadata(
        schema_version=NSGA2_SCHEMA_VERSION,
        checkpoint_type=NSGA2_CHECKPOINT_TYPE,
        algorithm=NSGA2_ALGORITHM_NAME,
        problem=str(getattr(problem, "name", config.problem)),
        seed=int(config.seed) if isinstance(config.seed, int) else 0,
        created_at=utc_timestamp(),
        generation_index=int(generation_index),
        actual_evaluations=int(actual_evaluations),
        requested_budget=budget_value,
        config_hash=build_nsga2_config_hash(config),
        operator_signature_summary=summarize_nsga2_operator_signature(config),
        problem_signature_summary=summarize_nsga2_problem_signature(
            config,
            problem=problem,
            objective_count=objective_count,
            objective_directions=objective_directions,
        ),
        objective_count=objective_count,
        objective_directions=[bool(value) for value in objective_directions],
        completed=completed,
    )
    population_state = Nsga2PopulationCheckpoint(
        decision_vectors=[[float(gene) for gene in genome] for genome in population],
        objective_values=[[float(value) for value in row] for row in objective_values],
        population_size=len(population),
        objective_count=objective_count,
        ranks=list(ranks) if ranks is not None else None,
        crowding_distances=list(crowding_distances) if crowding_distances is not None else None,
        front_indices=[list(front) for front in front_indices] if front_indices is not None else None,
        feasible_mask=None,
        finite_validation_status="pass",
    )
    budget_state = Nsga2EvaluationBudgetState(
        requested_budget=budget_value,
        actual_evaluations=int(actual_evaluations),
        remaining_budget=max(0, budget_value - int(actual_evaluations)),
        generation_index=int(generation_index),
        evaluations_per_generation=None,
    )
    checkpoint = Nsga2CheckpointState(
        metadata=metadata,
        population=population_state,
        rng=capture_nsga2_rng_state(rng),
        budget_state=budget_state,
        history=[dict(row) for row in (history or [])],
    )
    validate_nsga2_checkpoint_finite(checkpoint)
    return checkpoint


def checkpoint_from_dict(payload: dict[str, Any]) -> Nsga2CheckpointState:
    if payload.get("schema_version") != NSGA2_SCHEMA_VERSION:
        raise ValueError("unsupported NSGA-II checkpoint schema")
    metadata = Nsga2CheckpointMetadata(**payload["metadata"])
    population = Nsga2PopulationCheckpoint(**payload["population"])
    rng = Nsga2RNGCheckpoint(**payload["rng"])
    budget_state = Nsga2EvaluationBudgetState(**payload["budget_state"])
    selection_state = Nsga2SelectionState(**payload.get("selection_state", {}))
    checkpoint = Nsga2CheckpointState(
        metadata=metadata,
        population=population,
        rng=rng,
        budget_state=budget_state,
        selection_state=selection_state,
        history=[dict(row) for row in payload.get("history", [])],
    )
    validate_nsga2_checkpoint_finite(checkpoint)
    return checkpoint


def load_nsga2_checkpoint(path: str | Path) -> Nsga2CheckpointState:
    checkpoint_path = Path(path)
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return checkpoint_from_dict(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"corrupted NSGA-II checkpoint: {checkpoint_path}") from exc


def validate_nsga2_resume_compatibility(
    checkpoint: Nsga2CheckpointState,
    *,
    config: GAConfig,
    objective_count: int,
    objective_directions: list[bool],
    problem: object | None = None,
    requested_budget: int | None = None,
) -> Nsga2ResumeCompatibilityReport:
    failures: list[str] = []
    warnings: list[str] = []

    schema_version_compatible = checkpoint.metadata.schema_version == NSGA2_SCHEMA_VERSION
    if not schema_version_compatible:
        failures.append("schema_version mismatch")

    checkpoint_type_match = checkpoint.metadata.checkpoint_type == NSGA2_CHECKPOINT_TYPE
    if not checkpoint_type_match:
        failures.append("checkpoint_type mismatch")

    algorithm_match = checkpoint.metadata.algorithm == NSGA2_ALGORITHM_NAME
    if not algorithm_match:
        failures.append("algorithm mismatch")

    problem_signature = summarize_nsga2_problem_signature(
        config,
        problem=problem,
        objective_count=objective_count,
        objective_directions=objective_directions,
    )
    problem_match = (
        checkpoint.metadata.problem == problem_signature["problem"]
        and checkpoint.metadata.problem_signature_summary.get("configured_problem") == config.problem
    )
    if not problem_match:
        failures.append("problem mismatch")

    dimension_match = checkpoint.metadata.problem_signature_summary.get("dimension") == config.genome_length
    if not dimension_match:
        failures.append("dimension mismatch")

    bounds_match = (
        checkpoint.metadata.problem_signature_summary.get("representation_options")
        == dict(config.representation_options)
    )
    if not bounds_match:
        failures.append("bounds/representation options mismatch")

    objective_count_match = checkpoint.metadata.objective_count == int(objective_count)
    if not objective_count_match:
        failures.append("objective_count mismatch")

    objective_directions_match = checkpoint.metadata.objective_directions == [
        bool(value) for value in objective_directions
    ]
    if not objective_directions_match:
        failures.append("objective_directions mismatch")

    config_hash_match = checkpoint.metadata.config_hash == build_nsga2_config_hash(config)
    if not config_hash_match:
        failures.append("config_hash mismatch")

    operator_signature_match = (
        checkpoint.metadata.operator_signature_summary == summarize_nsga2_operator_signature(config)
    )
    if not operator_signature_match:
        failures.append("operator_signature mismatch")

    population_size_compatible = checkpoint.population.population_size == config.population_size
    if not population_size_compatible:
        failures.append("population_size mismatch")

    budget_value = requested_budget if requested_budget is not None else checkpoint.metadata.requested_budget
    budget_compatible = int(budget_value) >= int(checkpoint.metadata.actual_evaluations)
    if not budget_compatible:
        failures.append("budget smaller than checkpoint actual_evaluations")

    rng_state_available = checkpoint.rng.python_random_state is not None
    if not rng_state_available:
        failures.append("RNG state missing; deterministic true resume unavailable")

    try:
        validate_nsga2_checkpoint_finite(checkpoint)
        checkpoint_finite = True
    except ValueError:
        checkpoint_finite = False
        failures.append("checkpoint objective is not finite")

    cached_rank_crowding_policy_present = (
        checkpoint.metadata.rank_crowding_policy == "cache_recompute_on_resume"
    )
    if not cached_rank_crowding_policy_present:
        failures.append("cached rank/crowding policy missing")

    cache_report = validate_cached_rank_crowding(
        objective_values=checkpoint.population.objective_values,
        cached_ranks=checkpoint.population.ranks,
        cached_crowding_distances=checkpoint.population.crowding_distances,
        objective_directions=checkpoint.metadata.objective_directions,
        front_indices=checkpoint.population.front_indices,
    )
    if cache_report.decision == "fail":
        failures.extend(cache_report.failures)
    warnings.extend(cache_report.warnings)

    if failures:
        decision = "fail"
    elif warnings:
        decision = "warning"
    else:
        decision = "pass"

    return Nsga2ResumeCompatibilityReport(
        schema_version_compatible=schema_version_compatible,
        checkpoint_type_match=checkpoint_type_match,
        algorithm_match=algorithm_match,
        problem_match=problem_match,
        dimension_match=dimension_match,
        bounds_match=bounds_match,
        objective_count_match=objective_count_match,
        objective_directions_match=objective_directions_match,
        config_hash_match=config_hash_match,
        operator_signature_match=operator_signature_match,
        population_size_compatible=population_size_compatible,
        budget_compatible=budget_compatible,
        rng_state_available=rng_state_available,
        checkpoint_finite=checkpoint_finite,
        cached_rank_crowding_policy_present=cached_rank_crowding_policy_present,
        decision=decision,
        warnings=warnings,
        failures=failures,
    )


def _checkpoint_path(output_dir: str | Path, run_id: str, generation_index: int) -> Path:
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("NSGA-II checkpoint run_id contains unsafe characters")
    return Path(output_dir) / "nsga2" / run_id / f"checkpoint_gen_{generation_index}.json"


def write_nsga2_checkpoint_atomic(
    checkpoint: Nsga2CheckpointState,
    *,
    output_dir: str | Path,
    run_id: str,
    allow_overwrite: bool = False,
    write_latest: bool = False,
) -> Path:
    target = _checkpoint_path(output_dir, run_id, checkpoint.metadata.generation_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not allow_overwrite:
        raise FileExistsError(f"NSGA-II checkpoint already exists: {target}")

    payload = checkpoint_to_dict(checkpoint)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temp_path.replace(target)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise
    if write_latest:
        latest = target.parent / "latest.json"
        fd_latest, latest_temp_name = tempfile.mkstemp(
            prefix=".latest.json.",
            suffix=".tmp",
            dir=str(target.parent),
            text=True,
        )
        latest_temp_path = Path(latest_temp_name)
        try:
            with os.fdopen(fd_latest, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
            latest_temp_path.replace(latest)
        except Exception:
            try:
                latest_temp_path.unlink(missing_ok=True)
            finally:
                raise
    return target
