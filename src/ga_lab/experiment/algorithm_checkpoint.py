from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

try:  # NumPy is optional for this helper; the current GA path only uses Python RNG.
    import numpy as _np
except Exception:  # pragma: no cover - depends on local environment
    _np = None

from ga_lab.config import GAConfig

SCHEMA_VERSION = "algorithm_checkpoint_phase1_v1"
CHECKPOINT_TYPE = "single_objective_ga"
ALGORITHM_NAME = "single_objective_ga"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(slots=True)
class CheckpointConfig:
    enabled: bool
    output_dir: str | Path
    run_id: str
    interval_generations: int = 1
    resume_from: str | Path | None = None
    write_latest: bool = False
    allow_overwrite: bool = False
    stop_after_checkpoint_generation: int | None = None
    artifact_suffix: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("CheckpointConfig.enabled must be a boolean")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("CheckpointConfig.run_id must be a non-empty string")
        if not _SAFE_RUN_ID.fullmatch(self.run_id):
            raise ValueError("CheckpointConfig.run_id contains unsafe characters")
        if self.interval_generations <= 0:
            raise ValueError("CheckpointConfig.interval_generations must be positive")
        self.output_dir = Path(self.output_dir)
        if self.resume_from is not None:
            self.resume_from = Path(self.resume_from)


@dataclass(slots=True)
class CheckpointMetadata:
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
    completed: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PopulationCheckpoint:
    decision_vectors: list[list[float]]
    fitness_values: list[float]
    population_size: int
    best_index: int | None
    best_fitness: float | None
    objective_direction: bool
    finite_validation_status: str


@dataclass(slots=True)
class RNGCheckpoint:
    python_random_state: Any | None
    numpy_random_state: Any | None = None
    algorithm_rng_state: Any | None = None
    rng_capture_complete: bool = True
    rng_warning: str | None = None


@dataclass(slots=True)
class EvaluationBudgetState:
    requested_budget: int
    actual_evaluations: int
    remaining_budget: int
    generation_index: int
    evaluations_per_generation: int | None = None
    budget_policy: str = "configured_evaluation_budget"


@dataclass(slots=True)
class AlgorithmCheckpointState:
    metadata: CheckpointMetadata
    population: PopulationCheckpoint
    rng: RNGCheckpoint
    budget_state: EvaluationBudgetState
    resume_generation: int
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ResumeCompatibilityReport:
    schema_version_compatible: bool
    algorithm_match: bool
    problem_match: bool
    dimension_match: bool
    bounds_match: bool
    objective_direction_match: bool
    config_hash_match: bool
    operator_signature_match: bool
    mutation_contract_match: bool
    budget_compatible: bool
    rng_state_available: bool
    checkpoint_finite: bool
    population_size_compatible: bool
    decision: str
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
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


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_config_hash(config: GAConfig) -> str:
    return sha256(_canonical_json(config.to_canonical_dict()).encode("utf-8")).hexdigest()


def summarize_operator_signature(config: GAConfig) -> dict[str, Any]:
    return {
        "selection": config.operators_spec.selection.to_dict(),
        "crossover": config.operators_spec.crossover.to_dict(),
        "mutation": config.operators_spec.mutation.to_dict(),
        "rates": {
            "mutation_rate": config.mutation_rate,
            "crossover_rate": config.crossover_rate,
            "elitism": config.elitism,
            "population_size": config.population_size,
        },
        "mutation_contract": "mutation_fn_genome_rng_returns_genome",
    }


def summarize_problem_signature(config: GAConfig, problem: object | None = None) -> dict[str, Any]:
    return {
        "problem": str(getattr(problem, "name", config.problem)),
        "configured_problem": config.problem,
        "dimension": config.genome_length,
        "representation": config.representation,
        "representation_options": dict(config.representation_options),
        "problem_options": dict(config.problem_options),
        "objective_direction": bool(config.maximize),
    }


def capture_rng_state(rng: Any) -> RNGCheckpoint:
    warnings: list[str] = []
    python_state = None
    if hasattr(rng, "getstate"):
        python_state = _jsonable(rng.getstate())
    else:
        warnings.append("Python RNG state is unavailable")

    numpy_state = None
    if _np is not None:
        try:
            numpy_raw = _np.random.get_state()
            numpy_state = _jsonable(numpy_raw)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"NumPy RNG state capture failed: {exc}")

    return RNGCheckpoint(
        python_random_state=python_state,
        numpy_random_state=numpy_state,
        algorithm_rng_state=None,
        rng_capture_complete=not warnings,
        rng_warning="; ".join(warnings) if warnings else None,
    )


def restore_rng_state(rng: Any, checkpoint: RNGCheckpoint) -> list[str]:
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
            numpy_state = tuple(numpy_state)
            _np.random.set_state(numpy_state)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"NumPy RNG state restore failed: {exc}")
    return warnings


def _finite_values(values: list[float]) -> bool:
    return all(isinstance(value, int | float) and math.isfinite(float(value)) for value in values)


def checkpoint_to_dict(checkpoint: AlgorithmCheckpointState) -> dict[str, Any]:
    return {
        "schema_version": checkpoint.metadata.schema_version,
        "metadata": asdict(checkpoint.metadata),
        "population": asdict(checkpoint.population),
        "rng": asdict(checkpoint.rng),
        "budget_state": asdict(checkpoint.budget_state),
        "resume_generation": checkpoint.resume_generation,
        "history": checkpoint.history,
    }


def checkpoint_from_dict(payload: dict[str, Any]) -> AlgorithmCheckpointState:
    metadata = CheckpointMetadata(**payload["metadata"])
    population = PopulationCheckpoint(**payload["population"])
    rng = RNGCheckpoint(**payload["rng"])
    budget_state = EvaluationBudgetState(**payload["budget_state"])
    return AlgorithmCheckpointState(
        metadata=metadata,
        population=population,
        rng=rng,
        budget_state=budget_state,
        resume_generation=int(payload.get("resume_generation", metadata.generation_index + 1)),
        history=[dict(row) for row in payload.get("history", [])],
    )


def load_checkpoint(path: str | Path) -> AlgorithmCheckpointState:
    checkpoint_path = Path(path)
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint = checkpoint_from_dict(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"corrupted checkpoint: {checkpoint_path}") from exc
    if checkpoint.metadata.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {checkpoint.metadata.schema_version}")
    if not _finite_values(checkpoint.population.fitness_values):
        raise ValueError("checkpoint contains non-finite fitness values")
    return checkpoint


def validate_resume_compatibility(
    checkpoint: AlgorithmCheckpointState,
    *,
    config: GAConfig,
    problem: object | None = None,
    requested_budget: int | None = None,
) -> ResumeCompatibilityReport:
    failures: list[str] = []
    warnings: list[str] = []

    schema_version_compatible = checkpoint.metadata.schema_version == SCHEMA_VERSION
    if not schema_version_compatible:
        failures.append("schema_version mismatch")

    algorithm_match = (
        checkpoint.metadata.checkpoint_type == CHECKPOINT_TYPE
        and checkpoint.metadata.algorithm == ALGORITHM_NAME
    )
    if not algorithm_match:
        failures.append("algorithm/checkpoint_type mismatch")

    current_problem_signature = summarize_problem_signature(config, problem)
    problem_match = (
        checkpoint.metadata.problem == current_problem_signature["problem"]
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

    objective_direction_match = checkpoint.population.objective_direction == config.maximize
    if not objective_direction_match:
        failures.append("objective_direction mismatch")

    config_hash_match = checkpoint.metadata.config_hash == build_config_hash(config)
    if not config_hash_match:
        failures.append("config_hash mismatch")

    operator_signature_match = (
        checkpoint.metadata.operator_signature_summary == summarize_operator_signature(config)
    )
    if not operator_signature_match:
        failures.append("operator_signature mismatch")

    mutation_contract_match = (
        checkpoint.metadata.operator_signature_summary.get("mutation_contract")
        == "mutation_fn_genome_rng_returns_genome"
    )
    if not mutation_contract_match:
        failures.append("mutation_contract marker mismatch")

    budget_value = requested_budget if requested_budget is not None else checkpoint.metadata.requested_budget
    budget_compatible = int(budget_value) >= checkpoint.metadata.actual_evaluations
    if not budget_compatible:
        failures.append("budget smaller than checkpoint actual_evaluations")

    rng_state_available = checkpoint.rng.python_random_state is not None
    if not rng_state_available:
        warnings.append("RNG state missing; deterministic true resume unavailable")

    checkpoint_finite = _finite_values(checkpoint.population.fitness_values)
    if not checkpoint_finite:
        failures.append("checkpoint fitness is not finite")

    population_size_compatible = checkpoint.population.population_size == config.population_size
    if not population_size_compatible:
        failures.append("population_size mismatch")

    if failures:
        decision = "fail"
    elif warnings:
        decision = "warning"
    else:
        decision = "pass"

    return ResumeCompatibilityReport(
        schema_version_compatible=schema_version_compatible,
        algorithm_match=algorithm_match,
        problem_match=problem_match,
        dimension_match=dimension_match,
        bounds_match=bounds_match,
        objective_direction_match=objective_direction_match,
        config_hash_match=config_hash_match,
        operator_signature_match=operator_signature_match,
        mutation_contract_match=mutation_contract_match,
        budget_compatible=budget_compatible,
        rng_state_available=rng_state_available,
        checkpoint_finite=checkpoint_finite,
        population_size_compatible=population_size_compatible,
        decision=decision,
        warnings=warnings,
        failures=failures,
    )


def checkpoint_path(config: CheckpointConfig, generation_index: int) -> Path:
    return Path(config.output_dir) / config.run_id / f"checkpoint_gen_{generation_index}.json"


def write_checkpoint_atomic(
    checkpoint: AlgorithmCheckpointState,
    config: CheckpointConfig,
) -> Path:
    target = checkpoint_path(config, checkpoint.metadata.generation_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not config.allow_overwrite:
        raise FileExistsError(f"checkpoint already exists: {target}")

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

    if config.write_latest:
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


def write_resume_compatibility_report(
    report: ResumeCompatibilityReport,
    *,
    output_dir: str | Path,
    run_id: str,
    suffix: str,
    allow_overwrite: bool = False,
) -> Path:
    target_dir = Path(output_dir) / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"resume_compatibility_report_{suffix}.md"
    if target.exists() and not allow_overwrite:
        raise FileExistsError(f"resume compatibility report already exists: {target}")
    lines = [
        "# Resume Compatibility Report",
        "",
        f"- decision: {report.decision}",
        f"- warnings: {len(report.warnings)}",
        f"- failures: {len(report.failures)}",
        "",
        "## Warnings",
        "",
        *(f"- {warning}" for warning in report.warnings),
        "",
        "## Failures",
        "",
        *(f"- {failure}" for failure in report.failures),
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
