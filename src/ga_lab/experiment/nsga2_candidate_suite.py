from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.experiment.mo_metrics import reference_front_for_problem


@dataclass(frozen=True, slots=True)
class MOBenchmarkSpec:
    problem: str
    variables: int
    objectives: int
    bounds: tuple[float, float]
    hv_reference_point: tuple[float, ...]
    reference_front_name: str
    notes: str
    problem_options: dict[str, object]


def mo_candidate_suite_specs() -> dict[str, MOBenchmarkSpec]:
    return {
        "zdt1": MOBenchmarkSpec(
            problem="zdt1",
            variables=6,
            objectives=2,
            bounds=(0.0, 1.0),
            hv_reference_point=(1.05, 10.5),
            reference_front_name="analytic_zdt1",
            notes="convex smooth 2-objective minimization front",
            problem_options={},
        ),
        "zdt2": MOBenchmarkSpec(
            problem="zdt2",
            variables=6,
            objectives=2,
            bounds=(0.0, 1.0),
            hv_reference_point=(1.05, 10.5),
            reference_front_name="analytic_zdt2",
            notes="non-convex smooth 2-objective minimization front",
            problem_options={},
        ),
        "zdt3": MOBenchmarkSpec(
            problem="zdt3",
            variables=6,
            objectives=2,
            bounds=(0.0, 1.0),
            hv_reference_point=(1.05, 10.5),
            reference_front_name="analytic_zdt3_disconnected",
            notes="disconnected 2-objective minimization front",
            problem_options={},
        ),
        "dtlz2": MOBenchmarkSpec(
            problem="dtlz2",
            variables=6,
            objectives=2,
            bounds=(0.0, 1.0),
            hv_reference_point=(1.1, 1.1),
            reference_front_name="analytic_dtlz2_m2",
            notes="2-objective DTLZ2 smoke front on quarter unit circle",
            problem_options={"objective_count": 2},
        ),
        "dtlz3": MOBenchmarkSpec(
            problem="dtlz3",
            variables=6,
            objectives=2,
            bounds=(0.0, 1.0),
            hv_reference_point=(2.0, 2.0),
            reference_front_name="analytic_dtlz3_m2",
            notes="2-objective DTLZ3 multimodal smoke front sharing the DTLZ2 Pareto manifold",
            problem_options={"objective_count": 2},
        ),
        "dtlz4": MOBenchmarkSpec(
            problem="dtlz4",
            variables=6,
            objectives=2,
            bounds=(0.0, 1.0),
            hv_reference_point=(1.1, 1.1),
            reference_front_name="analytic_dtlz4_m2",
            notes="2-objective DTLZ4 bias smoke front sharing the DTLZ2 Pareto manifold",
            problem_options={"objective_count": 2},
        ),
        "wfg1": MOBenchmarkSpec(
            problem="wfg1",
            variables=6,
            objectives=2,
            bounds=(0.0, 1.0),
            hv_reference_point=(2.5, 4.5),
            reference_front_name="pymoo_wfg1_smoke_front",
            notes=(
                "2-objective WFG1 smoke using a pymoo-backed normalized evaluator; "
                "reference front and distance metrics are smoke-level only"
            ),
            problem_options={"objective_count": 2, "variable_count": 6},
        ),
        "wfg2": MOBenchmarkSpec(
            problem="wfg2",
            variables=6,
            objectives=2,
            bounds=(0.0, 1.0),
            hv_reference_point=(2.5, 4.5),
            reference_front_name="pymoo_wfg2_smoke_front",
            notes=(
                "2-objective WFG2 smoke using a pymoo-backed normalized evaluator; "
                "reference front and distance metrics are smoke-level only"
            ),
            problem_options={"objective_count": 2, "variable_count": 6},
        ),
    }


def build_problem_config(
    base_config: GAConfig,
    spec: MOBenchmarkSpec,
) -> GAConfig:
    clone = GAConfig.from_dict(base_config.to_dict())
    clone.problem = spec.problem
    clone.problem_options = deepcopy(spec.problem_options)
    clone.genome_length = spec.variables
    clone.objective_directions = [False for _ in range(spec.objectives)]
    clone.representation_options = {"low": spec.bounds[0], "high": spec.bounds[1]}
    clone.algorithm_options = deepcopy(clone.algorithm_options)
    clone.algorithm_options["hypervolume_reference_point"] = list(spec.hv_reference_point)
    clone.run_name = f"{base_config.run_name}_{spec.problem}"
    return clone


def reference_front_for_spec(spec: MOBenchmarkSpec, *, point_count: int = 201) -> list[list[float]]:
    return reference_front_for_problem(
        spec.problem,
        point_count=point_count,
        objective_count=spec.objectives,
        variable_count=spec.variables,
    )


def safe_artifact_path(root: Path, base_name: str, suffix: str | None, extension: str) -> Path:
    if suffix:
        safe_suffix = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in suffix.strip()
        ).strip("_")
        if safe_suffix:
            candidate = root / f"{base_name}_{safe_suffix}{extension}"
            if not candidate.exists():
                return candidate
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            return root / f"{base_name}_{safe_suffix}_{timestamp}{extension}"
    candidate = root / f"{base_name}{extension}"
    if candidate.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return root / f"{base_name}_{timestamp}{extension}"
    return candidate
