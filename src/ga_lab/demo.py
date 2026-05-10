from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ga_lab.config import GAConfig
from ga_lab.experiment.budget_baseline_comparison import run_manifests
from ga_lab.resources import (
    get_builtin_resource,
    materialize_builtin_resource,
    read_builtin_json,
)
from ga_lab.runner import run_experiment

DEMO_SPECS: dict[str, dict[str, str]] = {
    "baseline": {
        "mode": "comparison",
        "resource_kind": "demo-manifest",
        "resource_name": "baseline_onemax_demo_manifest",
        "solver_family": "baseline",
        "title": "Onemax baseline demo",
        "focus": "Hill climb versus the recommended pure GA preset on validated small onemax.",
    },
    "pure-ga": {
        "mode": "experiment",
        "resource_kind": "demo",
        "resource_name": "onemax_pure_ga_demo",
        "solver_family": "pure-ga",
        "title": "Pure GA demo",
        "focus": "A small onemax GA run that reaches the target quickly.",
    },
    "hybrid": {
        "mode": "experiment",
        "resource_kind": "demo",
        "resource_name": "tsp_medium_hybrid_demo",
        "solver_family": "hybrid-ga",
        "title": "Hybrid GA demo",
        "focus": "The current narrow official TSP quality-first hybrid path at medium size.",
    },
    "nsga2": {
        "mode": "experiment",
        "resource_kind": "demo",
        "resource_name": "zdt1_nsga2_demo",
        "solver_family": "nsga2",
        "title": "NSGA-II demo",
        "focus": "A small multi-objective smoke run with hypervolume and Pareto metrics.",
    },
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _materialize_demo_manifest(resource_name: str, demo_root: Path) -> Path:
    manifest_payload = read_builtin_json("demo-manifest", resource_name)
    inputs_root = demo_root / "_builtin_inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    for entry in manifest_payload.get("entries", []):
        raw_preset = entry.get("preset")
        if not isinstance(raw_preset, str):
            continue
        preset_name = Path(raw_preset).stem
        preset_path = materialize_builtin_resource(
            "preset",
            preset_name,
            inputs_root / "presets" / f"{preset_name}.json",
        )
        entry["preset"] = str(preset_path)
    manifest_path = inputs_root / f"{resource_name}.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def _baseline_demo_payload(
    summary: dict[str, Any],
    demo_root: Path,
    resource_name: str,
) -> dict[str, Any]:
    labels = {row["label"]: row for row in summary["aggregate_rows"]}
    baseline_row = labels["hill_climb"]
    ga_row = labels["recommended_preset"]
    resource_info = get_builtin_resource("demo-manifest", resource_name)
    payload = {
        "demo": "baseline",
        "title": DEMO_SPECS["baseline"]["title"],
        "solver_family": DEMO_SPECS["baseline"]["solver_family"],
        "focus": DEMO_SPECS["baseline"]["focus"],
        "manifest_resource": resource_info.resource_uri,
        "repo_manifest_path": resource_info.repo_relative_path,
        "summary_path": str((demo_root / "baseline_summary.json").resolve()),
        "markdown_summary_path": str((demo_root / "baseline_summary.md").resolve()),
        "configured_budget": baseline_row["configured_evaluation_budget"],
        "baseline_label": "hill_climb",
        "baseline_success_rate": baseline_row["success_rate"],
        "baseline_mean_evaluations_to_target": baseline_row["mean_evaluations_to_target"],
        "ga_success_rate": ga_row["success_rate"],
        "ga_mean_evaluations_to_target": ga_row["mean_evaluations_to_target"],
        "note": (
            "This demo uses the practical onemax default path. "
            "Hill climb should hit the target with fewer evaluations than the GA preset."
        ),
    }
    _write_json(demo_root / "demo_result.json", payload)
    return payload


def _experiment_demo_payload(
    *,
    demo_name: str,
    resource_name: str,
    demo_root: Path,
) -> dict[str, Any]:
    resource_info = get_builtin_resource("demo", resource_name)
    config = GAConfig.from_dict(read_builtin_json("demo", resource_name))
    result = run_experiment(config=config, output_root=demo_root)
    summary = result.summary
    payload = {
        "demo": demo_name,
        "title": DEMO_SPECS[demo_name]["title"],
        "solver_family": DEMO_SPECS[demo_name]["solver_family"],
        "focus": DEMO_SPECS[demo_name]["focus"],
        "config_resource": resource_info.resource_uri,
        "repo_config_path": resource_info.repo_relative_path,
        "output_dir": str(result.output_dir.resolve()),
        "summary_path": str((result.output_dir / "summary.json").resolve()),
        "runtime_seconds": summary.get("runtime_seconds"),
        "note": "",
    }

    if demo_name == "pure-ga":
        payload.update(
            {
                "best_fitness": summary.get("best_fitness"),
                "stop_reason": summary.get("stop_reason"),
                "convergence_generation": summary.get("convergence_generation"),
                "note": (
                    "This is a pure GA smoke run. On onemax, hill climb is still the practical "
                    "default; this demo exists to show the GA execution path."
                ),
            }
        )
    elif demo_name == "hybrid":
        payload.update(
            {
                "best_route_distance": summary.get("best_route_distance"),
                "hybrid_extra_evaluations": summary.get("hybrid_extra_evaluations"),
                "hybrid_local_search_improvements": summary.get(
                    "hybrid_local_search_improvements"
                ),
                "note": (
                    "Distance is the user-facing metric here. This remains a narrow internal "
                    "quality-first path, not the broad practical TSP default."
                ),
            }
        )
    elif demo_name == "nsga2":
        payload.update(
            {
                "hypervolume": summary.get("hypervolume"),
                "pareto_ratio": summary.get("pareto_ratio"),
                "spread": summary.get("spread"),
                "pareto_front_size": summary.get("pareto_front_size"),
                "note": (
                    "Read this demo through hypervolume and Pareto metrics rather than a single "
                    "best_fitness number."
                ),
            }
        )

    _write_json(demo_root / "demo_result.json", payload)
    return payload


def run_demo_payload(demo_name: str, output_root: Path) -> dict[str, Any]:
    spec = DEMO_SPECS[demo_name]
    demo_root = (output_root / demo_name.replace("-", "_")).resolve()
    demo_root.mkdir(parents=True, exist_ok=True)

    if spec["mode"] == "comparison":
        manifest_path = _materialize_demo_manifest(spec["resource_name"], demo_root)
        summary = run_manifests(
            [manifest_path],
            output_root=demo_root,
            summary_stem="baseline_summary",
        )
        return _baseline_demo_payload(summary, demo_root, spec["resource_name"])

    return _experiment_demo_payload(
        demo_name=demo_name,
        resource_name=spec["resource_name"],
        demo_root=demo_root,
    )
