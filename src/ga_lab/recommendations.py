"""Internal recommendation engine for validated portable solver choices.

Stable library callers should use ``ga_lab.api.recommend_preset`` and
``ga_lab.api.recommend_solver`` instead of importing this module directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ga_lab.project_paths import find_project_root
from ga_lab.resources import builtin_repo_relative_path, get_builtin_resource

PRESET_SIZE_RULES = {
    "onemax": {
        "size_key": "genome_length",
        "tiers": {
            32: ("small", "onemax_small"),
            64: ("medium", "onemax_medium"),
            128: ("large", "onemax_large"),
        },
        "priorities": {"default", "fast", "robust"},
    },
    "knapsack": {
        "size_key": "problem_options.num_items",
        "tiers": {
            20: ("small", "knapsack_small"),
            30: ("medium", "knapsack_medium"),
            80: ("large", "knapsack_large"),
        },
        "priorities": {"default"},
    },
    "tsp": {
        "size_key": "problem_options.num_cities",
        "tiers": {
            10: ("small", "tsp_small"),
            20: ("medium", "tsp_medium"),
            50: ("large", "tsp_large"),
        },
        "priorities": {"default"},
    },
    "zdt1": {
        "size_key": "genome_length",
        "tiers": {
            10: ("small", "zdt1_small"),
            20: ("medium", "zdt1_medium"),
            50: ("large", "zdt1_large"),
        },
        "priorities": {"default", "hv", "coverage"},
    },
}

VALIDATED_SIZES = {
    "onemax": {32: "small", 64: "medium", 128: "large"},
    "knapsack": {20: "small", 30: "medium", 80: "large"},
    "tsp": {10: "small", 20: "medium", 50: "large"},
    "zdt1": {10: "small", 20: "medium", 50: "large"},
}


def _resolved_repo_root(project_root: str | Path | None = None) -> Path | None:
    if project_root is not None:
        return Path(project_root).resolve()
    return find_project_root()


def _repo_abspath(repo_relative_path: str | None, project_root: Path | None) -> str | None:
    if repo_relative_path is None or project_root is None:
        return None
    return str((project_root / repo_relative_path).resolve())


def _preset_payload(name: str, project_root: Path | None) -> dict[str, str | None]:
    repo_relative_path = builtin_repo_relative_path("preset", name)
    resource = get_builtin_resource("preset", name)
    return {
        "preset_name": name,
        "preset_path": repo_relative_path,
        "preset_abspath": _repo_abspath(repo_relative_path, project_root),
        "resource_uri": resource.resource_uri,
        "portable_command_hint": f"ga-lab-run --preset {name}",
    }


def _config_payload(name: str, project_root: Path | None) -> dict[str, str | None]:
    repo_relative_path = builtin_repo_relative_path("preset", name)
    resource = get_builtin_resource("preset", name)
    return {
        "config_name": name,
        "config_path": repo_relative_path,
        "config_abspath": _repo_abspath(repo_relative_path, project_root),
        "resource_uri": resource.resource_uri,
        "portable_command_hint": f"ga-lab-run --preset {name}",
    }


def _build_preset_note(problem: str, size: int, tier: str, priority: str) -> str:
    if problem == "onemax" and size == 128:
        return (
            "Validated at genome_length=128. The same large preset currently wins for both "
            "practical speed and target-hit stability; the legacy pop80/gen40 large setup is "
            "no longer recommended for goal-seeking runs."
        )
    if problem == "zdt1" and size == 50:
        return (
            "Validated at genome_length=50. The same large preset currently leads both "
            "hypervolume-first and coverage/diversity-first checks; faster 80x160 alternatives "
            "exist, but they give up both hypervolume and pareto_ratio."
        )
    if problem == "knapsack" and size == 80:
        return (
            "Validated at num_items=80. Keep the current large preset unless you are explicitly "
            "trading more runtime for a small best_fitness gain."
        )
    if problem == "tsp" and size == 50:
        return (
            "Validated at num_cities=50. Keep the current large preset unless route quality is "
            "important enough to justify a higher-population retune."
        )
    return (
        f"Validated {tier} preset for {problem} at size {size}. Priority '{priority}' does not "
        "change the recommendation in the currently validated range."
    )


def recommend_preset(
    problem: str,
    size: int,
    priority: str,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    if problem not in PRESET_SIZE_RULES:
        raise ValueError(f"Unsupported problem: {problem}")
    rule = PRESET_SIZE_RULES[problem]
    if priority not in rule["priorities"]:
        allowed = ", ".join(sorted(rule["priorities"]))
        raise ValueError(f"Unsupported priority '{priority}' for {problem}. Allowed: {allowed}")
    if size not in rule["tiers"]:
        allowed_sizes = ", ".join(str(value) for value in sorted(rule["tiers"]))
        raise ValueError(
            f"Unsupported validated size '{size}' for {problem}. Validated sizes: {allowed_sizes}"
        )

    root = _resolved_repo_root(project_root)
    tier, preset_name = rule["tiers"][size]
    payload = {
        "problem": problem,
        "size": size,
        "size_key": rule["size_key"],
        "tier": tier,
        "priority": priority,
        "validated_sizes": sorted(rule["tiers"]),
        "note": _build_preset_note(problem, size, tier, priority),
    }
    payload.update(_preset_payload(preset_name, root))
    return payload


def _validated(problem: str, size: int) -> str:
    if problem not in VALIDATED_SIZES:
        raise ValueError(f"Unsupported problem: {problem}")
    if size not in VALIDATED_SIZES[problem]:
        allowed = ", ".join(str(value) for value in sorted(VALIDATED_SIZES[problem]))
        raise ValueError(
            f"Unsupported validated size '{size}' for {problem}. Validated sizes: {allowed}"
        )
    return VALIDATED_SIZES[problem][size]


def recommend_solver(
    problem: str,
    size: int,
    priority: str,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    tier = _validated(problem, size)
    validated_sizes = sorted(VALIDATED_SIZES[problem])
    root = _resolved_repo_root(project_root)

    if problem == "onemax":
        if priority == "ga":
            payload = {
                "problem": problem,
                "size": size,
                "tier": tier,
                "priority": priority,
                "solver_family": "pure-ga",
                "solver_name": "recommended_preset",
                "validated_sizes": validated_sizes,
                "note": (
                    "Use the pure GA preset for operator experiments or reproducible GA studies. "
                    "For practical target reaching on validated sizes, "
                    "hill climbing remains faster."
                ),
            }
            payload.update(_config_payload(f"onemax_{tier}", root))
            return payload
        return {
            "problem": problem,
            "size": size,
            "tier": tier,
            "priority": priority,
            "solver_family": "baseline",
            "solver_name": "hill_climb",
            "config_name": None,
            "config_path": None,
            "config_abspath": None,
            "resource_uri": None,
            "portable_command_hint": None,
            "validated_sizes": validated_sizes,
            "note": (
                "Hill climbing reaches the same validated onemax targets "
                "with far fewer evaluations than the pure GA preset."
            ),
        }

    if problem == "knapsack":
        if priority == "ga":
            payload = {
                "problem": problem,
                "size": size,
                "tier": tier,
                "priority": priority,
                "solver_family": "pure-ga",
                "solver_name": "recommended_preset",
                "validated_sizes": validated_sizes,
                "note": (
                    "The pure GA preset stays reproducible and competitive "
                    "against random sampling, but greedy local search "
                    "remains the practical default. Seed/repair hybrids are "
                    "still experimental, and the heavier local-improvement "
                    "hybrid is not recommended as a default path."
                ),
            }
            payload.update(_config_payload(f"knapsack_{tier}", root))
            return payload
        return {
            "problem": problem,
            "size": size,
            "tier": tier,
            "priority": priority,
            "solver_family": "baseline",
            "solver_name": "greedy_local_search",
            "config_name": None,
            "config_path": None,
            "config_abspath": None,
            "resource_uri": None,
            "portable_command_hint": None,
            "validated_sizes": validated_sizes,
            "note": (
                "Greedy local search is the practical default on validated "
                "knapsack sizes. Hybrid seed/repair closes the pure-GA gap "
                "but does not replace the cheap baseline as the default solver."
            ),
        }

    if problem == "tsp":
        if priority in {"quality", "ga"} and size == 20:
            payload = {
                "problem": problem,
                "size": size,
                "tier": tier,
                "priority": priority,
                "solver_family": "hybrid-ga",
                "solver_name": "tsp_medium_hybrid",
                "validated_sizes": validated_sizes,
                "note": (
                    "At validated medium TSP size (20 cities), the promoted "
                    "hybrid preset is the narrow quality-first official path. "
                    "Its edge over nearest-neighbor + 2-opt is small, and the "
                    "ablation study suggests nearest-neighbor seeding is the "
                    "main contributor."
                    if priority == "quality"
                    else "For validated medium TSP runs, the promoted hybrid "
                    "preset stays the official GA-family quality path, but "
                    "its advantage over the cheap baseline is small."
                ),
            }
            payload.update(_config_payload("tsp_medium_hybrid", root))
            return payload
        if priority == "ga":
            payload = {
                "problem": problem,
                "size": size,
                "tier": tier,
                "priority": priority,
                "solver_family": "pure-ga",
                "solver_name": "recommended_preset",
                "validated_sizes": validated_sizes,
                "note": (
                    "The pure GA preset remains available for GA-only "
                    "studies, but the practical TSP default is still "
                    "nearest-neighbor plus 2-opt outside the promoted "
                    "medium hybrid case."
                ),
            }
            payload.update(_config_payload(f"tsp_{tier}", root))
            return payload
        return {
            "problem": problem,
            "size": size,
            "tier": tier,
            "priority": priority,
            "solver_family": "baseline",
            "solver_name": "nearest_neighbor_2opt",
            "config_name": None,
            "config_path": None,
            "config_abspath": None,
            "resource_uri": None,
            "portable_command_hint": None,
            "validated_sizes": validated_sizes,
            "note": (
                "Nearest-neighbor plus 2-opt is the practical TSP default "
                "on validated sizes except for the promoted medium hybrid "
                "quality path."
            ),
        }

    if problem == "zdt1":
        if priority not in {"default", "quality", "ga", "hv", "coverage"}:
            raise ValueError(
                "Unsupported priority for zdt1. Allowed: coverage, default, ga, hv, quality"
            )
        payload = {
            "problem": problem,
            "size": size,
            "tier": tier,
            "priority": priority,
            "solver_family": "pure-ga",
            "solver_name": "recommended_preset",
            "validated_sizes": validated_sizes,
            "note": (
                "Use the NSGA-II preset as the default validated solver. At size "
                "50, mutation_archive is a cheaper hypervolume-only baseline, "
                "but the preset remains the recommended multi-metric path."
                if size == 50
                else "Use the NSGA-II preset as the default validated solver."
            ),
        }
        payload.update(_config_payload(f"zdt1_{tier}", root))
        return payload

    raise ValueError(f"Unsupported problem: {problem}")
