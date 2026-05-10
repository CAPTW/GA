# ruff: noqa: E501

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ga_lab.config import GAConfig
from ga_lab.experiment import budget_baseline_comparison as basecmp
from ga_lab.factory import build_runtime_context
from ga_lab.governance.run_metadata import build_run_metadata, write_run_metadata
from ga_lab.utils.seed import make_rng

PROJECT_ROOT = basecmp.PROJECT_ROOT


RUN_COLUMNS = (
    "suite_name",
    "suite_kind",
    "problem",
    "tier",
    "size",
    "size_key",
    "label",
    "family",
    "seed",
    "preset_path",
    "configured_evaluation_budget",
    "actual_evaluations",
    "hybrid_extra_evaluations",
    "hybrid_local_search_applications",
    "hybrid_local_search_improvements",
    "runtime_seconds",
    "success_to_target",
    "evaluations_to_target",
    "generations_to_target",
    "final_best_fitness",
    "final_best_distance",
    "best_feasible_fitness",
    "feasible_rate",
    "mean_violation",
    "hypervolume",
    "pareto_ratio",
    "spread",
    "pareto_front_size",
)


@dataclass(slots=True)
class HybridCandidate:
    label: str
    overrides: dict[str, Any]
    preset_path: Path | None = None


@dataclass(slots=True)
class ComparisonEntry:
    suite_name: str
    suite_kind: str
    problem: str
    size: int
    preset_path: Path
    baselines: tuple[str, ...]
    hybrids: tuple[HybridCandidate, ...]
    seeds: int
    seed_start: int


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = json.loads(json.dumps(value))
    return merged


def _resolve_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    candidates = [
        (PROJECT_ROOT / path).resolve(),
        (manifest_path.parent / path).resolve(),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _load_hybrids(manifest_path: Path, raw_entry: dict[str, Any]) -> tuple[HybridCandidate, ...]:
    hybrids: list[HybridCandidate] = []
    for raw_hybrid in raw_entry.get("hybrids", []):
        if not isinstance(raw_hybrid, dict):
            raise ValueError("Hybrid entries must be JSON objects")
        label = str(raw_hybrid["label"])
        preset_path = None
        if "preset" in raw_hybrid:
            preset_path = _resolve_path(manifest_path, str(raw_hybrid["preset"]))
        overrides = dict(raw_hybrid.get("overrides", {}))
        hybrids.append(HybridCandidate(label=label, overrides=overrides, preset_path=preset_path))
    return tuple(hybrids)


def load_comparison_manifest(path: str | Path) -> tuple[dict[str, Any], list[ComparisonEntry]]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Hybrid comparison manifest must be a JSON object")
    suite_name = str(manifest.get("suite_name", manifest_path.stem))
    suite_kind = str(manifest.get("suite_kind", "comparison"))
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Hybrid comparison manifest requires a non-empty entries list")

    entries: list[ComparisonEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Each hybrid comparison manifest entry must be an object")
        problem = str(raw_entry["problem"])
        size = int(raw_entry["size"])
        baselines = tuple(str(value) for value in raw_entry.get("baselines", []))
        basecmp._validated_entry(problem, size, baselines)
        preset_path = _resolve_path(manifest_path, str(raw_entry["preset"]))
        entries.append(
            ComparisonEntry(
                suite_name=suite_name,
                suite_kind=suite_kind,
                problem=problem,
                size=size,
                preset_path=preset_path,
                baselines=baselines,
                hybrids=_load_hybrids(manifest_path, raw_entry),
                seeds=int(raw_entry.get("seeds", manifest.get("default_seeds", 10))),
                seed_start=int(raw_entry.get("seed_start", 0)),
            )
        )
    return manifest, entries


def _config_reference(base_preset_path: Path, candidate: HybridCandidate | None) -> str:
    if candidate is None:
        return str(base_preset_path.as_posix())
    if candidate.preset_path is not None:
        return str(candidate.preset_path.as_posix())
    return f"{base_preset_path.as_posix()}#hybrid:{candidate.label}"


def _run_optimizer_trial(
    entry: ComparisonEntry,
    config: GAConfig,
    *,
    label: str,
    family: str,
    preset_reference: str,
    seed: int,
) -> dict[str, Any]:
    runtime = build_runtime_context(config)
    tracked_problem = basecmp.TrackedProblem(runtime.problem, config)
    rng = make_rng(seed)
    started = time.perf_counter()
    algorithm_summary, _history = runtime.algorithm_fn(
        config=config,
        problem=tracked_problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=rng,
    )
    elapsed = time.perf_counter() - started
    tracked_metrics = tracked_problem.tracked_metrics()
    return {
        "suite_name": entry.suite_name,
        "suite_kind": entry.suite_kind,
        "problem": entry.problem,
        "tier": basecmp._tier_name(entry.problem, entry.size),
        "size": entry.size,
        "size_key": basecmp.VALIDATED_SCOPE[entry.problem]["size_key"],
        "label": label,
        "family": family,
        "seed": seed,
        "preset_path": preset_reference,
        "configured_evaluation_budget": basecmp.configured_evaluation_budget(config),
        "actual_evaluations": tracked_metrics["actual_evaluations"],
        "hybrid_extra_evaluations": algorithm_summary.get("hybrid_extra_evaluations"),
        "hybrid_local_search_applications": algorithm_summary.get(
            "hybrid_local_search_applications"
        ),
        "hybrid_local_search_improvements": algorithm_summary.get(
            "hybrid_local_search_improvements"
        ),
        "runtime_seconds": elapsed,
        "success_to_target": (
            algorithm_summary["stop_reason"] == "target_fitness_reached"
            if config.target_fitness is not None
            else None
        ),
        "evaluations_to_target": tracked_metrics.get("evaluations_to_target"),
        "generations_to_target": algorithm_summary.get("convergence_generation"),
        "final_best_fitness": algorithm_summary.get("best_fitness"),
        "final_best_distance": algorithm_summary.get("best_route_distance"),
        "best_feasible_fitness": tracked_metrics.get(
            "best_feasible_fitness",
            algorithm_summary.get("best_total_value")
            if algorithm_summary.get("best_is_feasible")
            else None,
        ),
        "feasible_rate": tracked_metrics.get("feasible_rate"),
        "mean_violation": tracked_metrics.get("mean_violation"),
        "hypervolume": algorithm_summary.get("hypervolume"),
        "pareto_ratio": algorithm_summary.get("pareto_ratio"),
        "spread": algorithm_summary.get("spread"),
        "pareto_front_size": algorithm_summary.get("pareto_front_size"),
    }


def _run_hybrid_trial(
    entry: ComparisonEntry,
    base_config: GAConfig,
    candidate: HybridCandidate,
    seed: int,
) -> dict[str, Any]:
    if candidate.preset_path is not None:
        hybrid_data = json.loads(candidate.preset_path.read_text(encoding="utf-8"))
    else:
        hybrid_data = _deep_merge(base_config.to_dict(), candidate.overrides)
    hybrid_data["seed"] = seed
    config = GAConfig.from_dict(hybrid_data)
    return _run_optimizer_trial(
        entry,
        config,
        label=candidate.label,
        family="hybrid_ga",
        preset_reference=_config_reference(entry.preset_path, candidate),
        seed=seed,
    )


def _group_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["suite_name"],
            row["suite_kind"],
            row["problem"],
            row["tier"],
            row["size"],
            row["label"],
            row["family"],
            row["preset_path"],
            row["configured_evaluation_budget"],
        )
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        sample = bucket[0]
        aggregate_rows.append(
            {
                "suite_name": sample["suite_name"],
                "suite_kind": sample["suite_kind"],
                "problem": sample["problem"],
                "tier": sample["tier"],
                "size": sample["size"],
                "size_key": sample["size_key"],
                "label": sample["label"],
                "family": sample["family"],
                "preset_path": sample["preset_path"],
                "configured_evaluation_budget": sample["configured_evaluation_budget"],
                "run_count": len(bucket),
                "mean_actual_evaluations": basecmp._mean_or_none(bucket, "actual_evaluations"),
                "mean_hybrid_extra_evaluations": basecmp._mean_or_none(
                    bucket, "hybrid_extra_evaluations"
                ),
                "mean_hybrid_local_search_applications": basecmp._mean_or_none(
                    bucket, "hybrid_local_search_applications"
                ),
                "mean_hybrid_local_search_improvements": basecmp._mean_or_none(
                    bucket, "hybrid_local_search_improvements"
                ),
                "mean_runtime_seconds": basecmp._mean_or_none(bucket, "runtime_seconds"),
                "stdev_runtime_seconds": basecmp._stdev_or_none(bucket, "runtime_seconds"),
                "success_rate": basecmp._rate(bucket, "success_to_target"),
                "mean_evaluations_to_target": basecmp._mean_or_none(
                    bucket, "evaluations_to_target"
                ),
                "mean_generations_to_target": basecmp._mean_or_none(
                    bucket, "generations_to_target"
                ),
                "mean_final_best_fitness": basecmp._mean_or_none(bucket, "final_best_fitness"),
                "stdev_final_best_fitness": basecmp._stdev_or_none(
                    bucket, "final_best_fitness"
                ),
                "mean_final_best_distance": basecmp._mean_or_none(bucket, "final_best_distance"),
                "stdev_final_best_distance": basecmp._stdev_or_none(
                    bucket, "final_best_distance"
                ),
                "mean_best_feasible_fitness": basecmp._mean_or_none(
                    bucket, "best_feasible_fitness"
                ),
                "stdev_best_feasible_fitness": basecmp._stdev_or_none(
                    bucket, "best_feasible_fitness"
                ),
                "mean_feasible_rate": basecmp._mean_or_none(bucket, "feasible_rate"),
                "mean_violation": basecmp._mean_or_none(bucket, "mean_violation"),
                "mean_hypervolume": basecmp._mean_or_none(bucket, "hypervolume"),
                "stdev_hypervolume": basecmp._stdev_or_none(bucket, "hypervolume"),
                "mean_pareto_ratio": basecmp._mean_or_none(bucket, "pareto_ratio"),
                "mean_spread": basecmp._mean_or_none(bucket, "spread"),
                "mean_pareto_front_size": basecmp._mean_or_none(bucket, "pareto_front_size"),
            }
        )
    return aggregate_rows


def _lookup_row(
    aggregates: list[dict[str, Any]],
    *,
    suite_kind: str,
    problem: str,
    size: int,
    label: str,
) -> dict[str, Any] | None:
    for row in aggregates:
        if (
            row["suite_kind"] == suite_kind
            and row["problem"] == problem
            and row["size"] == size
            and row["label"] == label
        ):
            return row
    return None


def _gap_recap() -> list[dict[str, str]]:
    return [
        {
            "problem": "onemax",
            "current_recommended_pure_ga_preset": "configs/presets/onemax_{small,medium,large}.json",
            "strongest_cheap_baseline": "hill_climb",
            "current_gap": "hill climb reaches the same targets with far fewer evaluations on validated sizes",
            "hybridization_makes_sense": "low",
            "solver_choice_only_may_be_right": "yes",
        },
        {
            "problem": "knapsack",
            "current_recommended_pure_ga_preset": "configs/presets/knapsack_{small,medium,large}.json",
            "strongest_cheap_baseline": "greedy_local_search",
            "current_gap": "pure GA beats random, but greedy local search is equal or slightly better on feasible value",
            "hybridization_makes_sense": "high",
            "solver_choice_only_may_be_right": "possibly",
        },
        {
            "problem": "tsp",
            "current_recommended_pure_ga_preset": "configs/presets/tsp_{small,medium,large}.json",
            "strongest_cheap_baseline": "nearest_neighbor_2opt",
            "current_gap": "pure GA improves over random tours, but nearest-neighbor plus 2-opt is much stronger",
            "hybridization_makes_sense": "very_high",
            "solver_choice_only_may_be_right": "possibly",
        },
        {
            "problem": "zdt1",
            "current_recommended_pure_ga_preset": "configs/presets/zdt1_{small,medium,large}.json",
            "strongest_cheap_baseline": "mutation_archive (large), random_archive (overall weaker)",
            "current_gap": "pure NSGA-II is already the strongest GA story; large remains metric-tradeoff rather than scalar dominance",
            "hybridization_makes_sense": "medium",
            "solver_choice_only_may_be_right": "yes",
        },
    ]


def _problem_notes(aggregates: list[dict[str, Any]]) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []

    for size in basecmp.VALIDATED_SCOPE["knapsack"]["sizes"]:
        pure = _lookup_row(
            aggregates,
            suite_kind="comparison",
            problem="knapsack",
            size=size,
            label="recommended_preset",
        )
        greedy = _lookup_row(
            aggregates,
            suite_kind="comparison",
            problem="knapsack",
            size=size,
            label="greedy_local_search",
        )
        hybrid = _lookup_row(
            aggregates,
            suite_kind="comparison",
            problem="knapsack",
            size=size,
            label="hybrid_seeded_repair",
        )
        if pure and greedy and hybrid:
            notes.append(
                {
                    "problem": "knapsack",
                    "size": str(size),
                    "note": (
                        "seeded repair closes the pure-GA feasibility/value gap, but it still does not"
                        " create a clear edge over the cheap greedy baseline."
                    ),
                }
            )

    for size in basecmp.VALIDATED_SCOPE["tsp"]["sizes"]:
        pure = _lookup_row(
            aggregates,
            suite_kind="comparison",
            problem="tsp",
            size=size,
            label="recommended_preset",
        )
        baseline = _lookup_row(
            aggregates,
            suite_kind="comparison",
            problem="tsp",
            size=size,
            label="nearest_neighbor_2opt",
        )
        hybrid = _lookup_row(
            aggregates,
            suite_kind="comparison",
            problem="tsp",
            size=size,
            label="hybrid_memetic_tsp",
        )
        if pure and baseline and hybrid:
            notes.append(
                {
                    "problem": "tsp",
                    "size": str(size),
                    "note": (
                        "nearest-neighbor seeding plus bounded 2-opt dramatically reduces the pure-GA gap;"
                        " promotion depends on whether the confirm run can beat or at least robustly match"
                        " the strongest cheap baseline."
                    ),
                }
            )

    notes.extend(
        [
            {
                "problem": "onemax",
                "size": "32/64/128",
                "note": "No hybrid candidate was promoted because hill climbing remains the practical default.",
            },
            {
                "problem": "zdt1",
                "size": "10/20/50",
                "note": "No hybrid candidate was added because preserving NSGA-II quality and metric tradeoffs mattered more than forcing local search into a multi-objective workflow.",
            },
        ]
    )
    return notes


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Hybrid comparison summary",
        "",
        "## Scope",
        "",
        "Validated comparison scope only:",
        "",
        "| Problem | Size key | Validated sizes |",
        "| --- | --- | --- |",
    ]
    for problem, scope in basecmp.VALIDATED_SCOPE.items():
        sizes = " / ".join(str(size) for size in scope["sizes"])
        lines.append(f"| {problem} | `{scope['size_key']}` | `{sizes}` |")

    lines.extend(
        [
            "",
            "## Gap recap",
            "",
            "| Problem | Pure GA preset | Strongest cheap baseline | Current gap | Hybridization makes sense | Solver-choice only may be enough |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["gap_recap"]:
        lines.append(
            f"| {row['problem']} | `{row['current_recommended_pure_ga_preset']}` | "
            f"`{row['strongest_cheap_baseline']}` | {row['current_gap']} | "
            f"`{row['hybridization_makes_sense']}` | `{row['solver_choice_only_may_be_right']}` |"
        )

    lines.extend(
        [
            "",
            "## Fairness policy",
            "",
            "- Primary comparison basis: matched function evaluation budget.",
            "- Initial population evaluation is included in the GA or hybrid GA budget.",
            "- The final post-loop population re-evaluation is included because it is part of the current runner path.",
            "- Hybrid local-search evaluations are counted inside the same configured budget ceiling.",
            "- Heuristic repair or seeding that does not call the objective is documented as a note, not as extra evaluations.",
            "- Wall-clock runtime is reported, but it is secondary to matched-budget quality.",
            "",
            "## Budget definition",
            "",
            "| Algorithm | Configured budget formula |",
            "| --- | --- |",
            f"| `ga` | `{summary['budget_definition']['ga']}` |",
            f"| `hybrid_ga` | `{summary['budget_definition']['hybrid_ga']}` |",
            f"| `nsga2` | `{summary['budget_definition']['nsga2']}` |",
            "",
            "## Aggregate comparison snapshot",
            "",
            "| Problem | Size | Label | Budget | Actual evals | Hybrid extra evals | Key metric | Runtime |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in summary["aggregate_rows"]:
        key_metric = "-"
        if row.get("success_rate") is not None:
            key_metric = f"success `{row['success_rate']:.2f}`"
        elif row.get("mean_best_feasible_fitness") is not None:
            key_metric = f"feasible fitness `{row['mean_best_feasible_fitness']:.2f}`"
        elif row.get("mean_final_best_distance") is not None:
            key_metric = f"distance `{row['mean_final_best_distance']:.2f}`"
        elif row.get("mean_hypervolume") is not None:
            key_metric = f"HV `{row['mean_hypervolume']:.4f}`"
        lines.append(
            f"| {row['problem']} | `{row['size']}` | `{row['label']}` | "
            f"`{row['configured_evaluation_budget']}` | "
            f"`{row['mean_actual_evaluations']:.1f}` | "
            f"`{(row['mean_hybrid_extra_evaluations'] or 0.0):.1f}` | "
            f"{key_metric} | `{row['mean_runtime_seconds']:.3f}s` |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "| Problem | Size | Note |",
            "| --- | --- | --- |",
        ]
    )
    for row in summary["problem_notes"]:
        lines.append(f"| {row['problem']} | `{row['size']}` | {row['note']} |")
    return "\n".join(lines) + "\n"


def _run_manifest(
    manifest_path: str | Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    manifest, entries = load_comparison_manifest(manifest_path)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suite_dir = output_root / f"{timestamp}_{manifest['suite_name']}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []

    for entry in entries:
        base_config = GAConfig.from_dict(json.loads(entry.preset_path.read_text(encoding="utf-8")))
        if base_config.problem != entry.problem:
            raise ValueError(
                f"Preset problem mismatch for {entry.preset_path}: {base_config.problem} != {entry.problem}"
            )
        if basecmp._size_value(base_config) != entry.size:
            raise ValueError(
                f"Preset size mismatch for {entry.preset_path}: {basecmp._size_value(base_config)} != {entry.size}"
            )

        for offset in range(entry.seeds):
            seed = entry.seed_start + offset

            pure_data = base_config.to_dict()
            pure_data["seed"] = seed
            pure_config = GAConfig.from_dict(pure_data)
            run_rows.append(
                _run_optimizer_trial(
                    entry,
                    pure_config,
                    label="recommended_preset",
                    family="ga_preset",
                    preset_reference=_config_reference(entry.preset_path, None),
                    seed=seed,
                )
            )

            for candidate in entry.hybrids:
                run_rows.append(_run_hybrid_trial(entry, base_config, candidate, seed))

            for family in entry.baselines:
                baseline_data = base_config.to_dict()
                baseline_data["seed"] = seed
                baseline_config = GAConfig.from_dict(baseline_data)
                baseline_row = basecmp._run_baseline_trial(entry, baseline_config, family, seed)
                baseline_row.update(
                    {
                        "hybrid_extra_evaluations": None,
                        "hybrid_local_search_applications": None,
                        "hybrid_local_search_improvements": None,
                    }
                )
                run_rows.append(baseline_row)

    aggregate_rows = _group_run_rows(run_rows)
    (suite_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (suite_dir / "run_rows.json").write_text(
        json.dumps(run_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (suite_dir / "aggregate_rows.json").write_text(
        json.dumps(aggregate_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if aggregate_rows:
        basecmp._write_csv(suite_dir / "aggregate_rows.csv", aggregate_rows, tuple(aggregate_rows[0].keys()))
    return manifest, run_rows, suite_dir


def run_manifests(
    manifest_paths: list[str | Path],
    *,
    output_root: str | Path,
    summary_stem: str,
) -> dict[str, Any]:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    all_run_rows: list[dict[str, Any]] = []
    suite_inputs: dict[str, str] = {}
    manifests_payload: dict[str, Any] = {}

    for manifest_path in manifest_paths:
        manifest, run_rows, suite_dir = _run_manifest(manifest_path, output_root_path)
        suite_name = str(manifest["suite_name"])
        suite_inputs[suite_name] = str(suite_dir.as_posix())
        manifests_payload[suite_name] = manifest
        all_run_rows.extend(run_rows)

    aggregate_rows = _group_run_rows(all_run_rows)
    aggregate_columns = tuple(aggregate_rows[0].keys()) if aggregate_rows else ()
    run_metadata = build_run_metadata(
        project_root=PROJECT_ROOT,
        summary_stem=summary_stem,
        output_root=output_root_path,
        manifest_paths=manifest_paths,
        extra={
            "suite_kind": "hybrid_comparison",
            "suite_names": sorted(suite_inputs),
            "run_row_count": len(all_run_rows),
        },
    )
    summary = {
        "summary_version": 1,
        "summary_schema_version": 2,
        "validated_scope": {
            problem: {
                "size_key": scope["size_key"],
                "sizes": list(scope["sizes"]),
            }
            for problem, scope in basecmp.VALIDATED_SCOPE.items()
        },
        "suite_inputs": suite_inputs,
        "manifests": manifests_payload,
        "fairness_policy": {
            "primary_basis": "matched_function_evaluation_budget",
            "initial_population_evaluations_included": True,
            "final_post_loop_population_re_evaluation_included": True,
            "hybrid_extra_evaluations_counted_in_budget": True,
            "repair_without_objective_calls_counted_only_as_note": True,
            "runtime_is_secondary": True,
        },
        "budget_definition": {
            "ga": basecmp.budget_formula_text("ga"),
            "hybrid_ga": basecmp.budget_formula_text("hybrid_ga"),
            "nsga2": basecmp.budget_formula_text("nsga2"),
        },
        "gap_recap": _gap_recap(),
        "run_rows": all_run_rows,
        "aggregate_rows": aggregate_rows,
        "problem_notes": _problem_notes(aggregate_rows),
        "run_metadata": run_metadata,
    }

    (output_root_path / f"{summary_stem}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_run_metadata(output_root_path / f"{summary_stem}_run_metadata.json", run_metadata)
    if aggregate_rows:
        basecmp._write_csv(output_root_path / f"{summary_stem}.csv", aggregate_rows, aggregate_columns)
    (output_root_path / f"{summary_stem}.md").write_text(
        _markdown_summary(summary),
        encoding="utf-8",
    )
    return summary
