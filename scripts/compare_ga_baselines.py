# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from ga_lab.config import GAConfig
from ga_lab.experiment.baseline_protocol import (
    build_results_markdown,
    normalize_run_row,
    run_sphere_ga,
    run_sphere_hill_climb,
    run_sphere_random_search,
    serialize_baseline_result,
    sphere_result_to_row,
    summarize_paired_comparison,
    summarize_problem_algorithm,
    write_csv,
    write_json,
)
from ga_lab.experiment.budget_baseline_comparison import (
    configured_evaluation_budget,
    run_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeatable GA vs simple-baseline comparisons on small local benchmarks."
    )
    parser.add_argument(
        "--manifest",
        default="configs/baselines/local_ga_baseline_comparison.json",
        help="Built-in comparison manifest for registered problems.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/baseline_comparison",
        help="Directory where timestamped comparison outputs will be written.",
    )
    parser.add_argument(
        "--summary-stem",
        default="baseline_comparison_results",
        help="Stem for merged built-in comparison summaries.",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Directory where stable comparison artifacts will be written.",
    )
    parser.add_argument(
        "--skip-sphere",
        action="store_true",
        help="Skip the custom Sphere comparison path.",
    )
    parser.add_argument(
        "--sphere-seeds",
        type=int,
        default=10,
        help="Number of seeds for the custom Sphere comparison.",
    )
    parser.add_argument(
        "--sphere-seed-start",
        type=int,
        default=4101,
        help="Start seed for the custom Sphere comparison.",
    )
    return parser.parse_args()


def _sphere_config() -> GAConfig:
    return GAConfig(
        run_name="baseline_compare_sphere",
        problem="sphere",
        algorithm="ga",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=32,
        genome_length=8,
        generations=40,
        crossover_rate=0.9,
        mutation_rate=0.12,
        elitism=1,
        tournament_size=3,
        maximize=False,
        target_fitness=1e-2,
        log_every=5,
        representation_options={"low": -5.0, "high": 5.0},
        mutation_options={"sigma": 0.2},
    )


def _run_sphere_suite(*, seeds: list[int], output_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    config = _sphere_config()
    budget = configured_evaluation_budget(config)
    raw_results: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for seed in seeds:
        ga_result = run_sphere_ga(config, seed=seed)
        random_result = run_sphere_random_search(config, seed=seed, budget=budget)
        hill_result = run_sphere_hill_climb(config, seed=seed, budget=budget)
        raw_results.extend(
            [
                serialize_baseline_result(ga_result),
                serialize_baseline_result(random_result),
                serialize_baseline_result(hill_result),
            ]
        )
        run_rows.extend(
            [
                sphere_result_to_row(ga_result),
                sphere_result_to_row(random_result),
                sphere_result_to_row(hill_result),
            ]
        )

    sphere_dir = output_root / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_sphere_custom"
    sphere_dir.mkdir(parents=True, exist_ok=True)
    write_json(sphere_dir / "sphere_suite_results.json", {"runs": raw_results, "run_rows": run_rows})
    return raw_results, run_rows


def _collect_failures(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    failures = []
    for row in rows:
        error_message = row.get("error_message")
        if error_message:
            failures.append(
                {
                    "problem": row.get("problem"),
                    "algorithm": row.get("algorithm"),
                    "seed": row.get("seed"),
                    "error_message": error_message,
                }
            )
    return failures


def main() -> None:
    args = parse_args()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.argv)
    generated_at = datetime.now(UTC).isoformat()

    built_in_summary = run_manifests(
        [args.manifest],
        output_root=output_root,
        summary_stem=args.summary_stem,
    )
    built_in_rows = [
        row for row in built_in_summary["run_rows"] if row["problem"] in {"onemax", "knapsack", "tsp"}
    ]

    sphere_raw_results: list[dict[str, object]] = []
    sphere_rows: list[dict[str, object]] = []
    if not args.skip_sphere:
        sphere_seeds = list(range(args.sphere_seed_start, args.sphere_seed_start + args.sphere_seeds))
        sphere_raw_results, sphere_rows = _run_sphere_suite(seeds=sphere_seeds, output_root=output_root)

    normalized_rows = [normalize_run_row(row) for row in built_in_rows]
    normalized_rows.extend(normalize_run_row(row) for row in sphere_rows)

    grouped_rows: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in normalized_rows:
        grouped_rows[(str(row["problem"]), str(row["algorithm"]))].append(row)

    scope_rows = [
        summarize_problem_algorithm(problem=problem, algorithm=algorithm, rows=rows)
        for (problem, algorithm), rows in sorted(grouped_rows.items())
    ]

    paired_rows = []
    by_problem: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(dict)
    for row in normalized_rows:
        by_problem[str(row["problem"])].setdefault(str(row["algorithm"]), []).append(row)
    for problem, algorithm_rows in sorted(by_problem.items()):
        ga_rows = algorithm_rows.get("GA")
        if not ga_rows:
            continue
        for algorithm, rows in sorted(algorithm_rows.items()):
            if algorithm == "GA":
                continue
            paired_rows.append(
                summarize_paired_comparison(problem=problem, ga_rows=ga_rows, baseline_rows=rows)
            )

    failures = _collect_failures(sphere_raw_results)
    results_payload = {
        "generated_at": generated_at,
        "command": command,
        "scope": {
            "included_problems": ["onemax", "sphere", "knapsack", "tsp"]
            if not args.skip_sphere
            else ["onemax", "knapsack", "tsp"],
            "excluded_problems": [
                "vrp",
                "job-shop scheduling",
                "timetabling",
                "neural architecture search",
                "real-time optimization",
                "large-scale industrial optimization",
                "zdt1 full baseline comparison",
            ],
        },
        "fairness_contract": {
            "same_evaluation_budget": True,
            "explicit_seed_control": True,
            "direction_consistent": True,
            "runtime_recorded": True,
            "failures_recorded": True,
            "paired_seed_comparison": True,
            "single_run_not_used_for_claims": True,
        },
        "built_in_summary_path": str(
            (output_root / f"{args.summary_stem}.json").resolve().as_posix()
        ),
        "sphere_config": _sphere_config().to_dict() if not args.skip_sphere else None,
        "sphere_raw_results": sphere_raw_results,
        "normalized_rows": normalized_rows,
        "problem_summaries": scope_rows,
        "paired_summaries": paired_rows,
        "failures": failures,
    }

    json_path = artifact_root / "baseline_comparison_results.json"
    csv_path = artifact_root / "baseline_comparison_results.csv"
    md_path = artifact_root / "baseline_comparison_results.md"
    write_json(json_path, results_payload)
    write_csv(
        csv_path,
        scope_rows,
        [
            "problem",
            "algorithm",
            "seeds",
            "metric",
            "mean_best",
            "std",
            "median",
            "best",
            "worst",
            "success_rate",
            "mean_runtime_seconds",
            "mean_actual_evaluations",
        ],
    )
    md_path.write_text(
        build_results_markdown(
            generated_at=generated_at,
            command=command,
            scope_rows=scope_rows,
            paired_rows=paired_rows,
            failures=failures,
        ),
        encoding="utf-8",
    )
    print(json.dumps(results_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
