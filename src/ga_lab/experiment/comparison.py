from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterable, Mapping
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

COMPARISON_SCHEMA_VERSION = 2
COMPARISON_COLUMNS = (
    "suite_name",
    "baseline_label",
    "run_name",
    "problem",
    "seed",
    "algorithm",
    "representation",
    "selection",
    "crossover",
    "mutation",
    "population_size",
    "genome_length",
    "generations",
    "crossover_rate",
    "mutation_rate",
    "elitism",
    "tournament_size",
    "maximize",
    "target_fitness",
    "best_fitness",
    "mean_fitness",
    "worst_fitness",
    "stop_reason",
    "convergence_generation",
    "final_generation",
    "runtime_seconds",
    "objective_count",
    "pareto_front_size",
    "pareto_ratio",
    "hypervolume",
    "spread",
    "config_path",
    "output_dir",
)

_COMPARABLE_GROUP_FIELDS = (
    "problem",
    "algorithm",
    "representation",
    "population_size",
    "genome_length",
    "generations",
    "crossover_rate",
    "mutation_rate",
    "elitism",
    "objective_count",
    "maximize",
)


def _pick(
    summary: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    key: str,
) -> Any:
    if key in summary:
        return summary[key]
    if config is not None and key in config:
        return config[key]
    return None


def flatten_summary(
    summary: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    suite_name: str | None = None,
    baseline_label: str | None = None,
    config_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    row = {
        "suite_name": suite_name,
        "baseline_label": baseline_label,
        "run_name": _pick(summary, config, "run_name"),
        "problem": _pick(summary, config, "problem"),
        "seed": _pick(summary, config, "seed"),
        "algorithm": _pick(summary, config, "algorithm"),
        "representation": _pick(summary, config, "representation"),
        "selection": _pick(summary, config, "selection"),
        "crossover": _pick(summary, config, "crossover"),
        "mutation": _pick(summary, config, "mutation"),
        "population_size": _pick(summary, config, "population_size"),
        "genome_length": _pick(summary, config, "genome_length"),
        "generations": _pick(summary, config, "generations"),
        "crossover_rate": _pick(summary, config, "crossover_rate"),
        "mutation_rate": _pick(summary, config, "mutation_rate"),
        "elitism": _pick(summary, config, "elitism"),
        "tournament_size": _pick(summary, config, "tournament_size"),
        "maximize": _pick(summary, config, "maximize"),
        "target_fitness": _pick(summary, config, "target_fitness"),
        "best_fitness": _pick(summary, config, "best_fitness"),
        "mean_fitness": _pick(summary, config, "mean_fitness"),
        "worst_fitness": _pick(summary, config, "worst_fitness"),
        "stop_reason": _pick(summary, config, "stop_reason"),
        "convergence_generation": _pick(summary, config, "convergence_generation"),
        "final_generation": _pick(summary, config, "final_generation"),
        "runtime_seconds": _pick(summary, config, "runtime_seconds"),
        "objective_count": _pick(summary, config, "objective_count"),
        "pareto_front_size": _pick(summary, config, "pareto_front_size"),
        "pareto_ratio": _pick(summary, config, "pareto_ratio"),
        "hypervolume": _pick(summary, config, "hypervolume"),
        "spread": _pick(summary, config, "spread"),
        "config_path": config_path,
        "output_dir": output_dir,
    }
    return row


def _csv_value(value: Any) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_comparison_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    serialized_rows = []
    for row in rows:
        serialized_rows.append(
            {column: _csv_value(row.get(column)) for column in COMPARISON_COLUMNS}
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPARISON_COLUMNS))
        writer.writeheader()
        writer.writerows(serialized_rows)


def write_comparison_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(row), ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_run_artifacts(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path = summary_path.with_name("config.json")
    if not config_path.exists():
        return summary, None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return summary, config


def _comparison_key(run_name: Any, seed: Any, problem: Any) -> tuple[str, str, str]:
    return (str(run_name), str(seed), str(problem))


def _load_existing_rows(results_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    table_paths: dict[Path, Path] = {}
    for jsonl_path in results_dir.glob("**/RUNS.jsonl"):
        table_paths[jsonl_path.parent] = jsonl_path
    for csv_path in results_dir.glob("**/RUNS.csv"):
        table_paths.setdefault(csv_path.parent, csv_path)

    for table_path in sorted(
        table_paths.values(),
        key=lambda path: (len(path.parts), str(path)),
        reverse=True,
    ):
        rows: list[dict[str, Any]] = []
        if table_path.suffix == ".jsonl":
            for line in table_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        else:
            with table_path.open(encoding="utf-8", newline="") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))

        for row in rows:
            run_name = row.get("run_name")
            seed = row.get("seed")
            problem = row.get("problem")
            if run_name is None or seed is None or problem is None:
                continue
            key = _comparison_key(run_name, seed, problem)
            index.setdefault(key, row)
    return index


def collect_comparison_rows(
    results_dir: str | Path,
    *,
    suite_name: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(results_dir)
    existing_rows = _load_existing_rows(root)
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("**/summary.json")):
        summary, config = load_run_artifacts(summary_path)
        row = flatten_summary(
            summary,
            config=config,
            suite_name=suite_name,
            output_dir=str(summary_path.parent),
            config_path=str(summary_path.with_name("config.json")),
        )
        existing = existing_rows.get(
            _comparison_key(row.get("run_name"), row.get("seed"), row.get("problem"))
        )
        if existing is not None:
            for field_name in ("suite_name", "baseline_label"):
                if row.get(field_name) is None and existing.get(field_name):
                    row[field_name] = existing[field_name]
        rows.append(row)
    return rows


def _finite_values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float) and math.isfinite(value):
            values.append(float(value))
    return values


def _finite_mean(rows: list[Mapping[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    if not values:
        return None
    return mean(values)


def _finite_median(rows: list[Mapping[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    if not values:
        return None
    return median(values)


def _finite_stdev(rows: list[Mapping[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    if len(values) < 2:
        return 0.0 if values else None
    return stdev(values)


def _finite_min(rows: list[Mapping[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    if not values:
        return None
    return min(values)


def _finite_max(rows: list[Mapping[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    if not values:
        return None
    return max(values)


def _sorted_distinct(rows: list[Mapping[str, Any]], key: str) -> list[Any]:
    values = {row[key] for row in rows if row.get(key) is not None}
    return sorted(values)


def _rate(
    rows: list[Mapping[str, Any]],
    *,
    predicate: Callable[[Mapping[str, Any]], bool],
) -> float:
    if not rows:
        return 0.0
    matched = sum(1 for row in rows if predicate(row))
    return matched / len(rows)


def _baseline_record(label: str, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "baseline_label": label,
        "run_count": len(rows),
        "config_paths": _sorted_distinct(rows, "config_path"),
        "problems": _sorted_distinct(rows, "problem"),
        "algorithms": _sorted_distinct(rows, "algorithm"),
        "representations": _sorted_distinct(rows, "representation"),
        "selections": _sorted_distinct(rows, "selection"),
        "crossovers": _sorted_distinct(rows, "crossover"),
        "mutations": _sorted_distinct(rows, "mutation"),
        "seed_values": _sorted_distinct(rows, "seed"),
        "stop_reasons": _sorted_distinct(rows, "stop_reason"),
        "mean_best_fitness": _finite_mean(rows, "best_fitness"),
        "median_best_fitness": _finite_median(rows, "best_fitness"),
        "stdev_best_fitness": _finite_stdev(rows, "best_fitness"),
        "min_best_fitness": _finite_min(rows, "best_fitness"),
        "max_best_fitness": _finite_max(rows, "best_fitness"),
        "mean_mean_fitness": _finite_mean(rows, "mean_fitness"),
        "median_mean_fitness": _finite_median(rows, "mean_fitness"),
        "stdev_mean_fitness": _finite_stdev(rows, "mean_fitness"),
        "mean_final_generation": _finite_mean(rows, "final_generation"),
        "median_final_generation": _finite_median(rows, "final_generation"),
        "stdev_final_generation": _finite_stdev(rows, "final_generation"),
        "mean_runtime_seconds": _finite_mean(rows, "runtime_seconds"),
        "median_runtime_seconds": _finite_median(rows, "runtime_seconds"),
        "target_hit_rate": _rate(
            rows,
            predicate=lambda row: row.get("stop_reason") == "target_fitness_reached",
        ),
    }


def _group_signature(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    sample = rows[0]
    return {field: sample.get(field) for field in _COMPARABLE_GROUP_FIELDS}


def _ranking_value(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def _winner(records: list[dict[str, Any]], key: str, *, higher_is_better: bool) -> str | None:
    filtered = [record for record in records if _ranking_value(record, key) is not None]
    if not filtered:
        return None

    def sort_key(record: Mapping[str, Any]) -> tuple[float, str]:
        value = float(record[key])
        return ((-value) if higher_is_better else value, str(record["baseline_label"]))

    return sorted(filtered, key=sort_key)[0]["baseline_label"]


def _seed_index(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        seed = row.get("seed")
        if seed is None:
            continue
        index[str(seed)] = row
    return index


def _paired_metric_summary(
    left_rows: list[Mapping[str, Any]],
    right_rows: list[Mapping[str, Any]],
    *,
    key: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    left_by_seed = _seed_index(left_rows)
    right_by_seed = _seed_index(right_rows)
    common_seeds = sorted(set(left_by_seed).intersection(right_by_seed), key=lambda seed: int(seed))
    left_values: list[float] = []
    right_values: list[float] = []
    paired_seeds: list[int] = []
    left_wins = 0
    right_wins = 0
    ties = 0

    for seed in common_seeds:
        left_value = left_by_seed[seed].get(key)
        right_value = right_by_seed[seed].get(key)
        if not (
            isinstance(left_value, int | float)
            and math.isfinite(left_value)
            and isinstance(right_value, int | float)
            and math.isfinite(right_value)
        ):
            continue

        left_number = float(left_value)
        right_number = float(right_value)
        left_values.append(left_number)
        right_values.append(right_number)
        paired_seeds.append(int(seed))
        if left_number == right_number:
            ties += 1
        elif (left_number > right_number) == higher_is_better:
            left_wins += 1
        else:
            right_wins += 1

    if not left_values:
        return {
            "paired_seed_count": 0,
            "common_seeds": [],
            "higher_is_better": higher_is_better,
            "left_better_count": 0,
            "right_better_count": 0,
            "tie_count": 0,
        }

    left_mean = mean(left_values)
    right_mean = mean(right_values)
    if left_mean == right_mean:
        winner = "tie"
    elif (left_mean > right_mean) == higher_is_better:
        winner = "left"
    else:
        winner = "right"

    return {
        "paired_seed_count": len(left_values),
        "common_seeds": paired_seeds,
        "higher_is_better": higher_is_better,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "left_minus_right": left_mean - right_mean,
        "left_better_count": left_wins,
        "right_better_count": right_wins,
        "tie_count": ties,
        "winner_by_mean": winner,
    }


def _build_pairwise_comparison(
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    left_rows: list[Mapping[str, Any]],
    right_rows: list[Mapping[str, Any]],
    *,
    maximize: bool,
) -> dict[str, Any]:
    best_fitness = _paired_metric_summary(
        left_rows,
        right_rows,
        key="best_fitness",
        higher_is_better=maximize,
    )
    mean_fitness = _paired_metric_summary(
        left_rows,
        right_rows,
        key="mean_fitness",
        higher_is_better=maximize,
    )
    final_generation = _paired_metric_summary(
        left_rows,
        right_rows,
        key="final_generation",
        higher_is_better=False,
    )
    runtime_seconds = _paired_metric_summary(
        left_rows,
        right_rows,
        key="runtime_seconds",
        higher_is_better=False,
    )
    return {
        "left_label": left_record["baseline_label"],
        "right_label": right_record["baseline_label"],
        "maximize": maximize,
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "final_generation": final_generation,
        "runtime_seconds": runtime_seconds,
    }


def build_comparison_groups(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        signature = tuple(row.get(field) for field in _COMPARABLE_GROUP_FIELDS)
        grouped_rows.setdefault(signature, []).append(row)

    groups: list[dict[str, Any]] = []
    for signature in sorted(grouped_rows):
        group_rows = grouped_rows[signature]
        label_groups: dict[str, list[Mapping[str, Any]]] = {}
        for row in group_rows:
            label = str(row.get("baseline_label") or row.get("run_name") or "unlabeled")
            label_groups.setdefault(label, []).append(row)

        baseline_records = [
            _baseline_record(label, label_groups[label]) for label in sorted(label_groups)
        ]
        maximize = bool(group_rows[0].get("maximize", True))
        paired = []
        if len(baseline_records) > 1:
            record_index = {record["baseline_label"]: record for record in baseline_records}
            for left_label, right_label in combinations(sorted(label_groups), 2):
                paired.append(
                    _build_pairwise_comparison(
                        record_index[left_label],
                        record_index[right_label],
                        label_groups[left_label],
                        label_groups[right_label],
                        maximize=maximize,
                    )
                )

        groups.append(
            {
                "signature": _group_signature(group_rows),
                "baseline_labels": [record["baseline_label"] for record in baseline_records],
                "run_count": len(group_rows),
                "best_by_mean_best_fitness": _winner(
                    baseline_records,
                    "mean_best_fitness",
                    higher_is_better=maximize,
                ),
                "best_by_mean_mean_fitness": _winner(
                    baseline_records,
                    "mean_mean_fitness",
                    higher_is_better=maximize,
                ),
                "best_by_mean_final_generation": _winner(
                    baseline_records,
                    "mean_final_generation",
                    higher_is_better=False,
                ),
                "baselines": baseline_records,
                "paired_comparisons": paired,
            }
        )
    return groups


def build_collection_summary(
    collection_name: str,
    rows: list[Mapping[str, Any]],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = str(row.get("baseline_label") or row.get("run_name") or "unlabeled")
        grouped.setdefault(label, []).append(row)

    baselines = [_baseline_record(label, grouped[label]) for label in sorted(grouped)]
    return {
        "suite_name": collection_name,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "baseline_count": len(baselines),
        "run_count": len(rows),
        "baselines": baselines,
        "comparison_groups": build_comparison_groups(rows),
        "runs": [dict(row) for row in rows],
    }


def build_baseline_suite_summary(
    suite_name: str,
    manifest_path: str | Path,
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return build_collection_summary(suite_name, rows, manifest_path=manifest_path)
