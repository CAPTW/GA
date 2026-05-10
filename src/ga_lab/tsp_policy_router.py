from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ga_lab.tsp_instance_features import extract_tsp_instance_features_from_problem_options

_POLICIES = ("none", "low_diversity_injection", "decay_mutation")
_POLICY_COMPLEXITY = {"none": 0, "decay_mutation": 1, "low_diversity_injection": 2}


@dataclass(frozen=True, slots=True)
class TSPRoutingRule:
    trigger_min_cities: int
    trigger_bridge_threshold: float
    trigger_bridge_budget_mode: str
    trigger_ring_anisotropy_max: float
    trigger_ring_nn_cv_max: float
    decay_anisotropy_min: float
    decay_bridge_max: float

    def predict(self, record: dict[str, Any]) -> str:
        budget_band = str(record.get("budget_band", ""))
        if (
            float(record["bridge_score"]) >= self.trigger_bridge_threshold
            and int(record["num_cities"]) >= self.trigger_min_cities
            and (
                self.trigger_bridge_budget_mode == "any"
                or budget_band == self.trigger_bridge_budget_mode
            )
        ):
            return "low_diversity_injection"
        if (
            float(record["pca_anisotropy_ratio"]) <= self.trigger_ring_anisotropy_max
            and float(record["nn_distance_cv"]) <= self.trigger_ring_nn_cv_max
        ):
            return "low_diversity_injection"
        if (
            float(record["pca_anisotropy_ratio"]) >= self.decay_anisotropy_min
            and float(record["bridge_score"]) <= self.decay_bridge_max
        ):
            return "decay_mutation"
        return "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_min_cities": self.trigger_min_cities,
            "trigger_bridge_threshold": self.trigger_bridge_threshold,
            "trigger_bridge_budget_mode": self.trigger_bridge_budget_mode,
            "trigger_ring_anisotropy_max": self.trigger_ring_anisotropy_max,
            "trigger_ring_nn_cv_max": self.trigger_ring_nn_cv_max,
            "decay_anisotropy_min": self.decay_anisotropy_min,
            "decay_bridge_max": self.decay_bridge_max,
        }

    def render_lines(self) -> list[str]:
        budget_prefix = (
            f"budget band is `{self.trigger_bridge_budget_mode}` and "
            if self.trigger_bridge_budget_mode != "any"
            else ""
        )
        return [
            "1. if "
            + budget_prefix
            + f"`bridge_score >= {self.trigger_bridge_threshold:.3f}` "
            + f"and `num_cities >= {self.trigger_min_cities}` "
            + "-> `low_diversity_injection`",
            "2. elif "
            + f"`pca_anisotropy_ratio <= {self.trigger_ring_anisotropy_max:.3f}` "
            + f"and `nn_distance_cv <= {self.trigger_ring_nn_cv_max:.3f}` "
            + "-> `low_diversity_injection`",
            "3. elif "
            + f"`pca_anisotropy_ratio >= {self.decay_anisotropy_min:.3f}` "
            + f"and `bridge_score <= {self.decay_bridge_max:.3f}` "
            + "-> `decay_mutation`",
            "4. else -> `none`",
        ]


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        return plt
    except Exception:
        return None


def _plot_scatter(
    *,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]]]],
) -> bool:
    plt = _import_pyplot()
    if plt is None or not series:
        return False
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, points in series:
        if not points:
            continue
        axis.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            label=label,
            alpha=0.8,
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    if any(points for _, points in series):
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _plot_bar(
    *,
    output_path: Path,
    title: str,
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    y_label: str,
) -> bool:
    plt = _import_pyplot()
    if plt is None or not x_labels or not series:
        return False
    figure, axis = plt.subplots(figsize=(8, 4))
    positions = list(range(len(x_labels)))
    width = 0.8 / max(1, len(series))
    for index, (label, values) in enumerate(series):
        offsets = [pos + (index - (len(series) - 1) / 2) * width for pos in positions]
        axis.bar(offsets, values, width=width, label=label)
    axis.set_xticks(positions)
    axis.set_xticklabels(x_labels, rotation=20, ha="right")
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(max(0.0, variance))


def _budget_band_map(*study_dirs: Path) -> dict[int, str]:
    budgets: set[int] = set()
    for study_dir in study_dirs:
        payload = _load_json_dict(study_dir / "raw_results.json")
        for row in payload.get("rows", []):
            budget = row.get("configured_budget")
            if isinstance(budget, int):
                budgets.add(budget)
    ordered = sorted(budgets)
    mapping: dict[int, str] = {}
    if len(ordered) == 1:
        mapping[ordered[0]] = "current"
    elif len(ordered) == 2:
        mapping[ordered[0]] = "reduced"
        mapping[ordered[1]] = "current"
    else:
        for index, budget in enumerate(ordered):
            if index == 0:
                mapping[budget] = "reduced"
            elif index == len(ordered) - 1:
                mapping[budget] = "current"
            else:
                mapping[budget] = f"mid_{index}"
    return mapping


def _tie_margin(best_distance: float) -> float:
    return max(1.0, best_distance * 0.005)


def _policy_sort_key(performance: dict[str, Any]) -> tuple[float, float, int, float]:
    return (
        float(performance["mean_distance"]),
        float(performance["std_distance"]),
        _POLICY_COMPLEXITY[str(performance["policy"])],
        float(performance["mean_runtime_seconds"]),
    )


def _threshold_candidates(values: list[float], defaults: list[float]) -> list[float]:
    unique_values = sorted({round(value, 6) for value in values})
    midpoints = [
        round((left + right) / 2.0, 6)
        for left, right in zip(unique_values, unique_values[1:], strict=False)
    ]
    return sorted({*defaults, *unique_values, *midpoints})


def _load_case_runs(study_dir: Path, budget_map: dict[int, str]) -> list[dict[str, Any]]:
    payload = _load_json_dict(study_dir / "raw_results.json")
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"raw_results.json rows missing at {study_dir}")
    enriched_rows: list[dict[str, Any]] = []
    feature_cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        policy = row.get("algorithm_options.adaptive_policy")
        case_id = row.get("case_id")
        budget = row.get("configured_budget")
        if policy not in _POLICIES or not isinstance(case_id, str) or not isinstance(budget, int):
            continue
        if case_id not in feature_cache:
            config_path = Path(str(row["output_dir"])) / "config.canonical.json"
            config_payload = _load_json_dict(config_path)
            problem = config_payload.get("problem", {})
            problem_options = problem.get("options", {}) if isinstance(problem, dict) else {}
            feature_cache[case_id] = extract_tsp_instance_features_from_problem_options(
                case_id,
                problem_options,
            ).to_dict()
        enriched = dict(row)
        enriched.update(feature_cache[case_id])
        enriched["budget_band"] = budget_map.get(budget, f"budget_{budget}")
        enriched_rows.append(enriched)
    return enriched_rows


def label_policy_winners(study_dir: Path, *, budget_map: dict[int, str]) -> list[dict[str, Any]]:
    raw_rows = _load_case_runs(study_dir, budget_map)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((str(row["case_id"]), int(row["configured_budget"])), []).append(row)

    labels: list[dict[str, Any]] = []
    for (case_id, configured_budget), rows in sorted(grouped.items()):
        per_policy: list[dict[str, Any]] = []
        for policy in _POLICIES:
            matching = [row for row in rows if row["algorithm_options.adaptive_policy"] == policy]
            if not matching:
                continue
            distances = [float(row["best_route_distance"]) for row in matching]
            runtimes = [float(row["runtime_seconds"]) for row in matching]
            per_policy.append(
                {
                    "policy": policy,
                    "mean_distance": _mean(distances),
                    "std_distance": _std(distances),
                    "mean_runtime_seconds": _mean(runtimes),
                }
            )
        if len(per_policy) < 2:
            continue
        ranked = sorted(per_policy, key=_policy_sort_key)
        best = ranked[0]
        second = ranked[1]
        margin = float(second["mean_distance"]) - float(best["mean_distance"])
        tie_threshold = _tie_margin(float(best["mean_distance"]))
        winner = min(
            [
                performance
                for performance in per_policy
                if float(performance["mean_distance"]) - float(best["mean_distance"])
                <= tie_threshold
            ],
            key=_policy_sort_key,
        )
        base_row = rows[0]
        none_distance = float(
            next(item["mean_distance"] for item in per_policy if item["policy"] == "none")
        )
        trigger_distance = float(
            next(
                item["mean_distance"]
                for item in per_policy
                if item["policy"] == "low_diversity_injection"
            )
        )
        decay_distance = float(
            next(item["mean_distance"] for item in per_policy if item["policy"] == "decay_mutation")
        )
        label_row = {
            "study_dir": str(study_dir.resolve()),
            "case_id": case_id,
            "case_note": base_row.get("case_note", ""),
            "configured_budget": configured_budget,
            "budget_band": budget_map.get(configured_budget, f"budget_{configured_budget}"),
            "winner_policy": str(winner["policy"]),
            "margin_to_second_best": margin,
            "tie_flag": margin <= tie_threshold,
            "best_route_distance_oracle": float(best["mean_distance"]),
            "num_cities": int(base_row["num_cities"]),
            "bbox_aspect_ratio": float(base_row["bbox_aspect_ratio"]),
            "pca_anisotropy_ratio": float(base_row["pca_anisotropy_ratio"]),
            "nn_distance_cv": float(base_row["nn_distance_cv"]),
            "bridge_score": float(base_row["bridge_score"]),
            "radial_distance_cv": float(base_row["radial_distance_cv"]),
            "trigger_value": min(none_distance, decay_distance) - trigger_distance,
            "decay_value": min(none_distance, trigger_distance) - decay_distance,
        }
        for performance in per_policy:
            policy = str(performance["policy"])
            label_row[f"distance_{policy}"] = float(performance["mean_distance"])
            label_row[f"distance_std_{policy}"] = float(performance["std_distance"])
            label_row[f"runtime_{policy}"] = float(performance["mean_runtime_seconds"])
        labels.append(label_row)
    return labels


def evaluate_router(
    rule: TSPRoutingRule,
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
) -> dict[str, Any]:
    prediction_rows: list[dict[str, Any]] = []
    regrets: list[float] = []
    wrong = 0
    tie_excluded_total = 0
    tie_excluded_correct = 0
    runtimes: list[float] = []
    for row in rows:
        predicted_policy = rule.predict(row)
        oracle_policy = str(row["winner_policy"])
        regret = float(row[f"distance_{predicted_policy}"]) - float(
            row["best_route_distance_oracle"]
        )
        regrets.append(regret)
        if predicted_policy != oracle_policy:
            wrong += 1
        if not bool(row["tie_flag"]):
            tie_excluded_total += 1
            if predicted_policy == oracle_policy:
                tie_excluded_correct += 1
        runtimes.append(float(row[f"runtime_{predicted_policy}"]))
        prediction_rows.append(
            {
                "dataset": dataset_name,
                "case_id": row["case_id"],
                "budget_band": row["budget_band"],
                "configured_budget": row["configured_budget"],
                "winner_policy": oracle_policy,
                "predicted_policy": predicted_policy,
                "regret": regret,
                "tie_flag": row["tie_flag"],
                "margin_to_second_best": row["margin_to_second_best"],
                "bridge_score": row["bridge_score"],
                "pca_anisotropy_ratio": row["pca_anisotropy_ratio"],
                "nn_distance_cv": row["nn_distance_cv"],
                "radial_distance_cv": row["radial_distance_cv"],
                "predicted_runtime_seconds": row[f"runtime_{predicted_policy}"],
            }
        )
    return {
        "dataset": dataset_name,
        "strategy": "router",
        "average_regret": _mean(regrets),
        "max_regret": max(regrets) if regrets else 0.0,
        "wrong_policy_rate": wrong / len(rows) if rows else 0.0,
        "tie_excluded_accuracy": (
            tie_excluded_correct / tie_excluded_total if tie_excluded_total else 1.0
        ),
        "mean_predicted_runtime_seconds": _mean(runtimes),
        "prediction_rows": prediction_rows,
    }


def evaluate_fixed_policy(
    policy: str,
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
) -> dict[str, Any]:
    regrets = [
        float(row[f"distance_{policy}"]) - float(row["best_route_distance_oracle"]) for row in rows
    ]
    tie_excluded = [row for row in rows if not bool(row["tie_flag"])]
    return {
        "dataset": dataset_name,
        "strategy": f"always_{policy}",
        "average_regret": _mean(regrets),
        "max_regret": max(regrets) if regrets else 0.0,
        "wrong_policy_rate": (
            sum(1 for row in rows if str(row["winner_policy"]) != policy) / len(rows)
            if rows
            else 0.0
        ),
        "tie_excluded_accuracy": (
            sum(1 for row in tie_excluded if str(row["winner_policy"]) == policy)
            / len(tie_excluded)
            if tie_excluded
            else 1.0
        ),
        "mean_predicted_runtime_seconds": _mean(
            [float(row[f"runtime_{policy}"]) for row in rows]
        ),
        "prediction_rows": [],
    }


def search_simple_routing_rule(train_rows: list[dict[str, Any]]) -> TSPRoutingRule:
    num_cities = sorted({int(row["num_cities"]) for row in train_rows})
    bridge_scores = [float(row["bridge_score"]) for row in train_rows]
    anisotropy = [float(row["pca_anisotropy_ratio"]) for row in train_rows]
    nn_cvs = [float(row["nn_distance_cv"]) for row in train_rows]

    candidates: list[tuple[TSPRoutingRule, dict[str, Any]]] = []
    for trigger_min_cities in num_cities:
        for trigger_bridge_threshold in _threshold_candidates(bridge_scores, [2.5, 3.0, 3.5, 4.0]):
            for trigger_bridge_budget_mode in ("any", "current"):
                for trigger_ring_anisotropy_max in _threshold_candidates(
                    anisotropy,
                    [1.5, 2.0, 2.5, 3.0],
                ):
                    for trigger_ring_nn_cv_max in _threshold_candidates(
                        nn_cvs,
                        [0.08, 0.1, 0.12, 0.18, 0.24],
                    ):
                        for decay_anisotropy_min in _threshold_candidates(
                            anisotropy,
                            [8.0, 10.0, 12.0, 14.0],
                        ):
                            for decay_bridge_max in _threshold_candidates(
                                bridge_scores,
                                [1.2, 1.5, 1.8, 2.0, 2.5],
                            ):
                                rule = TSPRoutingRule(
                                    trigger_min_cities=trigger_min_cities,
                                    trigger_bridge_threshold=trigger_bridge_threshold,
                                    trigger_bridge_budget_mode=trigger_bridge_budget_mode,
                                    trigger_ring_anisotropy_max=trigger_ring_anisotropy_max,
                                    trigger_ring_nn_cv_max=trigger_ring_nn_cv_max,
                                    decay_anisotropy_min=decay_anisotropy_min,
                                    decay_bridge_max=decay_bridge_max,
                                )
                                candidates.append(
                                    (
                                        rule,
                                        evaluate_router(rule, train_rows, dataset_name="train"),
                                    )
                                )
    if not candidates:
        raise ValueError("Could not derive a routing rule")
    min_regret = min(float(evaluation["average_regret"]) for _, evaluation in candidates)
    regret_tolerance = 1.0
    acceptable = [
        (rule, evaluation)
        for rule, evaluation in candidates
        if float(evaluation["average_regret"]) <= min_regret + regret_tolerance
    ]
    acceptable.sort(
        key=lambda item: (
            0 if item[0].trigger_bridge_budget_mode == "any" else 1,
            float(item[1]["wrong_policy_rate"]),
            -float(item[1]["tie_excluded_accuracy"]),
            float(item[1]["mean_predicted_runtime_seconds"]),
            float(item[1]["average_regret"]),
        )
    )
    return acceptable[0][0]


def _instance_feature_rows(*label_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    for label_rows in label_sets:
        for row in label_rows:
            key = (str(row["case_id"]), int(row["configured_budget"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "case_id": row["case_id"],
                    "budget_band": row["budget_band"],
                    "configured_budget": row["configured_budget"],
                    "num_cities": row["num_cities"],
                    "bbox_aspect_ratio": row["bbox_aspect_ratio"],
                    "pca_anisotropy_ratio": row["pca_anisotropy_ratio"],
                    "nn_distance_cv": row["nn_distance_cv"],
                    "bridge_score": row["bridge_score"],
                    "radial_distance_cv": row["radial_distance_cv"],
                    "winner_policy": row["winner_policy"],
                    "tie_flag": row["tie_flag"],
                    "margin_to_second_best": row["margin_to_second_best"],
                    "trigger_value": row["trigger_value"],
                    "decay_value": row["decay_value"],
                }
            )
    return rows


def _render_router_markdown(
    *,
    rule: TSPRoutingRule,
    train_router: dict[str, Any],
    holdout_router: dict[str, Any],
    train_fixed: list[dict[str, Any]],
    holdout_fixed: list[dict[str, Any]],
    holdout_labels: list[dict[str, Any]],
) -> str:
    best_holdout_fixed = min(holdout_fixed, key=lambda row: float(row["average_regret"]))
    lines = [
        "# TSP Policy Router Analysis",
        "",
        "## Selected Rule",
        "",
        *[f"- {line}" for line in rule.render_lines()],
        "",
        "## Train Summary",
        "",
        f"- router average_regret: `{train_router['average_regret']:.4f}`",
        f"- router tie_excluded_accuracy: `{train_router['tie_excluded_accuracy']:.4f}`",
        "",
        "## Holdout Summary",
        "",
        f"- router average_regret: `{holdout_router['average_regret']:.4f}`",
        f"- router tie_excluded_accuracy: `{holdout_router['tie_excluded_accuracy']:.4f}`",
        f"- best fixed strategy on holdout: `{best_holdout_fixed['strategy']}` "
        + f"with average_regret `{best_holdout_fixed['average_regret']:.4f}`",
        (
            "- promotion_read: `no_simple_router_yet`"
            if float(holdout_router["average_regret"])
            > float(best_holdout_fixed["average_regret"])
            else "- promotion_read: `router_candidate_survived_holdout`"
        ),
        "",
        "| strategy | average_regret | tie_excluded_accuracy | wrong_policy_rate "
        "| mean_predicted_runtime_seconds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in [*holdout_fixed, holdout_router]:
        lines.append(
            f"| {row['strategy']} | {row['average_regret']:.4f} | "
            f"{row['tie_excluded_accuracy']:.4f} | {row['wrong_policy_rate']:.4f} | "
            f"{row['mean_predicted_runtime_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Holdout Per-Case Read",
            "",
            "| case_id | budget_band | winner_policy | tie_flag | "
            "margin_to_second_best | bridge_score | "
            "pca_anisotropy_ratio | nn_distance_cv |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in holdout_labels:
        lines.append(
            f"| {row['case_id']} | {row['budget_band']} | {row['winner_policy']} | "
            f"{row['tie_flag']} | {row['margin_to_second_best']:.4f} | {row['bridge_score']:.4f} | "
            f"{row['pca_anisotropy_ratio']:.4f} | {row['nn_distance_cv']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_tsp_router_analysis(
    *,
    train_study_dir: str | Path,
    holdout_study_dir: str | Path,
    budget_study_dir: str | Path | None = None,
    output_root: str | Path = "outputs/local_studies",
) -> dict[str, Any]:
    train_dir = Path(train_study_dir).resolve()
    holdout_dir = Path(holdout_study_dir).resolve()
    extra_dir = Path(budget_study_dir).resolve() if budget_study_dir is not None else None
    budget_map = _budget_band_map(
        train_dir,
        holdout_dir,
        *([extra_dir] if extra_dir is not None else []),
    )

    train_labels = label_policy_winners(train_dir, budget_map=budget_map)
    holdout_labels = label_policy_winners(holdout_dir, budget_map=budget_map)
    budget_labels = (
        label_policy_winners(extra_dir, budget_map=budget_map) if extra_dir is not None else []
    )

    rule = search_simple_routing_rule(train_labels)
    train_router = evaluate_router(rule, train_labels, dataset_name="train")
    holdout_router = evaluate_router(rule, holdout_labels, dataset_name="holdout")
    train_fixed = [
        evaluate_fixed_policy(policy, train_labels, dataset_name="train")
        for policy in _POLICIES
    ]
    holdout_fixed = [
        evaluate_fixed_policy(policy, holdout_labels, dataset_name="holdout")
        for policy in _POLICIES
    ]

    output_dir = (
        Path(output_root)
        / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_tsp_policy_router_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = _instance_feature_rows(train_labels, holdout_labels, budget_labels)
    prediction_rows = [*train_router["prediction_rows"], *holdout_router["prediction_rows"]]
    summary_rows = [train_router, *train_fixed, holdout_router, *holdout_fixed]
    for row in summary_rows:
        row.pop("prediction_rows", None)

    _write_rows_csv(output_dir / "instance_features.csv", feature_rows)
    _write_json(output_dir / "instance_features.json", {"rows": feature_rows})
    _write_rows_csv(output_dir / "policy_labels_train.csv", train_labels)
    _write_rows_csv(output_dir / "policy_labels_holdout.csv", holdout_labels)
    _write_rows_csv(output_dir / "raw_results.csv", prediction_rows)
    _write_json(output_dir / "raw_results.json", {"rows": prediction_rows})
    _write_rows_csv(output_dir / "summary.csv", summary_rows)
    _write_json(
        output_dir / "study_results.json",
        {
            "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "train_study_dir": str(train_dir),
            "holdout_study_dir": str(holdout_dir),
            "budget_study_dir": str(extra_dir) if extra_dir is not None else None,
            "rule": rule.to_dict(),
            "train_labels": train_labels,
            "holdout_labels": holdout_labels,
            "budget_labels": budget_labels,
            "summary_rows": summary_rows,
            "prediction_rows": prediction_rows,
        },
    )

    plots: dict[str, str] = {}
    if _plot_scatter(
        output_path=output_dir / "plot_instance_feature_map.png",
        title="TSP instance feature map",
        x_label="bridge_score",
        y_label="pca_anisotropy_ratio",
        series=[
            (
                policy,
                [
                    (float(row["bridge_score"]), float(row["pca_anisotropy_ratio"]))
                    for row in feature_rows
                    if str(row["winner_policy"]) == policy
                ],
            )
            for policy in _POLICIES
        ],
    ):
        plots["plot_instance_feature_map"] = str(
            (output_dir / "plot_instance_feature_map.png").resolve()
        )
    if _plot_scatter(
        output_path=output_dir / "plot_feature_vs_policy_gap.png",
        title="Feature vs policy margin",
        x_label="bridge_score",
        y_label="margin_to_second_best",
        series=[
            (
                policy,
                [
                    (float(row["bridge_score"]), float(row["margin_to_second_best"]))
                    for row in feature_rows
                    if str(row["winner_policy"]) == policy
                ],
            )
            for policy in _POLICIES
        ],
    ):
        plots["plot_feature_vs_policy_gap"] = str(
            (output_dir / "plot_feature_vs_policy_gap.png").resolve()
        )
    if _plot_scatter(
        output_path=output_dir / "plot_bridge_score_vs_trigger_value.png",
        title="Bridge score vs trigger value",
        x_label="bridge_score",
        y_label="trigger_value",
        series=[
            (
                "train",
                [
                    (float(row["bridge_score"]), float(row["trigger_value"]))
                    for row in train_labels
                ],
            ),
            (
                "holdout",
                [
                    (float(row["bridge_score"]), float(row["trigger_value"]))
                    for row in holdout_labels
                ],
            ),
        ],
    ):
        plots["plot_bridge_score_vs_trigger_value"] = str(
            (output_dir / "plot_bridge_score_vs_trigger_value.png").resolve()
        )
    if _plot_scatter(
        output_path=output_dir / "plot_anisotropy_vs_decay_value.png",
        title="Anisotropy vs decay value",
        x_label="pca_anisotropy_ratio",
        y_label="decay_value",
        series=[
            (
                "train",
                [
                    (float(row["pca_anisotropy_ratio"]), float(row["decay_value"]))
                    for row in train_labels
                ],
            ),
            (
                "holdout",
                [
                    (float(row["pca_anisotropy_ratio"]), float(row["decay_value"]))
                    for row in holdout_labels
                ],
            ),
        ],
    ):
        plots["plot_anisotropy_vs_decay_value"] = str(
            (output_dir / "plot_anisotropy_vs_decay_value.png").resolve()
        )
    if _plot_bar(
        output_path=output_dir / "plot_router_regret.png",
        title="Holdout regret by strategy",
        x_labels=[row["strategy"] for row in holdout_fixed] + ["router"],
        series=[
            (
                "average_regret",
                [float(row["average_regret"]) for row in holdout_fixed]
                + [float(holdout_router["average_regret"])],
            )
        ],
        y_label="average_regret",
    ):
        plots["plot_router_regret"] = str((output_dir / "plot_router_regret.png").resolve())
    budget_bands = sorted({str(row["budget_band"]) for row in feature_rows})
    if _plot_bar(
        output_path=output_dir / "plot_budget_band_vs_policy_win.png",
        title="Budget band vs policy win count",
        x_labels=budget_bands,
        series=[
            (
                policy,
                [
                    sum(
                        1
                        for row in feature_rows
                        if str(row["budget_band"]) == budget_band
                        and str(row["winner_policy"]) == policy
                    )
                    for budget_band in budget_bands
                ],
            )
            for policy in _POLICIES
        ],
        y_label="win_count",
    ):
        plots["plot_budget_band_vs_policy_win"] = str(
            (output_dir / "plot_budget_band_vs_policy_win.png").resolve()
        )

    decision_table_path = output_dir / "router_decision_table.md"
    decision_table_path.write_text(
        "\n".join(["# Router Decision Table", "", *rule.render_lines(), ""]),
        encoding="utf-8",
    )

    summary_md = output_dir / "summary.md"
    summary_md.write_text(
        _render_router_markdown(
            rule=rule,
            train_router=train_router,
            holdout_router=holdout_router,
            train_fixed=train_fixed,
            holdout_fixed=holdout_fixed,
            holdout_labels=holdout_labels,
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir.resolve()),
        "rule": rule.to_dict(),
        "summary_csv": str((output_dir / "summary.csv").resolve()),
        "summary_md": str(summary_md.resolve()),
        "router_decision_table": str(decision_table_path.resolve()),
        "plots": plots,
    }
