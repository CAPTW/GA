from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "benchmarks"


BENCHMARK_SOURCES: dict[str, dict[str, Any]] = {
    "tsplib": {
        "name": "TSPLIB95 mirror",
        "problem": "tsp",
        "provenance": (
            "TSPLIB instances from Reinelt (1991), fetched through the public "
            "mastqe/tsplib mirror because the original Heidelberg host is not "
            "reliably reachable in automation."
        ),
        "primary_url": "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/",
        "reference_url": "https://github.com/mastqe/tsplib",
        "license_note": (
            "The mirror does not publish an explicit repository license. This repo "
            "treats TSPLIB files as fetch-only cache artifacts and does not commit them."
        ),
        "redistribution_policy": "fetch_only",
    },
    "kplib": {
        "name": "kplib",
        "problem": "knapsack",
        "provenance": (
            "0/1 knapsack benchmark instances generated according to Kellerer, "
            "Pferschy, and Pisinger (2004), provided by the likr/kplib repository."
        ),
        "primary_url": "https://github.com/likr/kplib",
        "reference_url": "https://github.com/likr/kplib",
        "license_note": "CC BY 4.0 as stated in the kplib README.",
        "redistribution_policy": "fetch_or_cache",
    },
    "synthetic_bitstring": {
        "name": "Synthetic bitstring families",
        "problem": "onemax",
        "provenance": (
            "Canonical synthetic benchmark families implemented locally: OneMax, "
            "LeadingOnes, deceptive concatenated trap, and Jump_k."
        ),
        "primary_url": None,
        "reference_url": None,
        "license_note": "No external dataset; deterministic local problem definitions only.",
        "redistribution_policy": "local_definition",
    },
    "zdt_family": {
        "name": "ZDT family",
        "problem": "zdt1",
        "provenance": (
            "Canonical ZDT multi-objective benchmark family implemented locally "
            "(ZDT1, ZDT2, ZDT3)."
        ),
        "primary_url": None,
        "reference_url": None,
        "license_note": "No external dataset; deterministic analytical benchmark family.",
        "redistribution_policy": "local_definition",
    },
}


BENCHMARK_INSTANCES: dict[str, dict[str, Any]] = {
    "tsplib_ulysses22": {
        "problem": "tsp",
        "family": "tsplib",
        "display_name": "ulysses22",
        "size": 22,
        "cache_relpath": "cache/tsplib/ulysses22.tsp",
        "source_url": "https://raw.githubusercontent.com/mastqe/tsplib/master/ulysses22.tsp",
        "parser": "tsplib",
        "reference_note": "GEO instance, close to the internal medium TSP scale.",
    },
    "tsplib_berlin52": {
        "problem": "tsp",
        "family": "tsplib",
        "display_name": "berlin52",
        "size": 52,
        "cache_relpath": "cache/tsplib/berlin52.tsp",
        "source_url": "https://raw.githubusercontent.com/mastqe/tsplib/master/berlin52.tsp",
        "parser": "tsplib",
        "reference_note": "EUC_2D instance, close to the internal large TSP scale.",
    },
    "kplib_uncorrelated_50_s000": {
        "problem": "knapsack",
        "family": "kplib_uncorrelated",
        "display_name": "uncorrelated_n50_r1000_s000",
        "size": 50,
        "cache_relpath": "cache/kplib/00Uncorrelated/n00050/R01000/s000.kp",
        "source_url": "https://raw.githubusercontent.com/likr/kplib/master/00Uncorrelated/n00050/R01000/s000.kp",
        "parser": "kplib",
        "reference_note": "Canonical uncorrelated profit/weight family.",
    },
    "kplib_weakly_correlated_50_s000": {
        "problem": "knapsack",
        "family": "kplib_weakly_correlated",
        "display_name": "weakly_correlated_n50_r1000_s000",
        "size": 50,
        "cache_relpath": "cache/kplib/01WeaklyCorrelated/n00050/R01000/s000.kp",
        "source_url": "https://raw.githubusercontent.com/likr/kplib/master/01WeaklyCorrelated/n00050/R01000/s000.kp",
        "parser": "kplib",
        "reference_note": "Canonical weakly correlated family.",
    },
    "kplib_strongly_correlated_100_s000": {
        "problem": "knapsack",
        "family": "kplib_strongly_correlated",
        "display_name": "strongly_correlated_n100_r1000_s000",
        "size": 100,
        "cache_relpath": "cache/kplib/02StronglyCorrelated/n00100/R01000/s000.kp",
        "source_url": "https://raw.githubusercontent.com/likr/kplib/master/02StronglyCorrelated/n00100/R01000/s000.kp",
        "parser": "kplib",
        "reference_note": "Canonical strongly correlated family.",
    },
    "kplib_subset_sum_100_s000": {
        "problem": "knapsack",
        "family": "kplib_subset_sum",
        "display_name": "subset_sum_n100_r1000_s000",
        "size": 100,
        "cache_relpath": "cache/kplib/05SubsetSum/n00100/R01000/s000.kp",
        "source_url": "https://raw.githubusercontent.com/likr/kplib/master/05SubsetSum/n00100/R01000/s000.kp",
        "parser": "kplib",
        "reference_note": "Subset-sum-like structured family.",
    },
}


def _cache_path(instance_id: str, cache_root: str | Path | None = None) -> Path:
    instance = BENCHMARK_INSTANCES[instance_id]
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    return (root / instance["cache_relpath"]).resolve()


def benchmark_metadata_payload() -> dict[str, Any]:
    return {
        "cache_root": str(DEFAULT_CACHE_ROOT.resolve()),
        "sources": BENCHMARK_SOURCES,
        "instances": {
            instance_id: {
                **instance,
                "cache_path": str(_cache_path(instance_id)),
            }
            for instance_id, instance in BENCHMARK_INSTANCES.items()
        },
    }


def benchmark_inventory_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id, instance in sorted(BENCHMARK_INSTANCES.items()):
        source_key = "tsplib" if instance["family"] == "tsplib" else "kplib"
        source = BENCHMARK_SOURCES[source_key]
        rows.append(
            {
                "instance_id": instance_id,
                "problem": instance["problem"],
                "family": instance["family"],
                "display_name": instance["display_name"],
                "size": instance["size"],
                "source_name": source["name"],
                "source_url": instance["source_url"],
                "license_note": source["license_note"],
                "redistribution_policy": source["redistribution_policy"],
                "cache_path": str(_cache_path(instance_id)),
                "reference_note": instance["reference_note"],
            }
        )
    return rows


def ensure_benchmark_files(
    instance_ids: list[str] | tuple[str, ...] | None = None,
    *,
    cache_root: str | Path | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    selected_ids = list(instance_ids) if instance_ids is not None else list(BENCHMARK_INSTANCES)
    plan: list[dict[str, Any]] = []
    for instance_id in selected_ids:
        if instance_id not in BENCHMARK_INSTANCES:
            raise ValueError(f"Unknown benchmark instance: {instance_id}")
        instance = BENCHMARK_INSTANCES[instance_id]
        cache_path = _cache_path(instance_id, cache_root)
        exists = cache_path.exists()
        plan.append(
            {
                "instance_id": instance_id,
                "cache_path": str(cache_path),
                "source_url": instance["source_url"],
                "already_present": exists,
                "action": "skip" if exists else "download",
            }
        )
        if dry_run or exists:
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(instance["source_url"], cache_path)
    return plan


def _parse_kplib_text(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("kplib file is too short")
    num_items = int(lines[0])
    capacity = float(lines[1])
    rows = lines[2 : 2 + num_items]
    if len(rows) != num_items:
        raise ValueError("kplib file does not contain the declared number of items")
    values: list[float] = []
    weights: list[float] = []
    for row in rows:
        profit_text, weight_text, *_rest = row.split()
        values.append(float(profit_text))
        weights.append(float(weight_text))
    return {
        "num_items": num_items,
        "capacity": capacity,
        "values": values,
        "weights": weights,
    }


def _geo_to_radians(value: float) -> float:
    degrees = int(value)
    minutes = value - degrees
    return math.pi * (degrees + (5.0 * minutes / 3.0)) / 180.0


def _distance_from_coords(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    edge_weight_type: str,
) -> float:
    if edge_weight_type == "EUC_2D":
        return float(int(round(math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))))
    if edge_weight_type == "CEIL_2D":
        return float(int(math.ceil(math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))))
    if edge_weight_type == "ATT":
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]
        rij = math.sqrt((dx * dx + dy * dy) / 10.0)
        tij = int(round(rij))
        return float(tij + 1 if tij < rij else tij)
    if edge_weight_type == "GEO":
        latitude_a = _geo_to_radians(point_a[0])
        longitude_a = _geo_to_radians(point_a[1])
        latitude_b = _geo_to_radians(point_b[0])
        longitude_b = _geo_to_radians(point_b[1])
        q1 = math.cos(longitude_a - longitude_b)
        q2 = math.cos(latitude_a - latitude_b)
        q3 = math.cos(latitude_a + latitude_b)
        radius = 6378.388
        distance = radius * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
        return float(int(distance))
    raise ValueError(f"Unsupported TSPLIB EDGE_WEIGHT_TYPE: {edge_weight_type}")


def _build_distance_matrix(
    coordinates: list[tuple[float, float]],
    edge_weight_type: str,
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row_idx, row_point in enumerate(coordinates):
        row: list[float] = []
        for col_idx, col_point in enumerate(coordinates):
            if row_idx == col_idx:
                row.append(0.0)
            else:
                row.append(_distance_from_coords(row_point, col_point, edge_weight_type))
        matrix.append(row)
    return matrix


def _parse_tsplib_text(text: str) -> dict[str, Any]:
    header: dict[str, str] = {}
    coordinate_rows: list[str] = []
    in_coord_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "EOF":
            continue
        if in_coord_section:
            coordinate_rows.append(line)
            continue
        if line == "NODE_COORD_SECTION":
            in_coord_section = True
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip().upper()] = value.strip()
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            header[parts[0].strip().upper()] = parts[1].strip()

    edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "").upper()
    if edge_weight_type not in {"EUC_2D", "CEIL_2D", "ATT", "GEO"}:
        raise ValueError(f"Unsupported or missing EDGE_WEIGHT_TYPE: {edge_weight_type!r}")

    dimension = int(header["DIMENSION"])
    coordinates: list[tuple[float, float]] = []
    for row in coordinate_rows:
        index_text, x_text, y_text, *_rest = row.split()
        index = int(index_text)
        if index != len(coordinates) + 1:
            raise ValueError("TSPLIB NODE_COORD_SECTION must be 1-indexed and ordered")
        coordinates.append((float(x_text), float(y_text)))
    if len(coordinates) != dimension:
        raise ValueError("TSPLIB NODE_COORD_SECTION length does not match DIMENSION")

    return {
        "name": header.get("NAME", "unknown"),
        "num_cities": dimension,
        "edge_weight_type": edge_weight_type,
        "coordinates": [[x, y] for x, y in coordinates],
        "distance_matrix": _build_distance_matrix(coordinates, edge_weight_type),
    }


def _read_instance_file(
    instance_id: str,
    cache_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    if instance_id not in BENCHMARK_INSTANCES:
        raise ValueError(f"Unknown benchmark instance: {instance_id}")
    instance = BENCHMARK_INSTANCES[instance_id]
    path = _cache_path(instance_id, cache_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark cache file is missing: {path}. Run scripts/fetch_benchmarks.py first."
        )
    text = path.read_text(encoding="utf-8")
    if instance["parser"] == "kplib":
        return _parse_kplib_text(text), path
    if instance["parser"] == "tsplib":
        return _parse_tsplib_text(text), path
    raise ValueError(f"Unsupported benchmark parser: {instance['parser']}")


def load_benchmark_problem_overrides(
    instance_id: str,
    *,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    payload, path = _read_instance_file(instance_id, cache_root)
    instance = BENCHMARK_INSTANCES[instance_id]
    if instance["problem"] == "knapsack":
        return {
            "genome_length": payload["num_items"],
            "problem_options": {
                "num_items": payload["num_items"],
                "weights": payload["weights"],
                "values": payload["values"],
                "capacity": payload["capacity"],
                "instance_name": instance["display_name"],
                "instance_source": str(path),
            },
        }
    if instance["problem"] == "tsp":
        return {
            "genome_length": payload["num_cities"],
            "problem_options": {
                "num_cities": payload["num_cities"],
                "distance_matrix": payload["distance_matrix"],
                "instance_name": payload["name"],
                "instance_source": str(path),
            },
        }
    raise ValueError(f"Unsupported problem for benchmark overrides: {instance['problem']}")


def write_metadata_file(path: str | Path) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(benchmark_metadata_payload(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target
