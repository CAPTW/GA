from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class SuiteEntry:
    label: str
    config_data: dict[str, Any]
    config_source: str
    seed_start: int
    seeds: int


def _resolve_config_path(manifest_path: Path, config_ref: str) -> Path:
    candidate = Path(config_ref)
    if candidate.is_absolute():
        return candidate

    for base_dir in (PROJECT_ROOT, manifest_path.parent):
        resolved = (base_dir / candidate).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Config not found: {config_ref}")


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        elif isinstance(value, Mapping):
            merged[key] = _deep_merge({}, value)
        else:
            merged[key] = value
    return merged


def _set_dotted_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not parts:
        raise ValueError("Override key must not be empty")
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def apply_overrides(base_data: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = _deep_merge(dict(base_data), {})
    for key, value in overrides.items():
        if "." in key:
            _set_dotted_value(merged, key, value)
        elif isinstance(value, Mapping):
            existing = merged.get(key)
            if isinstance(existing, dict):
                merged[key] = _deep_merge(existing, value)
            else:
                merged[key] = _deep_merge({}, value)
        else:
            merged[key] = value
    return merged


def _default_label(values: Mapping[str, Any]) -> str:
    parts = []
    for key, value in values.items():
        safe_key = key.replace(".", "_")
        safe_value = str(value).replace(" ", "_")
        parts.append(f"{safe_key}_{safe_value}")
    return "__".join(parts) if parts else "variant"


def _render_label(template: str | None, values: Mapping[str, Any]) -> str:
    if not template:
        return _default_label(values)
    context = {key.replace(".", "_"): value for key, value in values.items()}
    return template.format_map(context)


def _seed_defaults(payload: Mapping[str, Any], default_seeds: int) -> tuple[int, int]:
    seed_start = int(payload.get("seed_start", payload.get("seed", 0)))
    seeds = int(payload.get("seeds", default_seeds))
    if seeds <= 0:
        raise ValueError("Suite entry seeds must be positive")
    return seed_start, seeds


def _expand_manual_entry(
    raw_entry: Mapping[str, Any],
    *,
    manifest_path: Path,
    default_seeds: int,
) -> SuiteEntry:
    config_ref = raw_entry.get("config")
    if not isinstance(config_ref, str) or not config_ref.strip():
        raise ValueError("Each manifest entry requires a non-empty config path")
    config_path = _resolve_config_path(manifest_path, config_ref.strip())
    config_data = _load_json_dict(config_path)
    overrides = raw_entry.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError("Manifest entry overrides must be a JSON object")
    config_data = apply_overrides(config_data, overrides)

    label = raw_entry.get("label", config_data.get("run_name", config_path.stem))
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Each manifest entry label must be a non-empty string")
    if "run_name" not in config_data:
        config_data["run_name"] = label.strip()
    seed_start, seeds = _seed_defaults(
        {
            "seed_start": raw_entry.get("seed_start", config_data.get("seed", 0)),
            "seeds": raw_entry.get("seeds", default_seeds),
        },
        default_seeds,
    )
    return SuiteEntry(
        label=label.strip(),
        config_data=config_data,
        config_source=str(config_path),
        seed_start=seed_start,
        seeds=seeds,
    )


def _expand_matrix_entry(
    matrix: Mapping[str, Any],
    *,
    manifest_path: Path,
    default_seeds: int,
) -> list[SuiteEntry]:
    base_config_ref = matrix.get("base_config")
    if not isinstance(base_config_ref, str) or not base_config_ref.strip():
        raise ValueError("Matrix entries require a non-empty base_config path")
    base_config_path = _resolve_config_path(manifest_path, base_config_ref.strip())
    base_data = _load_json_dict(base_config_path)

    shared_overrides = matrix.get("shared_overrides", {})
    if shared_overrides is None:
        shared_overrides = {}
    if not isinstance(shared_overrides, Mapping):
        raise ValueError("matrix.shared_overrides must be a JSON object")

    axes = matrix.get("axes", {})
    if axes is None:
        axes = {}
    if not isinstance(axes, Mapping):
        raise ValueError("matrix.axes must be a JSON object")

    axis_names = list(axes.keys())
    axis_values = []
    for axis_name in axis_names:
        values = axes[axis_name]
        if not isinstance(values, list) or not values:
            raise ValueError("Each matrix axis must be a non-empty array")
        axis_values.append(values)

    label_template = matrix.get("label_template")
    if label_template is not None and not isinstance(label_template, str):
        raise ValueError("matrix.label_template must be a string when provided")

    seed_start, seeds = _seed_defaults(
        {
            "seed_start": matrix.get("seed_start", base_data.get("seed", 0)),
            "seeds": matrix.get("seeds", default_seeds),
        },
        default_seeds,
    )

    entries = []
    for values in product(*axis_values) if axis_values else [()]:
        combo = dict(zip(axis_names, values, strict=False))
        overrides = apply_overrides(shared_overrides, combo)
        config_data = apply_overrides(base_data, overrides)
        label = _render_label(label_template, combo)
        config_data["run_name"] = str(config_data.get("run_name") or label)
        if config_data["run_name"] == base_data.get("run_name"):
            config_data["run_name"] = label
        entries.append(
            SuiteEntry(
                label=label,
                config_data=config_data,
                config_source=str(base_config_path),
                seed_start=seed_start,
                seeds=seeds,
            )
        )
    return entries


def load_suite_manifest(
    manifest_path: str | Path,
    *,
    default_seeds: int = 1,
) -> tuple[dict[str, Any], list[SuiteEntry]]:
    path = Path(manifest_path).resolve()
    manifest = _load_json_dict(path)
    suite_name = manifest.get("suite_name", path.stem)
    if not isinstance(suite_name, str) or not suite_name.strip():
        raise ValueError("Manifest.suite_name must be a non-empty string")
    manifest["suite_name"] = suite_name.strip()

    raw_entries = manifest.get("entries", [])
    matrices = []
    if "matrix" in manifest:
        matrices.append(manifest["matrix"])
    if "matrices" in manifest:
        extra = manifest["matrices"]
        if not isinstance(extra, list):
            raise ValueError("Manifest.matrices must be an array")
        matrices.extend(extra)

    if not raw_entries and not matrices:
        raise ValueError("Manifest requires entries, matrix, or matrices")
    if raw_entries and not isinstance(raw_entries, list):
        raise ValueError("Manifest.entries must be an array")

    entries: list[SuiteEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Each manifest entry must be a JSON object")
        entries.append(
            _expand_manual_entry(
                raw_entry,
                manifest_path=path,
                default_seeds=default_seeds,
            )
        )
    for matrix in matrices:
        if not isinstance(matrix, Mapping):
            raise ValueError("Each matrix definition must be a JSON object")
        entries.extend(
            _expand_matrix_entry(
                matrix,
                manifest_path=path,
                default_seeds=default_seeds,
            )
        )

    return manifest, entries
