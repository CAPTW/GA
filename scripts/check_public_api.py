from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SNAPSHOT_SCHEMA_VERSION = 1
DEFAULT_SNAPSHOT = PROJECT_ROOT / "artifacts" / "public_api_snapshot.json"
MODULE_PATH = "ga_lab.api"


def _annotation_name(value: Any) -> str:
    if value is inspect.Signature.empty:
        return "empty"
    return str(value)


def _symbol_record(name: str, obj: Any) -> dict[str, Any]:
    if inspect.isfunction(obj):
        signature = inspect.signature(obj)
        return {
            "name": name,
            "kind": "function",
            "signature": str(signature),
            "return_annotation": _annotation_name(signature.return_annotation),
            "docstring_present": bool(inspect.getdoc(obj)),
        }

    if inspect.isclass(obj):
        record = {
            "name": name,
            "kind": "class",
            "signature": None,
            "return_annotation": None,
            "docstring_present": bool(inspect.getdoc(obj)),
        }
        if is_dataclass(obj):
            record["fields"] = [
                {"name": field.name, "type": _annotation_name(field.type)}
                for field in fields(obj)
            ]
        return record

    return {
        "name": name,
        "kind": type(obj).__name__,
        "signature": None,
        "return_annotation": None,
        "docstring_present": bool(inspect.getdoc(obj)),
    }


def snapshot_payload() -> dict[str, Any]:
    module = importlib.import_module(MODULE_PATH)
    exports = []
    for name in module.__all__:
        exports.append(_symbol_record(name, getattr(module, name)))
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "module_path": MODULE_PATH,
        "status": "stable_public_api",
        "exports": sorted(exports, key=lambda item: item["name"]),
    }


def _load_snapshot(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _export_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in payload["exports"]}


def compare_snapshot(current: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    current_exports = _export_map(current)
    expected_exports = _export_map(expected)

    missing = sorted(set(expected_exports) - set(current_exports))
    added = sorted(set(current_exports) - set(expected_exports))
    for name in missing:
        messages.append(f"missing export: {name}")
    for name in added:
        messages.append(f"new export: {name}")

    for name in sorted(set(current_exports) & set(expected_exports)):
        current_item = current_exports[name]
        expected_item = expected_exports[name]
        for key in ("kind", "signature", "return_annotation", "fields"):
            if current_item.get(key) != expected_item.get(key):
                messages.append(
                    f"changed {name}.{key}: expected {expected_item.get(key)!r}, "
                    f"got {current_item.get(key)!r}"
                )
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the stable public ga_lab.api surface against a checked-in snapshot."
    )
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument(
        "--update-snapshot",
        action="store_true",
        help="Write the current public API payload to the snapshot path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_path = Path(args.snapshot).resolve()
    current = snapshot_payload()

    if args.update_snapshot:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(current, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Updated public API snapshot: {snapshot_path}")
        return 0

    if not snapshot_path.exists():
        print(
            f"Snapshot not found: {snapshot_path}\n"
            "Run with --update-snapshot to create the checked-in contract file.",
            file=sys.stderr,
        )
        return 2

    expected = _load_snapshot(snapshot_path)
    drift = compare_snapshot(current, expected)
    if drift:
        print("Public API drift detected:", file=sys.stderr)
        for message in drift:
            print(f"- {message}", file=sys.stderr)
        return 1

    print(f"Public API snapshot check passed: {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
