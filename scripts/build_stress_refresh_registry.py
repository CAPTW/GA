from __future__ import annotations

import argparse
import json
from pathlib import Path

from ga_lab.local_stress_refresh import build_stress_refresh_registry, discover_latest_study_dirs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge refreshed stress-suite outputs into a single current-stress catalog and future target registry."
    )
    parser.add_argument(
        "--study-name",
        action="append",
        default=[],
        help="Study name to resolve from the latest matching output directory under --search-root. Repeat for multiple studies.",
    )
    parser.add_argument(
        "--study-dir",
        action="append",
        default=[],
        help="Explicit study output directory. Repeat for multiple studies.",
    )
    parser.add_argument(
        "--search-root",
        default="outputs/local_studies",
        help="Root directory used to resolve --study-name entries.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/local_studies",
        help="Directory where merged registry artifacts should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    search_root = Path(args.search_root).resolve()
    explicit_dirs = [Path(entry).resolve() for entry in args.study_dir]
    resolved_dirs = list(explicit_dirs)
    if args.study_name:
        resolved_dirs.extend(discover_latest_study_dirs(search_root, args.study_name))
    if not resolved_dirs:
        raise SystemExit("Provide at least one --study-name or --study-dir.")

    payload = build_stress_refresh_registry(
        study_dirs=resolved_dirs,
        output_dir=Path(args.output_dir).resolve(),
    )
    payload["source_study_dirs"] = [str(path) for path in resolved_dirs]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
