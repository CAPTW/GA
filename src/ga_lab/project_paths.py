from __future__ import annotations

import os
from pathlib import Path


def looks_like_project_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").is_file()
        and (path / "scripts").is_dir()
        and (path / "src" / "ga_lab").is_dir()
    )


def find_project_root(start: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    env_root = os.environ.get("GA_LAB_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).resolve())
    if start is not None:
        candidates.append(Path(start).resolve())
    cwd = Path.cwd().resolve()
    if cwd not in candidates:
        candidates.append(cwd)

    for base in candidates:
        for candidate in [base, *base.parents]:
            if looks_like_project_root(candidate):
                return candidate
    return None
