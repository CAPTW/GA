from __future__ import annotations

import os
import subprocess
import sys

from ga_lab.consumer_cli import (
    demo_main,
    recommend_preset_main,
    recommend_solver_main,
    run_experiment_main,
)
from ga_lab.project_paths import find_project_root


def _run_repo_script(script_name: str) -> int:
    project_root = find_project_root()
    if project_root is None:
        print(
            "This command is maintained in repo mode only. "
            "Run it from the repository checkout or set GA_LAB_PROJECT_ROOT.",
            file=sys.stderr,
        )
        return 2

    script_path = project_root / "scripts" / script_name
    if not script_path.is_file():
        print(f"Missing script: {script_path}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.setdefault("GA_LAB_PROJECT_ROOT", str(project_root))
    completed = subprocess.run(
        [sys.executable, str(script_path), *sys.argv[1:]],
        cwd=project_root,
        env=env,
        check=False,
    )
    return completed.returncode


def run_experiment_entrypoint() -> int:
    return run_experiment_main(sys.argv[1:])


def recommend_preset_entrypoint() -> int:
    return recommend_preset_main(sys.argv[1:])


def recommend_solver_entrypoint() -> int:
    return recommend_solver_main(sys.argv[1:])


def check_claims_entrypoint() -> int:
    return _run_repo_script("check_claim_drift.py")


def render_release_artifacts_entrypoint() -> int:
    return _run_repo_script("render_release_artifacts.py")


def demo_entrypoint() -> int:
    return demo_main(sys.argv[1:])
