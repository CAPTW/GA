from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_public_api_python_example_runs_out_of_tree(tmp_path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_project_root() / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from ga_lab.api import recommend_solver, run_preset; "
                "recommendation = recommend_solver('onemax', 128, 'default', format='dict'); "
                "result = run_preset('onemax_small', output_root='api_outputs'); "
                "print(json.dumps({'solver_name': recommendation['solver_name'], "
                "'best_fitness': result.metrics['best_fitness']}))"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["solver_name"] == "hill_climb"
    assert payload["best_fitness"] == 32.0
    assert (tmp_path / "api_outputs").exists()
