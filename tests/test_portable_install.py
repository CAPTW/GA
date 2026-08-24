from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _portable_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(_project_root() / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.pop("GA_LAB_PROJECT_ROOT", None)
    return env


def test_recommend_preset_works_out_of_tree() -> None:
    # NOTE: the conftest `tmp_path` fixture is intentionally workspace-local
    # (<repo>/.pytest-local-tmp/...) for Windows temp-dir cleanup, so it lives *inside* the repo
    # and cannot exercise the genuine out-of-tree path (find_project_root would correctly resolve
    # the repo root). Use a real system temp dir instead and assert it is outside the project root.
    with tempfile.TemporaryDirectory(prefix="ga_lab_portable_out_of_tree_") as tmpdir:
        out_of_tree_cwd = Path(tmpdir).resolve()
        repo_root = _project_root()
        assert not out_of_tree_cwd.is_relative_to(repo_root), (
            f"expected an out-of-tree cwd, but {out_of_tree_cwd} is under {repo_root}"
        )

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from ga_lab.consumer_cli import recommend_preset_main; "
                    "raise SystemExit(recommend_preset_main(["
                    "'--problem','onemax','--size','32','--priority','default','--format','json'"
                    "]))"
                ),
            ],
            cwd=out_of_tree_cwd,
            env=_portable_env(),
            capture_output=True,
            text=True,
            check=True,
        )

    payload = json.loads(completed.stdout)
    assert payload["resource_uri"] == "preset:onemax_small"
    assert payload["preset_abspath"] is None


def test_run_builtin_preset_works_out_of_tree(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ga_lab.consumer_cli import run_experiment_main; "
                "raise SystemExit(run_experiment_main(["
                "'--preset','onemax_small','--output-root','portable_outputs'"
                "]))"
            ),
        ],
        cwd=tmp_path,
        env=_portable_env(),
        capture_output=True,
        text=True,
        check=True,
    )

    output_dir_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("Output directory: ")
    )
    output_dir = Path(output_dir_line.split(": ", 1)[1])
    if not output_dir.is_absolute():
        output_dir = (tmp_path / output_dir).resolve()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert output_dir.exists()
    assert summary["best_fitness"] == 32.0


def test_baseline_demo_works_out_of_tree(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ga_lab.consumer_cli import demo_main; "
                "raise SystemExit(demo_main(["
                "'--demo','baseline','--output-root','portable_demo'"
                "]))"
            ),
        ],
        cwd=tmp_path,
        env=_portable_env(),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["baseline_label"] == "hill_climb"
    assert Path(payload["summary_path"]).exists()
