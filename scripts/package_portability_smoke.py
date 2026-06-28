from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import textwrap
import venv
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _script_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _script_path(venv_dir: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _script_dir(venv_dir) / f"{name}{suffix}"


def _python_path(venv_dir: Path) -> Path:
    return _script_path(venv_dir, "python")


def _run(
    command: list[str],
    *,
    cwd: Path,
    expected_field: str | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    payload: dict[str, Any] = {
        "command": command,
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_tail": completed.stdout.strip().splitlines()[-10:],
        "stderr_tail": completed.stderr.strip().splitlines()[-10:],
    }
    if expected_field is not None:
        payload["expected_field"] = expected_field
    return payload


def _parse_json_stdout(stdout: str) -> dict[str, Any]:
    return json.loads(stdout)


def _parse_output_dir(stdout: str) -> Path:
    for line in reversed(stdout.splitlines()):
        if line.startswith("Output directory: "):
            return Path(line.split(": ", 1)[1].strip())
    raise ValueError("Could not find 'Output directory:' line in ga-lab-run output")


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("evolutionary_solver_benchmark_lab-*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No evolutionary_solver_benchmark_lab wheel found in {dist_dir}")
    return wheels[-1]


def _wheel_inventory(wheel_path: Path) -> dict[str, Any]:
    required = {
        "ga_lab/builtin_resources/presets/onemax_small.json",
        "ga_lab/builtin_resources/presets/tsp_medium_hybrid.json",
        "ga_lab/builtin_resources/demo/onemax_pure_ga_demo.json",
        "ga_lab/builtin_resources/demo/baseline_onemax_demo_manifest.json",
    }
    with zipfile.ZipFile(wheel_path) as handle:
        names = set(handle.namelist())
    return {
        "wheel": str(wheel_path),
        "required_members_present": all(name in names for name in required),
        "required_members": sorted(required),
    }


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Portable Install Smoke",
        "",
        f"- wheel: `{summary['wheel']}`",
        f"- temp workdir: `{summary['temp_root']}`",
        f"- consumer cwd: `{summary['consumer_cwd']}`",
        "- required packaged resources present in wheel: "
        f"`{summary['wheel_inventory']['required_members_present']}`",
        "",
        "| step | key field | value |",
        "| --- | --- | --- |",
    ]
    for row in summary["checks"]:
        lines.append(f"| {row['step']} | {row['field']} | `{row['value']}` |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean venv, install the wheel, "
            "and verify portable out-of-tree commands."
        )
    )
    parser.add_argument(
        "--wheel",
        default=None,
        help="Explicit wheel path. Defaults to dist/*.whl.",
    )
    parser.add_argument(
        "--dist-dir",
        default="dist",
        help="Directory that contains the built wheel.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/portable_smoke",
        help="Directory where smoke summaries will be written.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else output_root / "portable_install_smoke.json"
    )
    output_md = (
        Path(args.output_md).resolve()
        if args.output_md
        else output_root / "portable_install_smoke.md"
    )

    wheel_path = (
        Path(args.wheel).resolve()
        if args.wheel
        else _default_wheel(Path(args.dist_dir).resolve())
    )
    temp_root = Path(tempfile.mkdtemp(prefix="ga_lab_portable_"))
    venv_dir = temp_root / "venv"
    consumer_cwd = temp_root / "consumer_cwd"
    consumer_cwd.mkdir(parents=True, exist_ok=True)

    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    python_path = _python_path(venv_dir)
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [str(python_path), "-m", "pip", "install", str(wheel_path)],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    run_cmd = str(_script_path(venv_dir, "ga-lab-run"))
    demo_cmd = str(_script_path(venv_dir, "ga-lab-demo"))
    recommend_preset_cmd = str(_script_path(venv_dir, "ga-lab-recommend-preset"))
    recommend_solver_cmd = str(_script_path(venv_dir, "ga-lab-recommend-solver"))

    version_step = _run([run_cmd, "--version"], cwd=consumer_cwd)
    help_step = _run([demo_cmd, "--help"], cwd=consumer_cwd)
    api_step = _run(
        [
            str(python_path),
            "-c",
            textwrap.dedent(
                """
                import json
                from ga_lab.api import (
                    get_version,
                    list_demos,
                    list_presets,
                    recommend_solver,
                    run_preset,
                )

                result = run_preset("onemax_small", output_root="portable_outputs/api")
                payload = {
                    "version": get_version(),
                    "preset_count": len(list_presets()),
                    "demo_count": len(list_demos()),
                    "solver_name": recommend_solver(
                        "onemax", 128, "default", format="dict"
                    )["solver_name"],
                    "best_fitness": result.metrics.get("best_fitness"),
                }
                print(json.dumps(payload))
                """
            ),
        ],
        cwd=consumer_cwd,
        expected_field="best_fitness",
    )
    api_payload = _parse_json_stdout(str(api_step["stdout"]))
    list_presets_step = _run([run_cmd, "--list-presets"], cwd=consumer_cwd)
    list_presets_payload = json.loads(str(list_presets_step["stdout"]))

    recommend_preset_step = _run(
        [
            recommend_preset_cmd,
            "--problem",
            "onemax",
            "--size",
            "32",
            "--priority",
            "default",
            "--format",
            "json",
        ],
        cwd=consumer_cwd,
        expected_field="resource_uri",
    )
    recommend_preset_payload = _parse_json_stdout(str(recommend_preset_step["stdout"]))

    recommend_solver_step = _run(
        [
            recommend_solver_cmd,
            "--problem",
            "tsp",
            "--size",
            "50",
            "--priority",
            "default",
            "--format",
            "json",
        ],
        cwd=consumer_cwd,
        expected_field="solver_name",
    )
    recommend_solver_payload = _parse_json_stdout(str(recommend_solver_step["stdout"]))

    run_step = _run(
        [
            run_cmd,
            "--preset",
            "onemax_small",
            "--output-root",
            "portable_outputs/runs",
        ],
        cwd=consumer_cwd,
        expected_field="best_fitness",
    )
    run_output_dir = _parse_output_dir(str(run_step["stdout"]))
    if not run_output_dir.is_absolute():
        run_output_dir = (consumer_cwd / run_output_dir).resolve()
    run_summary = _load_summary(run_output_dir / "summary.json")

    baseline_step = _run(
        [demo_cmd, "--demo", "baseline", "--output-root", "portable_outputs/demo"],
        cwd=consumer_cwd,
        expected_field="baseline_label",
    )
    baseline_payload = json.loads(str(baseline_step["stdout"]))

    hybrid_step = _run(
        [demo_cmd, "--demo", "hybrid", "--output-root", "portable_outputs/demo"],
        cwd=consumer_cwd,
        expected_field="best_route_distance",
    )
    hybrid_payload = json.loads(str(hybrid_step["stdout"]))

    nsga2_step = _run(
        [demo_cmd, "--demo", "nsga2", "--output-root", "portable_outputs/demo"],
        cwd=consumer_cwd,
        expected_field="hypervolume",
    )
    nsga2_payload = json.loads(str(nsga2_step["stdout"]))

    summary = {
        "mode": "portable_wheel_install",
        "wheel": str(wheel_path),
        "temp_root": str(temp_root),
        "consumer_cwd": str(consumer_cwd),
        "wheel_inventory": _wheel_inventory(wheel_path),
        "steps": {
            "version": version_step,
            "help": help_step,
            "python_api": api_step,
            "list_presets": list_presets_step,
            "recommend_preset": recommend_preset_step,
            "recommend_solver": recommend_solver_step,
            "run_preset": run_step,
            "demo_baseline": baseline_step,
            "demo_hybrid": hybrid_step,
            "demo_nsga2": nsga2_step,
        },
        "checks": [
            {
                "step": "python_api",
                "field": "best_fitness",
                "value": api_payload["best_fitness"],
            },
            {
                "step": "python_api",
                "field": "solver_name",
                "value": api_payload["solver_name"],
            },
            {
                "step": "list_presets",
                "field": "contains_onemax_small",
                "value": any(
                    row["name"] == "onemax_small" for row in list_presets_payload["builtin_presets"]
                ),
            },
            {
                "step": "recommend_preset",
                "field": "resource_uri",
                "value": recommend_preset_payload["resource_uri"],
            },
            {
                "step": "recommend_solver",
                "field": "solver_name",
                "value": recommend_solver_payload["solver_name"],
            },
            {"step": "run_preset", "field": "best_fitness", "value": run_summary["best_fitness"]},
            {
                "step": "demo_baseline",
                "field": "baseline_label",
                "value": baseline_payload["baseline_label"],
            },
            {
                "step": "demo_hybrid",
                "field": "best_route_distance",
                "value": hybrid_payload["best_route_distance"],
            },
            {"step": "demo_nsga2", "field": "hypervolume", "value": nsga2_payload["hypervolume"]},
        ],
    }

    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(_markdown_summary(summary), encoding="utf-8")
    print(
        textwrap.dedent(
            f"""
            Wrote portable smoke summary:
            - {output_json}
            - {output_md}
            """
        ).strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
