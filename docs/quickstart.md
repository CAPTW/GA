# Quickstart

This page now starts with the portable consumer path. You can run these commands from a directory
that is not the repo root after installing the package.

## 3-Minute Consumer Path

Install from a checkout:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install .
```

Or install from a built wheel:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install dist/ga_codex_lab-0.1.0-py3-none-any.whl
```

Then change into any working directory you want and use the packaged commands below.

## 1) First Recommendation

```bash
ga-lab-recommend-solver --problem onemax --size 128 --priority default --format json
ga-lab-recommend-preset --problem zdt1 --size 50 --priority hv --format json
```

What to look for:

- `solver_family`
- `solver_name`
- `resource_uri`
- `portable_command_hint`

## 1b) First Python API Call

```python
from ga_lab.api import recommend_solver, run_preset

recommendation = recommend_solver("onemax", 128, "default")
result = run_preset("onemax_small", output_root="ga-lab-outputs")

print(recommendation.solver_name)
print(result.metrics["best_fitness"])
```

Use `ga_lab.api` for stable library imports. The CLI and Python consumer paths share the same
portable backend now.

## 2) First Built-In Preset Run

```bash
ga-lab-run --preset onemax_small --output-root ./ga-lab-outputs
```

Expected output:

- a printed `summary.json` payload
- an `Output directory: ...` line
- `best_fitness = 32.0` on the small onemax preset smoke

## 3) First Baseline Demo

```bash
ga-lab-demo baseline --output-root ./ga-lab-outputs
```

Read:

- `baseline_label`
- `baseline_success_rate`
- `ga_success_rate`

## 4) First Hybrid Demo

```bash
ga-lab-demo hybrid --output-root ./ga-lab-outputs
```

Read this through:

- `best_route_distance`
- `hybrid_extra_evaluations`

This remains the narrow internal TSP medium quality-first path, not the practical default.

## 5) First NSGA-II Demo

```bash
ga-lab-demo nsga2 --output-root ./ga-lab-outputs
```

Read this through:

- `hypervolume`
- `pareto_ratio`
- `spread`
- `pareto_front_size`

## 6) Version / Help

```bash
ga-lab-run --version
ga-lab-demo --help
ga-lab-run --list-presets
```

## Repo-Mode Advanced Path

Use this only when you have a checkout and want maintainer features.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .[dev]
```

Repo-only commands:

```bash
ga-lab-check-claims --fail-on FAIL
ga-lab-render-release-artifacts
python scripts/run_ci_benchmarks.py --tier tier1
```

## What To Read Next

- [Install](install.md)
- [FAQ](faq.md)
- [Python API](python_api.md)
- [API stability](api_stability.md)
- [Examples](../examples/README.md)
- [Project card](project_card.md)
- [Benchmark card](benchmark_card.md)
- [Solver matrix](solver_matrix.md)
- [Reproducibility and governance](reproducibility_and_governance.md)
