# Install

`ga-codex-lab` now has three explicit execution modes. They do not all support the same commands.

| Mode | Who it is for | Install shape | What it is good for |
| --- | --- | --- | --- |
| Mode A: repo dev mode | contributors, experiment authors | `pip install -e .[dev]` | code changes, tests, benchmark regeneration, governance maintenance |
| Mode B: installed consumer mode | external users, evaluators, portfolio reviewers | `pip install .` or `pip install dist/*.whl` | packaged presets, packaged demos, solver recommendation helpers, out-of-tree runs |
| Mode C: maintainer / release mode | maintainers | repo checkout + dev deps | claim drift checks, release artifact rendering, benchmark tier workflows |

## Supported Python

- package requirement: `>=3.11`
- portable smoke workflow target: Python `3.11` on Windows, Linux, and macOS

## Mode A: Repo Dev Install

Use this when you want to edit code, rerun tests, or regenerate checked-in evidence artifacts.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .[dev]
```

Typical repo-dev commands:

```bash
python -m pytest
python scripts/check_claim_drift.py --fail-on FAIL
python scripts/render_release_artifacts.py
python scripts/run_ci_benchmarks.py --tier tier1
```

## Mode B: Installed Consumer Install

Use this when you want a lightweight, package-supported path that works outside the repo root.

From a checkout:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install .
```

From a built release artifact:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install dist/ga_codex_lab-0.1.0-py3-none-any.whl
```

Portable commands supported from any working directory:

```bash
ga-lab-run --list-presets
ga-lab-run --preset onemax_small --output-root ./ga-lab-outputs
ga-lab-demo baseline --output-root ./ga-lab-outputs
ga-lab-demo hybrid --output-root ./ga-lab-outputs
ga-lab-demo nsga2 --output-root ./ga-lab-outputs
ga-lab-recommend-preset --problem onemax --size 32 --priority default --format json
ga-lab-recommend-solver --problem tsp --size 50 --priority default --format json
```

Stable Python import path in this mode:

```python
from ga_lab.api import list_presets, recommend_solver, run_preset
```

You do not need `GA_LAB_PROJECT_ROOT` for those packaged commands anymore.

## Mode C: Maintainer / Release Install

Use this when you need checked-in summaries, docs marker regeneration, or benchmark-governance workflows.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .[dev,release]
```

Repo-only commands in this mode:

```bash
ga-lab-check-claims --fail-on FAIL
ga-lab-render-release-artifacts
python scripts/package_portability_smoke.py --dist-dir dist
```

## What Is Portable vs Repo-Only

Portable consumer commands:

- `ga-lab-run`
- `ga-lab-demo`
- `ga-lab-recommend-preset`
- `ga-lab-recommend-solver`

Repo-only maintainer commands:

- `ga-lab-check-claims`
- `ga-lab-render-release-artifacts`
- benchmark regeneration via `scripts/run_ci_benchmarks.py`
- external benchmark fetch and deeper governance tiers

## Stable Import Boundary

Stable library namespace:

- `ga_lab.api`

Compatibility-only or internal modules:

- `ga_lab.consumer_cli`
- `ga_lab.resources`
- `ga_lab.recommendations`
- `scripts/*`

Library users should import the stable facade instead of reaching into implementation modules.

## Direct Script Fallback

The `scripts/` commands are still available in repo mode when you want explicit repo-relative control:

```bash
python scripts/run_experiment.py --preset onemax_small --output-root outputs
python scripts/run_demo_suite.py --demo pure-ga --output-root outputs/demo
python scripts/check_claim_drift.py
python scripts/render_release_artifacts.py
```
