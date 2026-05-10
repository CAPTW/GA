# Python API

`ga-codex-lab` now freezes a small public Python API for package consumers at
[`ga_lab.api`](../src/ga_lab/api.py).

That module is the only stable import path for library users.

## Stable Public Surface

- `get_version()`
- `list_presets()`
- `list_demos()`
- `load_builtin_preset(name)`
- `load_builtin_demo(name)`
- `recommend_preset(problem, size, priority, family=None, format="python")`
- `recommend_solver(problem, size, priority, family=None, format="python")`
- `run_preset(name, output_root, overrides=None)`
- `run_config(config, output_root)`
- `run_demo(name, output_root)`

Returned objects are intentionally small:

- `PresetInfo`
- `DemoInfo`
- `RecommendationResult`
- `RunResultSummary`

## Minimal Example

```python
from ga_lab.api import recommend_solver, run_preset

recommendation = recommend_solver("onemax", 128, "default")
print(recommendation.solver_name)

result = run_preset("onemax_small", output_root="ga-lab-outputs")
print(result.metrics["best_fitness"])
print(result.summary_path)
```

## Dict / JSON Output

The recommendation helpers can also return compatibility-friendly payloads:

```python
from ga_lab.api import recommend_preset

payload = recommend_preset("zdt1", 50, "hv", format="dict")
print(payload["resource_uri"])
```

Allowed `format` values:

- `"python"`: dataclass result
- `"dict"`: JSON-serializable mapping
- `"json"`: formatted JSON string

## Stable Scope

The stable portable API is intentionally narrow.

- It supports the installed consumer path.
- It works out-of-tree.
- It does not promote repo-only governance helpers into the public library contract.

`family` is reserved in the signature so the call shape does not need to change later, but
family-conditioned benchmark semantics still live in the docs and evidence summaries rather than
the stable portable API.

## Not Stable For Library Imports

These remain implementation details:

- `ga_lab.consumer_cli`
- `ga_lab.resources`
- `ga_lab.recommendations`
- `ga_lab.governance.*`
- `scripts/*`

Use the public docs when you need family-conditioned or maintainer-mode guidance:

- [API stability](api_stability.md)
- [Quickstart](quickstart.md)
- [Solver choice guide](solver_choice_guide.md)
- [Reproducibility and governance](reproducibility_and_governance.md)
