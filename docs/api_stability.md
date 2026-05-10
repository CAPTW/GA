# API Stability

This repo now has a small frozen public surface for package consumers.

## Versioning Stance

The package stays in the `0.1.x` line for now.

That means:

- the repo is not claiming a broad 1.0 platform surface
- the public API is intentionally small
- compatibility promises apply only to the frozen public namespace and public consumer CLI

## Stable Namespace

Stable library namespace:

- [`ga_lab.api`](../src/ga_lab/api.py)

Stable consumer CLI:

- `ga-lab-run`
- `ga-lab-demo`
- `ga-lab-recommend-preset`
- `ga-lab-recommend-solver`

## Experimental vs Internal vs Repo-Only

Public experimental:

- none are promoted as stable library imports right now

Internal / implementation-only:

- `ga_lab.consumer_cli`
- `ga_lab.resources`
- `ga_lab.recommendations`
- `ga_lab.demo`
- `scripts/run_experiment.py`
- `scripts/run_demo_suite.py`
- `scripts/recommend_preset.py`
- `scripts/recommend_solver.py`

Repo-only maintainer path:

- `ga-lab-check-claims`
- `ga-lab-render-release-artifacts`
- benchmark regeneration and benchmark governance scripts

## Breaking Change Rule

Within `0.1.x`, the stable public API should not remove names or change callable signatures without
an explicit deprecation step.

The stable contract is tracked by:

- [`artifacts/public_api_snapshot.json`](../artifacts/public_api_snapshot.json)
- [`scripts/check_public_api.py`](../scripts/check_public_api.py)

## Deprecation Policy

When a public stable symbol or CLI behavior needs to change:

1. remove it from recommendation docs first only if a better path exists
2. keep the symbol or command working during at least one minor release transition unless there is
   a correctness or security reason not to
3. add a snapshot update and test change in the same PR that formally changes the contract
4. note the change in release notes and API docs

Compatibility wrappers may continue to exist for repo users, but they are not a stability promise.

## CLI Compatibility Rule

The stable CLI should remain a thin wrapper over the same backend used by `ga_lab.api`.

That means:

- recommendation payloads should stay aligned
- preset and demo execution should return the same core result fields
- portable consumer commands should keep working out-of-tree

## Why Snapshot Drift Matters

This repo already governs benchmark claims with registry and drift checks.
The public API snapshot extends the same idea to the library contract:

- accidental symbol removal becomes visible in CI
- silent signature changes become visible in CI
- docs/examples can point at one stable import path with less ambiguity
