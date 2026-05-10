# FAQ

## Do I need a repo checkout to use this package?

No for the lightweight consumer path, yes for the governance and regeneration path.

Portable after install:

- `ga-lab-run`
- `ga-lab-demo`
- `ga-lab-recommend-preset`
- `ga-lab-recommend-solver`

Repo-only:

- `ga-lab-check-claims`
- `ga-lab-render-release-artifacts`
- benchmark regeneration and external benchmark fetch helpers

## Where do the packaged demos and presets come from?

They are shipped as package resources and loaded through Python resource APIs. The installed wheel
includes lightweight preset JSON files and demo configs, so commands such as `ga-lab-run --preset
onemax_small` and `ga-lab-demo baseline` do not need repo-relative config paths.

## Should I use the CLI or the Python API?

Use the CLI when you want one-command runs and shell-friendly JSON.

Use the Python API when you want to import the package from your own code:

```python
from ga_lab.api import recommend_solver, run_preset
```

That is the stable public import path. Internal modules such as `ga_lab.consumer_cli` and
`ga_lab.recommendations` are not the library contract.

## Why are claim drift checks and release artifact rendering still repo-only?

Those commands depend on checked-in summaries, the claim registry, docs marker blocks, and
maintainer-facing governance paths. That is a different mode from lightweight solver consumption, so
the package keeps them explicit instead of pretending they are consumer features.

## Why are repo-only governance commands not part of the stable public API?

Because they depend on checked-in evidence, repo marker blocks, benchmark manifests, and maintainer
workflow assumptions. The stable public API is limited to package-supported consumer tasks that work
out-of-tree.

## Why is a baseline sometimes recommended over GA?

Because this repo is a solver-choice lab, not a "GA must win" demo. If a cheap baseline is the
practical default under matched evaluation budgets, the docs say so directly.

## Why is TSP fitness negative?

Internally, route distance is converted into a maximization-friendly fitness sign. The user-facing
metric is still `best_route_distance`, and that is the field to read first.

## Why should I not read ZDT through one best_fitness number?

ZDT is multi-objective. The important fields are `hypervolume`, `pareto_ratio`, `spread`, and front
size. This repo does not present ZDT as a scalar one-best story.

## Why are some recommendations family-conditional?

Because the evidence splits by family. Bitstring monotone and deceptive families do not behave the
same way, and knapsack families do not support one broad external default either.

## How stable are preset and solver recommendations?

The stable recommendation helpers freeze the portable `problem + size + priority` contract on
validated ranges. Family-conditioned benchmark semantics are still documented in the evidence docs,
but they are not yet promoted to a stable programmable contract.

## What does PASS / WARN / FAIL mean in claim drift?

- `PASS`: the checked-in evidence still satisfies the registry pass condition
- `WARN`: the direction still looks right, but the margin or confidence weakened
- `FAIL`: a CI-gated official claim no longer satisfies its pass condition
- `NOT_EVALUATED`: the required row or summary is missing

## Why does public API snapshot drift matter?

It catches accidental contract changes the same way claim drift catches evidence regressions. If a
stable symbol disappears or a signature changes, CI can flag it before users discover the breakage.

## What should I run first after install?

The fastest success path is:

1. `ga-lab-recommend-solver --problem onemax --size 128 --priority default --format json`
2. `ga-lab-run --preset onemax_small --output-root ./ga-lab-outputs`
3. `ga-lab-demo baseline --output-root ./ga-lab-outputs`
4. `ga-lab-demo nsga2 --output-root ./ga-lab-outputs`
