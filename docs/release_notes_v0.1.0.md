# Release Notes v0.1.0

## Highlights

- Public-facing project card, benchmark card, solver matrix, and release status pages
- Machine-readable release artifacts generated from the claim registry and drift report
- README redesigned as a landing page for external readers
- Governance-aware packaging for portfolio, OSS, and collaboration review

## Supported Problem Families

- bitstring families:
  - monotone subset
  - deceptive trap tested subset
  - Jump_k tested subset
- knapsack:
  - validated internal path
  - family-conditioned external slices from kplib
- tsp:
  - validated internal path
  - representative TSPLIB subset
- zdt family:
  - validated zdt1 presets
  - tested ZDT2 / ZDT3 external support subset

## Solver-Choice Summary

- monotone bitstring:
  - practical default is hill climbing
- deceptive trap tested family:
  - a representative pure GA path may be better
- knapsack:
  - keep the guidance family-conditioned
- tsp:
  - practical default is nearest-neighbor + 2-opt
  - `tsp_medium_hybrid.json` stays narrow and internal-only
- zdt:
  - pure NSGA-II stays the default path

## Evidence Governance Summary

- claim registry: `claims/claim_registry.json`
- drift checker: `scripts/check_claim_drift.py`
- release renderer: `scripts/render_release_artifacts.py`
- benchmark tiers:
  - PR: fast smoke + governance checks
  - nightly: internal regression subset
  - weekly/manual: deeper confirm suite

## Known Limitations

- validated ranges should not be generalized beyond the tested scope
- knapsack does not have one broad external-wide default
- `tsp_medium_hybrid.json` is not an external official path
- large-tier ZDT remains tradeoff-based

## Reproducibility Note

Release-facing artifacts are generated from checked-in summaries, the claim registry, and the latest
drift report. Future reruns can refresh the same artifact set without rewriting the narrative from
scratch.

## Still Experimental

- Jump_k broad solver rule
- knapsack subset-sum structured-family rule
- large-tier ZDT cheap HV-first note
