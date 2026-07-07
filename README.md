# DP-EOT

Code scaffold for **identity-preserving extended-object tracking of temporarily unresolved target groups**.

The goal is not to use a Dirichlet process as a physical target-cardinality model. The nonparametric component is used for measurement-partition proposal or weighting, while target identity is handled by explicit labels and unresolved-group member sets.

## Working paper claim

Temporarily unresolved groups create an identity problem, not only a localization problem:

```text
A + B  ->  G_{A,B}  ->  A + B
```

A tracker should preserve the identities of `A` and `B` through the merged interval. The intended model separates:

1. **physical measurement generation**: Gamma-Poisson measurement rates, extent-dependent likelihoods, and clutter;
2. **partition inference**: distance, oracle, DP/MFM, or other candidate partition mechanisms;
3. **identity management**: explicit track labels and group member-label sets.

## Initial scope

This repository starts with a deliberately small synthetic benchmark:

- two extended targets,
- constant-velocity motion,
- elliptical measurement clouds,
- a controlled unresolved interval,
- optional clutter,
- metrics for identity preservation and split recovery.

The first falsifiable claim is that an identity-aware unresolved-group model reduces post-split identity switches relative to plain clustering/partitioning baselines at comparable localization error.

## Repository layout

```text
dpeot/
  scenarios/      Synthetic merge/split scenarios.
  models/         Physical measurement/rate/extent models.
  partitions/     Distance, oracle, DP/MFM partition proposals.
  tracking/       Labels, unresolved groups, merge/split hypotheses.
  metrics/        Identity, split-recovery, and localization metrics.
  experiments/    Reproducible experiment entry points.
tests/            Unit tests for scenario and metric primitives.
```

## Minimal run target

Once dependencies are installed, the first experiment entry point is:

```bash
python -m dpeot.experiments.run_two_target_crossing
```

This produces a small Monte Carlo summary for the two-target merge/split scenario.

## Design rule

Every module should support the merge/split identity claim. General-purpose DP tracking, clutter learning, hierarchical scattering-center models, and full PMBM/GLMB machinery should remain out of scope until the core unresolved-group benchmark demonstrates value.
