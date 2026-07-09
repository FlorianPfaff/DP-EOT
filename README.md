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

## Minimal benchmark

Install the package in editable mode:

```bash
python -m pip install -e .[dev]
```

Run the first Monte Carlo benchmark:

```bash
python -m dpeot.experiments.run_two_target_crossing --num-trials 100
```

Export paper-ready benchmark artifacts:

```bash
python -m dpeot.experiments.export_two_target_benchmark --num-trials 100 --output-dir results
```

The exporter writes:

```text
results/two_target_benchmark.json
results/two_target_benchmark.md
results/two_target_benchmark_table.tex
```

The benchmark currently compares:

- `distance_collapse`: a baseline that loses identity memory during the unresolved interval;
- `dp_x_order`: an unlabeled Dirichlet-process-style partition baseline with canonical left-to-right labels;
- `mfm_x_order`: an unlabeled finite-cardinality MFM-style partition baseline with canonical left-to-right labels;
- `labeled_split_hypothesis`: a compact labeled hypothesis baseline that keeps competing label-to-cell assignments through the merge;
- `proposed_group_labels`: the identity-aware unresolved-group filter with a likelihood-ratio merge detector;
- `oracle_group_labels`: the same group-label filter with the unresolved interval supplied as a controlled diagnostic;
- `oracle_identity`: a perfect-identity upper bound.

The MFM ablation is included to separate the value of flexible partition inference from any claim that the Dirichlet process prior itself is necessary.

The headline metrics are post-split identity switches, post-split label recovery, split recovery delay, group-detection F1, merge-onset delay, split-release delay, false group scans, unlabeled position error, and runtime per scan. Group-detection F1 is membership-aware: it gives credit only when the detected unresolved member set matches truth exactly. The JSON artifact also keeps group-detection precision/recall, missed group scans, wrong-membership scans, group-membership accuracy during unresolved scans, and total, pre-merge, during-unresolved, post-split, and resolved-period identity-switch diagnostics. Total identity switches are useful for debugging, but post-split switches are the paper-facing identity metric because individual identities may be physically unobservable inside the merged blob.

## Stress sweeps

Run the merge-duration/clutter-rate heatmap slice:

```bash
python -m dpeot.experiments.export_stress_sweep \
  --num-trials 5 \
  --profile heatmap \
  --output-dir results/stress_sweep \
  --figure-dir figures \
  --workers 1
```

The heatmap profile compares `distance_collapse`, `dp_x_order`, `mfm_x_order`, `labeled_split_hypothesis`, and `proposed_group_labels` over merge durations `3, 5, 7, 10, 15` and clutter rates `0, 2, 5, 10`. It writes JSON, CSV, Markdown heatmap tables, and `figures/stress_label_recovery_heatmaps.png`; the JSON/CSV rows include the same group-detection diagnostics as the initial benchmark.

Run the full synthetic stress grid across merge duration, clutter rate, measurement noise, extent similarity, rate asymmetry, and crossing angle:

```bash
python -m dpeot.experiments.export_stress_sweep \
  --num-trials 5 \
  --profile full \
  --output-dir results/stress_sweep_full \
  --figure-dir figures \
  --workers 32
```

## GitHub Actions

The `CI and benchmark` workflow runs on push, pull request, and manual dispatch. It installs the package, runs `pytest`, executes the 100-trial two-target benchmark, and uploads the `two-target-benchmark` artifact containing JSON, Markdown, and LaTeX result files.

## Diagnostic figure

Install the optional plotting dependency:

```bash
python -m pip install -e .[plot]
```

Create the first merge/split scenario figure:

```bash
python -m dpeot.experiments.plot_two_target_timeline --output two_target_merge_split.pdf
```

## Design rule

Every module should support the merge/split identity claim. General-purpose DP tracking, clutter learning, hierarchical scattering-center models, and full PMBM/GLMB machinery should remain out of scope until the core unresolved-group benchmark demonstrates value.
