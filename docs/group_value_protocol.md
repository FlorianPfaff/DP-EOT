# Group-value study protocol, version 1.1

Revision 1.1 fixes the uniform-clutter support: an unassigned point outside the
declared clutter box has zero likelihood. Version 1 runs are superseded. No
thresholds, scenarios, or decision criteria were changed in this correction;
fresh root seed 20261004 separates all new phases from the inspected version-1
data. Version-1 outputs remain on the compute server for provenance and are not
included in the final paper study.

This protocol supersedes the scientific interpretation of the legacy two-target
benchmark. The legacy simulator supplies true initial states and changes merge
annotations without changing the observations. Its x-order methods are illustrative
failure modes, not competitive identity trackers. Its single-target negative
control is guaranteed by known cardinality. Those results remain reproducible but
must not be used as evidence for the corrected study.

## Question and scope

Does conditioning on aggregate observations help beyond predicting labeled tracks
through ambiguity? If so, does explicitly storing a group add anything beyond a
labeled joint-state filter with the same information?

Two or three tracks are already established. All methods know extents, mean rates,
clutter intensity, and the declared noise/dynamics parameters. Initial estimates
are perturbed by independent Gaussian noise with position/velocity standard
deviations 0.5/0.1; the filter receives the matching prior covariance. This is an
initialized-tracking experiment, not a birth or cardinality experiment. No method
receives future states, origins, or true grouping except the named oracle.

## Generative model and approximations

The 41-scan simulator has dt=1 and white Gaussian acceleration (standard deviation
0.02 or 0.08). The maneuver case adds acceleration (0,0.12) to target A during
scans 18 through 22 while the filter retains the declared constant-velocity model.
Resolution groups are connected components of the graph whose edges connect
target centers at distance less than r. Values r=1,2,4 change the generated data.
Groups emit Poisson clouds with summed rate and a Gaussian shape whose mean and
covariance match the mixture of the member targets, including member separation.
This is a finite-resolution synthetic sensor, not a radar or lidar claim.

Heavy-tail mismatch uses covariance-matched Student t noise with five degrees of
freedom. Noise mismatch doubles sensor noise without telling the filter. Sensor
resolution is recorded but not used to impose truth-like hard grouping gates on
the detector: grouping is selected by the scan likelihood comparison.

For each disjoint allocation of candidate cells to sources, the score uses the
complete Poisson set density: exp(-sum rates - clutter rate), source intensity
factors for allocated points, and uniform clutter intensity for every leftover
point. No per-cell factorial is inserted. Gaussian point likelihoods factor into
scatter and centroid terms, with the centroid covariance including the joint
predicted state covariance. Missed sources retain their zero-count probability.
Group spatial covariance is frozen at the predicted member positions. This is
an assumed-density approximation, not exact nonlinear Bayesian inference.

Distance partitions at thresholds 0.5,0.75,1,1.25,1.5 supply a deduplicated pool of
cells. Each source keeps its top three gated cells plus a missed-cell option;
the squared centroid Mahalanobis gate is 36. Disjoint allocations may combine
cells from different thresholds; every unused point remains clutter. The best
group score must exceed the best resolved score by tau. The selected family is
expanded into a beam of eight hypotheses (one for the single-hypothesis coast
baseline). Candidate generation and group-mode selection are approximations.
Reported retained mass is conditional on this candidate bank, not the full posterior.

The coupled group filter updates the stacked member state using the rate-weighted
centroid operator and a Joseph-form covariance update. Off-diagonal member
covariances persist after separation. The matched labeled joint filter assembles
the same operator from labeled coordinate selectors. Their equivalence is an
explicit null hypothesis and a correctness test, not an independent new algorithm.

## Locked experiments

The conditions are uncertain crossings at each resolution, high acceleration,
maneuvering, unequal rates/extents, three-target partial merging, near miss,
parallel tracks, high clutter, heavy tails, and noise mismatch. Stochastic truth
determines actual grouping, including accidental events in nominal negative
controls. Such events are reported, never discarded or relabeled.

Independent SeedSequence streams separate initialization, motion, and measurements;
root seed is 20261004. Phase, condition, and trial identify each stream. All methods,
thresholds, and beam-budget variants share the exact same trial within a phase.
Pilot/calibration/confirmation/runtime use distinct phase identifiers.

- Pilot: 20 trials per condition at tau=0.
- Calibration: 100 trials per condition, tau in {-10,-5,0,5,10}.
- Confirmation: 200 trials per condition, frozen method-specific thresholds.
- Beam sensitivity: repeat the first 20 confirmation trials with beam size 16.
- Timing: 20 new trials per condition, one worker and one BLAS thread.

Threshold selection maximizes pooled membership F1 under a false-group scan rate
of at most 1% on actually resolved scans of the two negative-control conditions.
Ties favor lower false rates, then smaller absolute thresholds. If no threshold
qualifies, select the lowest-false-rate point only for diagnostics and mark the
constraint as failed. A useful operating point additionally requires recall >=0.8.
Confirmation cannot run without a source-fingerprinted calibration artifact.

## Metrics and decisions

Report trial-level labeled position RMSE over all scans, post-split label recovery,
post-split identity switches, membership precision/recall/F1, detection/release
delay, and false/missed/wrong-member scan counts. After each actual group episode,
post-split scoring stops when its members enter another group. Recovery delay
requires two consecutive correct scans; unrecovered events are censored and their
count is explicit. Trials without eligible split events have missing recovery,
not zero or perfect recovery.

Uncertainty diagnostics are the marginal position coverage of moment-matched
95% ellipses and the joint position Gaussian-mixture negative log density. Ellipse
coverage is a moment diagnostic, not an exact multimodal credible-region claim.
Use 5000 paired bootstrap replicates over independent trials, never over scans.
Each metric reports its eligible-trial count. Runtime from parallel runs is not a
performance claim; use the separate sequential timing experiment.

Mechanism gate: at least 10% RMSE reduction or five percentage points more recovery
against labeled coasting in at least two predeclared difficult conditions; the
95% paired interval excludes zero benefit. Difficult conditions are r4, high
acceleration, maneuver, three targets, clutter, heavy tails, and noise mismatch.
Recovery noninferiority is not established if any condition's interval includes a
loss greater than two percentage points. Calibration, uncertainty failures, and
beam sensitivity must be visible even if the error-based gate passes.

Contribution gate: do not claim a new group-tracking method if the matched labeled
joint filter is equivalent. Examine prior merged-measurement tracking literature
before claiming novelty. Only a distinct demonstrated mechanism, defensible
computation benefit, or derived theoretical result can authorize broader stress
and DP studies. The current implementation intentionally provides no automatic
path from a favorable coast comparison to a contribution claim.

## Execution and artifacts

Use gpuserver4090 (gpuserver6000 fallback), at most 32 processes, and set
OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=OMP_NUM_THREADS=1 before Python starts.
The exporter supports checkpointed per-trial JSON, manifest/source hashes,
aggregate JSON/Markdown/LaTeX, paired intervals, figures, and an explicit verdict.
Do not mix version-1 artifacts with legacy results. Confirmation is frozen:
implementation changes require fresh calibration and must be recorded.

```bash
python -m dpeot.experiments.export_group_value_study --stage pilot --workers 32 --output-dir results/group_value_v1/pilot
python -m dpeot.experiments.export_group_value_study --stage calibrate --workers 32 --output-dir results/group_value_v1/calibrate
python -m dpeot.experiments.export_group_value_study --stage confirm --workers 32 --calibration results/group_value_v1/calibrate/calibration.json --output-dir results/group_value_v1/confirm
python -m dpeot.experiments.export_group_value_study --stage confirm --num-trials 20 --hypotheses 16 --workers 32 --calibration results/group_value_v1/calibrate/calibration.json --output-dir results/group_value_v1/beam16
python -m dpeot.experiments.export_group_value_study --stage runtime --workers 1 --calibration results/group_value_v1/calibrate/calibration.json --output-dir results/group_value_v1/runtime
```

Before launching full runs: pass the truth-isolation, sensor-response, symmetry,
permutation, density-normalization, covariance, and three-target membership tests;
inspect a smoke export and the pilot. Preserve all held-out failures in the report.
