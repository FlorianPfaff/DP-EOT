from dataclasses import replace

import numpy as np
import pytest

from dpeot.experiments.export_group_value_study import select_thresholds, study_verdict
from dpeot.metrics.group_value import merge_events, paired_interval, summarize_trace
from dpeot.scenarios.group_value import (
    CONDITIONS, ObservationScan, StudyCondition, generate_study_instance,
    group_moments, sample_observations, sensor_grouping,
)
from dpeot.tracking.group_value import (
    Cell, JointHypothesis, SourceCell, StudyFilterConfig, allocation_log_likelihood,
    joint_update, observation_operator, prune_hypotheses, run_study_filter,
)


def instance(steps=12):
    return generate_study_instance(CONDITIONS[1], "pilot", 1, 3, num_steps=steps)


def test_initialization_and_prediction_are_not_oracles():
    trial = instance(41)
    assert not np.array_equal(trial.inputs.initial_mean, trial.truth.states[0].ravel())
    trace = run_study_filter(trial.inputs, "prediction_only")
    metrics = summarize_trace(trial, trace)
    assert metrics["labeled_position_rmse"] > 0.25
    assert not np.array_equal(trial.truth.states[1, :, 2:], trial.truth.states[0, :, 2:])


def test_resolution_changes_observations_not_only_annotations():
    trial = instance()
    states = np.array([[[0., 0., 0., 0.], [2., 0., 0., 0.]]])
    small, small_groups = sample_observations(states, replace(trial.inputs.sensor, resolution=1), np.random.default_rng(5))
    large, large_groups = sample_observations(states, replace(trial.inputs.sensor, resolution=4), np.random.default_rng(5))
    assert small_groups == (((0,), (1,)),)
    assert large_groups == (((0, 1),),)
    assert not np.array_equal(small[0].measurements, large[0].measurements)


def test_sensor_group_covariance_includes_member_separation():
    inputs = instance().inputs
    positions = np.array([[-2., 0.], [2., 0.]])
    center, covariance, rate = group_moments(positions, inputs.sensor, (0, 1))
    np.testing.assert_allclose(center, [0, 0])
    assert rate == 24
    assert covariance[0, 0] > 4.0


def test_joint_update_couples_members_and_preserves_information():
    inputs = instance().inputs
    operator = observation_operator((0, 1), inputs)
    mean, covariance = joint_update(np.zeros(8), np.eye(8), np.array([1., -1.]), operator, 0.1 * np.eye(2))
    assert np.linalg.norm(mean) > 0
    assert covariance[0, 4] < 0
    assert np.linalg.eigvalsh(covariance).min() > 0
    next_mean, next_covariance = joint_update(mean, covariance, np.array([1., -1.]), operator, 0.1 * np.eye(2))
    assert np.trace(next_covariance) < np.trace(covariance)
    assert np.linalg.norm(next_mean) > np.linalg.norm(mean)


def test_labeled_and_group_operators_and_filters_are_equivalent():
    inputs = instance(8).inputs
    np.testing.assert_array_equal(observation_operator((0, 1), inputs), observation_operator((0, 1), inputs, "labeled"))
    group = run_study_filter(inputs, "coupled_group")
    labeled = run_study_filter(inputs, "labeled_joint")
    np.testing.assert_allclose(group.means, labeled.means, atol=1e-10)
    np.testing.assert_allclose(group.covariances, labeled.covariances, atol=1e-10)
    assert group.groupings == labeled.groupings
    for a, b in zip(group.hypothesis_weights, labeled.hypothesis_weights):
        np.testing.assert_allclose(a, b, atol=1e-10)


def test_truth_cannot_leak_through_tracker_interface():
    trial = instance(5)
    before = run_study_filter(trial.inputs, "coupled_group")
    poisoned = replace(trial, truth=replace(trial.truth, states=trial.truth.states + 1000,
                                           groupings=tuple(((0, 1),) for _ in trial.inputs.scans)))
    after = run_study_filter(poisoned.inputs, "coupled_group")
    np.testing.assert_array_equal(before.means, after.means)
    with pytest.raises(ValueError, match="explicit oracle"):
        run_study_filter(trial.inputs, "coupled_group", oracle_groupings=trial.truth.groupings)


def test_measurement_order_and_label_permutation_invariance():
    inputs = instance(5).inputs
    baseline = run_study_filter(inputs, "coupled_group")
    reversed_scans = tuple(replace(scan, measurements=scan.measurements[::-1]) for scan in inputs.scans)
    reordered = run_study_filter(replace(inputs, scans=reversed_scans), "coupled_group")
    np.testing.assert_array_equal(baseline.means, reordered.means)
    indices = np.r_[4:8, 0:4]
    relabeled = replace(inputs, labels=inputs.labels[::-1],
                        initial_mean=inputs.initial_mean[indices],
                        initial_covariance=inputs.initial_covariance[np.ix_(indices, indices)],
                        sensor=replace(inputs.sensor, extents=inputs.sensor.extents[::-1], rates=inputs.sensor.rates[::-1]))
    permuted = run_study_filter(relabeled, "coupled_group")
    np.testing.assert_allclose(baseline.means, permuted.means[:, indices], atol=1e-8)


def test_symmetric_assignments_keep_equal_weights():
    trial = instance(1 + 2)
    inputs = trial.inputs
    mean = np.zeros(8)
    prior = replace(inputs, initial_mean=mean, initial_covariance=np.eye(8),
                    sensor=replace(inputs.sensor, clutter_rate=0),
                    scans=(ObservationScan(0, np.array([[-3., 0.], [-3.1, .1], [3., 0.], [3.1, -.1]])),))
    trace = run_study_filter(prior, "labeled_hypothesis", StudyFilterConfig(threshold=1e6))
    weights = trace.hypothesis_weights[0]
    assert len(weights) >= 2
    np.testing.assert_allclose(weights[:2], [0.5, 0.5], atol=1e-12)


def test_three_target_sensor_and_detector_can_select_a_pair():
    assert sensor_grouping(np.array([[0., 0.], [0.5, 0.], [8., 0.]]), 1.0) == ((0, 1), (2,))
    trial = generate_study_instance(StudyCondition("three", geometry="three"), "pilot", 6, 0, 3)
    initial = np.array([[0., 0., 0., 0.], [.5, 0., 0., 0.], [8., 0., 0., 0.]])
    sensor = replace(trial.inputs.sensor, resolution=1., clutter_rate=0.)
    scans, _ = sample_observations(initial[None], sensor, np.random.default_rng(8))
    inputs = replace(trial.inputs, scans=scans, sensor=sensor, initial_mean=initial.ravel(), initial_covariance=0.01 * np.eye(12))
    trace = run_study_filter(inputs, "coupled_group")
    assert trace.groupings == (((0, 1), (2,)),)


def test_complete_set_density_counts_unused_measurements_and_misses():
    inputs = instance().inputs
    hypothesis = JointHypothesis(inputs.initial_mean, inputs.initial_covariance, 0., ((0,), (1,)))
    empty = allocation_log_likelihood(inputs, hypothesis, (), np.empty((0, 2)))
    assert empty == -26.0
    extra = allocation_log_likelihood(inputs, hypothesis, (), np.zeros((1, 2)))
    assert extra == pytest.approx(empty + np.log(2.0 / 3600.0))
    no_clutter = replace(inputs, sensor=replace(inputs.sensor, clutter_rate=0))
    assert allocation_log_likelihood(no_clutter, hypothesis, (), np.zeros((1, 2))) == -np.inf
    assert allocation_log_likelihood(inputs, hypothesis, (), np.array([[31., 0.]])) == -np.inf


def test_scatter_centroid_likelihood_matches_direct_point_product():
    inputs = instance().inputs
    hypothesis = JointHypothesis(inputs.initial_mean, np.zeros((8, 8)), 0., ((0,), (1,)))
    operator = observation_operator((0,), inputs)
    center, shape, rate = group_moments(hypothesis.mean.reshape(-1, 4)[:, :2], inputs.sensor, (0,))
    points = center + np.array([[.1, .2], [-.1, -.2]])
    cell = Cell((0, 1), points.mean(axis=0), (points - points.mean(axis=0)).T @ (points - points.mean(axis=0)))
    _, ld = np.linalg.slogdet(shape)
    factor = 2 * np.log(rate) - .5 * (2 * np.log(2 * np.pi) + ld + np.trace(np.linalg.solve(shape, cell.scatter)) + 2 * np.log(2))
    option = SourceCell((0,), cell, operator, shape / 2, factor)
    from dpeot.tracking.group_value import gaussian_logpdf
    direct = -26 + 2 * np.log(rate) + sum(gaussian_logpdf(p - center, shape) for p in points)
    assert allocation_log_likelihood(inputs, hypothesis, (option,), points) == pytest.approx(direct)


def test_target_points_outside_the_clutter_box_are_still_allowed():
    inputs = instance().inputs
    hypothesis = JointHypothesis(inputs.initial_mean, inputs.initial_covariance, 0., ((0,), (1,)))
    points = np.array([[31., 0.]])
    cell = Cell((0,), points[0], np.zeros((2, 2)))
    source = SourceCell((0,), cell, observation_operator((0,), inputs), np.eye(2), np.log(12.))
    assert np.isfinite(allocation_log_likelihood(inputs, hypothesis, (source,), points))
    assert allocation_log_likelihood(inputs, hypothesis, (), points) == -np.inf


def test_empty_scans_only_propagate_the_prior_and_covariance():
    inputs = instance(8).inputs
    empty = replace(inputs, scans=tuple(ObservationScan(s.k, np.empty((0, 2))) for s in inputs.scans))
    prediction = run_study_filter(empty, "prediction_only")
    group = run_study_filter(empty, "coupled_group")
    np.testing.assert_array_equal(group.means, prediction.means)
    np.testing.assert_array_equal(group.covariances, prediction.covariances)
    assert np.trace(group.covariances[-1]) > np.trace(group.covariances[0])
    assert not group.proposal_failures.any()


def test_no_merge_is_not_reported_as_perfect_recovery():
    trial = instance(3)
    trace = run_study_filter(trial.inputs, "prediction_only")
    metrics = summarize_trace(trial, trace)
    assert metrics["recovery"] is None
    assert metrics["merge_onset_delay"] is None
    assert metrics["merge_events"] == 0


def test_events_and_paired_intervals_preserve_trial_structure():
    assert merge_events((((0,), (1,)), ((0, 1),), ((0,), (1,)), ((0, 1),))) == [(1, 1, (0, 1)), (3, 3, (0, 1))]
    interval = paired_interval([0.8, 1.6], [1., 2.], relative=True)
    assert interval["estimate"] == pytest.approx(.2)
    assert interval["low"] == pytest.approx(.2)
    assert interval["high"] == pytest.approx(.2)


def test_pruning_records_discarded_candidate_mass():
    a = JointHypothesis(np.zeros(8), np.eye(8), np.log(.7), ((0,), (1,)))
    b = JointHypothesis(np.ones(8), np.eye(8), np.log(.3), ((0,), (1,)))
    retained, mass = prune_hypotheses([a, b], 1)
    assert mass == pytest.approx(.7)
    assert retained[0].log_weight == 0


def test_calibration_exposes_an_infeasible_false_positive_constraint():
    rows = [dict(method="coupled_group", threshold=t, condition=c, resolved_scans=100,
                 false_group_scans=10, group_tp=90, group_fp=10, group_fn=10)
            for t in (-10., -5., 0., 5., 10.) for c in ("near_miss", "parallel")]
    result = select_thresholds(rows)["coupled_group"]
    assert not result["false_rate_feasible"]
    assert not result["useful_operating_point"]


def test_incomplete_evidence_cannot_pass_a_gate():
    verdict = study_verdict([], None, "pilot")
    assert not verdict["confirmatory"]
    assert not verdict["mechanism_gate"]
    assert not verdict["contribution_gate"]
