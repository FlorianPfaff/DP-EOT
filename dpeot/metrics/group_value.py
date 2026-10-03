"""Trial-level metrics and paired uncertainty for the corrected study."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from dpeot.scenarios.group_value import StudyInstance
from dpeot.tracking.group_value import StudyTrace, gaussian_logpdf, has_group, logsumexp


def _groups(grouping: tuple) -> set[tuple[int, ...]]:
    return {group for group in grouping if len(group) > 1}


def merge_events(groupings: tuple) -> list[tuple[int, int, tuple[int, ...]]]:
    result = []
    active = {}
    for k in range(len(groupings) + 1):
        present = _groups(groupings[k]) if k < len(groupings) else set()
        for group in set(active) - present:
            result.append((active.pop(group), k - 1, group))
        for group in present - set(active):
            active[group] = k
    return sorted(result)


def summarize_trace(instance: StudyInstance, trace: StudyTrace) -> dict[str, float | int | None]:
    truth = instance.truth
    steps, count, _ = truth.states.shape
    estimates = trace.means.reshape(steps, count, 4)
    position_error = estimates[:, :, :2] - truth.states[:, :, :2]
    assignments = []
    for k in range(steps):
        assignments.append(min(permutations(range(count)), key=lambda perm: float(np.sum(
            (estimates[k, list(perm), :2] - truth.states[k, :, :2])**2
        ))))
    assignments = np.asarray(assignments)
    correct = assignments == np.arange(count)[None]
    events = merge_events(truth.groupings)
    post_mask = np.zeros((steps, count), dtype=bool)
    delays, onset, release, censored = [], [], [], 0
    for start, end, group in events:
        next_group = next((k for k in range(end + 1, steps) if any(
            set(group).intersection(other) for other in _groups(truth.groupings[k])
        )), steps)
        post_mask[end + 1:next_group, list(group)] = True
        detection = next((k for k in range(start, end + 1) if group in _groups(trace.groupings[k])), None)
        onset.append(end - start + 1 if detection is None else detection - start)
        recovery = next((k for k in range(end + 1, next_group - 1)
                         if correct[k:k + 2, list(group)].all()), None)
        if recovery is not None:
            delays.append(recovery - end - 1)
        elif next_group > end + 1:
            censored += 1
        if end + 1 < steps:
            release.append(next((k - end - 1 for k in range(end + 1, steps)
                                 if group not in _groups(trace.groupings[k])), steps - end - 1))
    tp = fp = fn = false_scans = missed_scans = wrong_scans = resolved_scans = 0
    for predicted, actual in zip(trace.groupings, truth.groupings):
        p, t = _groups(predicted), _groups(actual)
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
        resolved_scans += int(not t)
        false_scans += int(bool(p) and not t)
        missed_scans += int(bool(t) and not p)
        wrong_scans += int(bool(p) and bool(t) and p != t)
    switches = sum(int(np.sum((assignments[k] != assignments[k - 1]) & post_mask[k] & post_mask[k - 1])) for k in range(1, steps))
    covered, negative_log_density = [], []
    position_indices = np.array([4 * i + j for i in range(count) for j in (0, 1)])
    for k in range(steps):
        weights = trace.hypothesis_weights[k]
        means = trace.hypothesis_means[k][:, position_indices]
        covariances = trace.hypothesis_covariances[k][:, position_indices][:, :, position_indices]
        true_position = truth.states[k, :, :2].ravel()
        mixture_mean = weights @ means
        difference = means - mixture_mean
        mixture_covariance = np.einsum("i,ijk->jk", weights, covariances) + np.einsum("i,ij,ik->jk", weights, difference, difference)
        for target in range(count):
            indices = slice(2 * target, 2 * target + 2)
            residual = true_position[indices] - mixture_mean[indices]
            mahalanobis = residual @ np.linalg.solve(mixture_covariance[indices, indices], residual)
            covered.append(mahalanobis <= 5.991464547107979)
        negative_log_density.append(-logsumexp([
            float(np.log(weight)) + gaussian_logpdf(true_position - mean, covariance)
            for weight, mean, covariance in zip(weights, means, covariances) if weight > 0
        ]))
    actual_duration = [end - start + 1 for start, end, _ in events]
    return {
        "labeled_position_rmse": float(np.sqrt(np.mean(np.sum(position_error**2, axis=2)))),
        "recovery": float(correct[post_mask].mean()) if post_mask.any() else None,
        "post_split_identity_switches": switches if post_mask.any() else None,
        "post_split_target_scans": int(post_mask.sum()),
        "split_recovery_delay": float(np.mean(delays)) if delays else None,
        "censored_recovery_events": censored,
        "merge_onset_delay": float(np.mean(onset)) if onset else None,
        "split_release_delay": float(np.mean(release)) if release else None,
        "group_precision": tp / (tp + fp) if tp + fp else None,
        "group_recall": tp / (tp + fn) if tp + fn else None,
        "group_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "group_tp": tp, "group_fp": fp, "group_fn": fn,
        "false_group_scans": false_scans, "resolved_scans": resolved_scans,
        "false_group_scan_rate": false_scans / resolved_scans if resolved_scans else None,
        "missed_group_scans": missed_scans, "wrong_membership_scans": wrong_scans,
        "merge_events": len(events),
        "actual_merge_duration": float(np.mean(actual_duration)) if actual_duration else None,
        "coverage_95": float(np.mean(covered)),
        "position_nll": float(np.mean(negative_log_density)),
        "retained_mass_mean": float(np.mean(trace.retained_mass)),
        "retained_mass_min": float(np.min(trace.retained_mass)),
        "proposal_failure_scans": int(trace.proposal_failures.sum()),
        "detected_group_scans": sum(has_group(g) for g in trace.groupings),
    }


def paired_interval(
    candidate: list[float], baseline: list[float], *, relative: bool = False,
    seed: int = 0, repetitions: int = 5000,
) -> dict[str, float | int]:
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired interval requires nonempty equally sized trial vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(repetitions, len(a)))
    if relative:
        estimate = 1.0 - a.mean() / b.mean()
        distribution = 1.0 - a[indices].mean(axis=1) / b[indices].mean(axis=1)
    else:
        estimate = (a - b).mean()
        distribution = (a[indices] - b[indices]).mean(axis=1)
    lo, hi = np.quantile(distribution, [0.025, 0.975])
    return {"estimate": float(estimate), "low": float(lo), "high": float(hi), "pairs": len(a)}
