"""Lightweight DP-style partition proposal.

This file is a deliberately small baseline/proposal mechanism, not the final
tracker. It implements one collapsed Gibbs-style sweep for a Gaussian CRP
mixture with fixed covariance. The resulting occupied clusters are useful as a
baseline against the identity-aware unresolved-group tracker.

Important modeling note: this proposal should not be interpreted as a physical
target-cardinality estimator. The paper framing treats it as a partition proposal
inside a physically calibrated extended-object likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DPPartitionConfig:
    """Configuration for a simple CRP/Gaussian partition proposal."""

    alpha: float = 1.0
    covariance_scale: float = 0.5
    num_sweeps: int = 5
    seed: int = 11

    def validate(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.covariance_scale <= 0:
            raise ValueError("covariance_scale must be positive")
        if self.num_sweeps < 1:
            raise ValueError("num_sweeps must be at least one")


def dp_partition(
    measurements: np.ndarray, config: DPPartitionConfig | None = None
) -> list[list[int]]:
    """Return a CRP/Gaussian partition proposal for a scan.

    The implementation is intentionally compact and deterministic for a fixed
    seed. It is meant for small synthetic experiments and ablations.
    """

    config = config or DPPartitionConfig()
    config.validate()

    if measurements.ndim != 2 or measurements.shape[1] != 2:
        raise ValueError("measurements must have shape (n, 2)")

    n = measurements.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    rng = np.random.default_rng(config.seed)
    assignments = np.arange(n, dtype=int)

    for _ in range(config.num_sweeps):
        for i in rng.permutation(n):
            assignments[i] = -1
            labels = _compact_labels(assignments)
            assignments = labels
            candidate_labels = sorted(set(assignments[assignments >= 0]))
            log_weights = []
            proposal_labels = []

            for label in candidate_labels:
                members = measurements[assignments == label]
                log_weights.append(
                    np.log(len(members))
                    + _log_predictive(measurements[i], members, config.covariance_scale)
                )
                proposal_labels.append(label)

            log_weights.append(
                np.log(config.alpha)
                + _log_predictive(measurements[i], np.empty((0, 2)), config.covariance_scale)
            )
            proposal_labels.append(max(candidate_labels, default=-1) + 1)

            probabilities = _softmax(np.asarray(log_weights, dtype=float))
            assignments[i] = rng.choice(proposal_labels, p=probabilities)

    assignments = _compact_labels(assignments)
    return [np.flatnonzero(assignments == label).tolist() for label in sorted(set(assignments))]


def _compact_labels(assignments: np.ndarray) -> np.ndarray:
    labels = [label for label in sorted(set(assignments.tolist())) if label >= 0]
    mapping = {label: new_label for new_label, label in enumerate(labels)}
    compact = np.full_like(assignments, fill_value=-1)
    for i, label in enumerate(assignments):
        if label >= 0:
            compact[i] = mapping[int(label)]
    return compact


def _log_predictive(point: np.ndarray, members: np.ndarray, covariance_scale: float) -> float:
    if members.size == 0:
        mean = np.zeros(2, dtype=float)
        variance = 25.0 + covariance_scale
    else:
        mean = members.mean(axis=0)
        variance = covariance_scale * (1.0 + 1.0 / max(len(members), 1))

    residual = point - mean
    return -0.5 * (2.0 * np.log(2.0 * np.pi * variance) + residual @ residual / variance)


def _softmax(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / weights.sum()
