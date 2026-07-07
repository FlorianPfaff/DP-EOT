"""State primitives for identity-preserving unresolved groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Label = str


@dataclass(frozen=True)
class ResolvedTarget:
    """Minimal labeled extended-target state used by the first benchmark."""

    label: Label
    mean: np.ndarray  # shape: (4,), columns x, y, vx, vy
    covariance: np.ndarray  # shape: (4, 4)
    extent: np.ndarray  # shape: (2, 2)
    measurement_rate: float

    def predict(self, dt: float, process_noise: float = 0.01) -> "ResolvedTarget":
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        q = process_noise * np.eye(4)
        return ResolvedTarget(
            label=self.label,
            mean=transition @ self.mean,
            covariance=transition @ self.covariance @ transition.T + q,
            extent=self.extent,
            measurement_rate=self.measurement_rate,
        )


@dataclass(frozen=True)
class UnresolvedGroup:
    """Aggregate group hypothesis with explicit member-label identity memory."""

    member_labels: frozenset[Label]
    mean: np.ndarray  # shape: (4,)
    covariance: np.ndarray  # shape: (4, 4)
    extent: np.ndarray  # shape: (2, 2)
    measurement_rate: float

    def contains(self, label: Label) -> bool:
        return label in self.member_labels


def merge_targets(targets: tuple[ResolvedTarget, ...]) -> UnresolvedGroup:
    """Create an unresolved-group state from resolved targets.

    This first scaffold uses moment matching. A later tracker can replace this
    with a full GGIW/PMBM-style hypothesis transition.
    """

    if len(targets) < 2:
        raise ValueError("at least two targets are required to form an unresolved group")

    member_labels = frozenset(target.label for target in targets)
    weights = np.asarray([target.measurement_rate for target in targets], dtype=float)
    if np.any(weights <= 0):
        weights = np.ones(len(targets), dtype=float)
    weights = weights / weights.sum()

    means = np.vstack([target.mean for target in targets])
    mean = weights @ means

    covariance = np.zeros_like(targets[0].covariance)
    for weight, target in zip(weights, targets):
        delta = target.mean - mean
        covariance += weight * (target.covariance + np.outer(delta, delta))

    extent = sum(target.extent for target in targets) / len(targets)
    measurement_rate = float(sum(target.measurement_rate for target in targets))

    return UnresolvedGroup(
        member_labels=member_labels,
        mean=mean,
        covariance=covariance,
        extent=extent,
        measurement_rate=measurement_rate,
    )


def split_group(
    group: UnresolvedGroup,
    prior_targets: tuple[ResolvedTarget, ...],
) -> tuple[ResolvedTarget, ...]:
    """Recover resolved target hypotheses from a group and member priors.

    The first version deliberately preserves labels by returning the member
    priors for labels contained in the group. Future implementations should add
    measurement-conditioned split likelihoods.
    """

    resolved = tuple(target for target in prior_targets if target.label in group.member_labels)
    if len(resolved) != len(group.member_labels):
        missing = group.member_labels.difference(target.label for target in prior_targets)
        raise ValueError(f"missing prior targets for group members: {sorted(missing)}")
    return resolved
