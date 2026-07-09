"""Mixture-of-finite-mixtures style partition proposal.

This module provides a small finite-cardinality alternative to the DP/CRP
partition proposal. It is intended as a reviewer-facing ablation: if performance
comes from flexible partition proposals rather than the DP prior specifically,
this baseline should reveal that.

The implementation samples a small set of K-means++ style initializations for a
range of component counts and returns the partition with the best penalized
within-cluster sum of squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MFMPartitionConfig:
    """Configuration for the MFM-style finite partition proposal."""

    min_components: int = 1
    max_components: int = 4
    num_initializations: int = 5
    num_iterations: int = 10
    penalty_per_component: float = 2.0
    seed: int = 17

    def validate(self) -> None:
        if self.min_components < 1:
            raise ValueError("min_components must be at least one")
        if self.max_components < self.min_components:
            raise ValueError("max_components must be no smaller than min_components")
        if self.num_initializations < 1:
            raise ValueError("num_initializations must be at least one")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be at least one")
        if self.penalty_per_component < 0:
            raise ValueError("penalty_per_component cannot be negative")


def mfm_partition(
    measurements: np.ndarray,
    config: MFMPartitionConfig | None = None,
) -> list[list[int]]:
    """Return a finite-cardinality partition proposal for one scan."""

    config = config or MFMPartitionConfig()
    config.validate()

    if measurements.ndim != 2 or measurements.shape[1] != 2:
        raise ValueError("measurements must have shape (n, 2)")

    n = measurements.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    rng = np.random.default_rng(config.seed)
    max_components = min(config.max_components, n)
    min_components = min(config.min_components, max_components)

    best_score = float("inf")
    best_assignments = np.zeros(n, dtype=int)

    for k in range(min_components, max_components + 1):
        for _ in range(config.num_initializations):
            assignments, score = _fit_kmeans(measurements, k, config.num_iterations, rng)
            penalized_score = score + config.penalty_per_component * k * np.log(n + 1.0)
            if penalized_score < best_score:
                best_score = penalized_score
                best_assignments = assignments

    return [
        np.flatnonzero(best_assignments == label).tolist()
        for label in sorted(set(best_assignments.tolist()))
    ]


def _fit_kmeans(
    measurements: np.ndarray,
    num_components: int,
    num_iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    centers = _initialize_centers(measurements, num_components, rng)
    assignments = np.zeros(measurements.shape[0], dtype=int)

    for _ in range(num_iterations):
        assignments = _assign_to_centers(measurements, centers)
        new_centers = centers.copy()
        for label in range(num_components):
            members = measurements[assignments == label]
            if len(members):
                new_centers[label] = members.mean(axis=0)
            else:
                new_centers[label] = measurements[rng.integers(0, measurements.shape[0])]
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    assignments = _assign_to_centers(measurements, centers)
    residuals = measurements - centers[assignments]
    score = float(np.einsum("ij,ij->", residuals, residuals))
    return assignments, score


def _initialize_centers(
    measurements: np.ndarray,
    num_components: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = [measurements[rng.integers(0, measurements.shape[0])]]
    while len(centers) < num_components:
        center_array = np.vstack(centers)
        distances_sq = np.min(
            np.sum((measurements[:, None, :] - center_array[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        total = float(distances_sq.sum())
        if total <= 0:
            centers.append(measurements[rng.integers(0, measurements.shape[0])])
            continue
        probabilities = distances_sq / total
        centers.append(measurements[rng.choice(measurements.shape[0], p=probabilities)])
    return np.vstack(centers)


def _assign_to_centers(measurements: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances_sq = np.sum((measurements[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances_sq, axis=1)
