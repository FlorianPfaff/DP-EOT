"""Physical scoring utilities for measurement partitions.

The functions in this module deliberately separate physical measurement
likelihoods from nonparametric partition proposals. A DP/CRP mechanism may
suggest candidate cells, but the count likelihood is still evaluated with an
explicit Gamma-Poisson rate model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dpeot.models.rate import GammaPoissonRate


def log_cell_likelihood(
    measurements: np.ndarray,
    cell: Sequence[int],
    predicted_position: np.ndarray,
    expected_rate: float,
    covariance: np.ndarray,
    rate_strength: float = 2.0,
) -> float:
    """Score one measurement cell against a predicted extended target.

    The score is the sum of a Gamma-Poisson predictive count term and an
    elliptical Gaussian spatial likelihood. It is intentionally simple; the
    first benchmark needs a physically interpretable score before introducing a
    full GGIW update.
    """

    if measurements.ndim != 2 or measurements.shape[1] != 2:
        raise ValueError("measurements must have shape (n, 2)")
    if predicted_position.shape != (2,):
        raise ValueError("predicted_position must have shape (2,)")
    if covariance.shape != (2, 2):
        raise ValueError("covariance must have shape (2, 2)")
    if expected_rate <= 0:
        raise ValueError("expected_rate must be positive")
    if rate_strength <= 0:
        raise ValueError("rate_strength must be positive")

    indices = np.asarray(cell, dtype=int)
    if indices.size == 0:
        count = 0
        points = np.empty((0, 2), dtype=float)
    else:
        count = int(indices.size)
        points = measurements[indices]

    rate_model = GammaPoissonRate(
        shape=rate_strength,
        rate=rate_strength / expected_rate,
    )
    return rate_model.log_predictive_pmf(count) + log_gaussian_points(
        points, predicted_position, covariance
    )


def log_gaussian_points(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    """Return summed log Gaussian likelihood for 2-D points."""

    if points.size == 0:
        return 0.0
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")

    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")

    residuals = points - mean
    solved = np.linalg.solve(covariance, residuals.T).T
    mahalanobis = np.einsum("ij,ij->i", residuals, solved)
    dim = 2
    return float(-0.5 * np.sum(dim * np.log(2.0 * np.pi) + logdet + mahalanobis))


def cell_centroid(measurements: np.ndarray, cell: Sequence[int]) -> np.ndarray:
    """Return the centroid of a nonempty measurement cell."""

    indices = np.asarray(cell, dtype=int)
    if indices.size == 0:
        raise ValueError("cannot compute centroid of an empty cell")
    return measurements[indices].mean(axis=0)
