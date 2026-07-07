"""Distance-based measurement partitioning baseline.

This is intentionally simple and reviewer-friendly: measurements within a fixed
Euclidean distance threshold are connected into partition cells. It approximates
the class of hand-tuned gating/DBSCAN-style partition heuristics commonly used
before a full extended-object update.
"""

from __future__ import annotations

import numpy as np


def distance_partition(measurements: np.ndarray, threshold: float) -> list[list[int]]:
    """Partition measurements into connected components under a distance gate."""

    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if measurements.ndim != 2 or measurements.shape[1] != 2:
        raise ValueError("measurements must have shape (n, 2)")

    n = measurements.shape[0]
    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    threshold_sq = threshold * threshold
    for i in range(n):
        delta = measurements[i + 1 :] - measurements[i]
        distances_sq = np.einsum("ij,ij->i", delta, delta)
        for offset in np.flatnonzero(distances_sq <= threshold_sq):
            union(i, i + 1 + int(offset))

    cells_by_root: dict[int, list[int]] = {}
    for i in range(n):
        cells_by_root.setdefault(find(i), []).append(i)

    return list(cells_by_root.values())
