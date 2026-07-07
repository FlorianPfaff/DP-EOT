"""Synthetic two-target merge/split scenario.

This module intentionally keeps the first benchmark small. It generates two
extended targets that are resolvable before and after a controlled unresolved
interval. Measurements are still sampled from the individual physical targets,
but the scan metadata marks the interval in which the sensor-level explanation
should be treated as one unresolved group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Label = str


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for the two-target merge/split scenario."""

    num_steps: int = 41
    dt: float = 1.0
    merge_start: int = 17
    merge_end: int = 23
    measurement_rate: float = 12.0
    clutter_rate: float = 2.0
    clutter_extent: float = 12.0
    extent_axes: tuple[float, float] = (0.8, 0.25)
    measurement_noise_std: float = 0.08
    seed: int = 7

    def validate(self) -> None:
        if self.num_steps <= 2:
            raise ValueError("num_steps must be greater than 2")
        if not (0 <= self.merge_start <= self.merge_end < self.num_steps):
            raise ValueError("merge_start and merge_end must define a valid interval")
        if self.measurement_rate <= 0:
            raise ValueError("measurement_rate must be positive")
        if self.clutter_rate < 0:
            raise ValueError("clutter_rate cannot be negative")


@dataclass(frozen=True)
class TargetTrajectory:
    """Ground-truth trajectory for one labeled target."""

    label: Label
    states: np.ndarray  # shape: (num_steps, 4), columns x, y, vx, vy
    extent: np.ndarray  # shape: (2, 2)


@dataclass(frozen=True)
class Scan:
    """One scan of measurements and truth metadata."""

    k: int
    measurements: np.ndarray  # shape: (num_measurements, 2)
    origins: tuple[Label | None, ...]  # None denotes clutter
    unresolved_members: frozenset[Label]

    @property
    def is_unresolved(self) -> bool:
        return len(self.unresolved_members) > 1


@dataclass(frozen=True)
class Scenario:
    """Complete synthetic scenario."""

    config: ScenarioConfig
    targets: tuple[TargetTrajectory, TargetTrajectory]
    scans: tuple[Scan, ...]

    @property
    def labels(self) -> tuple[Label, Label]:
        return tuple(target.label for target in self.targets)  # type: ignore[return-value]


def generate_two_target_merge_split(config: ScenarioConfig | None = None) -> Scenario:
    """Generate a two-target unresolved-group benchmark instance.

    The targets cross near the center of the surveillance region. During the
    configured unresolved interval, downstream trackers should prefer a group
    hypothesis carrying the member set {"A", "B"} instead of treating the merged
    measurement cloud as a newly born unlabeled object.
    """

    config = config or ScenarioConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)

    states_a = _constant_velocity_states(
        start=np.array([-8.0, -0.35]),
        velocity=np.array([0.4, 0.0175]),
        num_steps=config.num_steps,
        dt=config.dt,
    )
    states_b = _constant_velocity_states(
        start=np.array([8.0, 0.35]),
        velocity=np.array([-0.4, -0.0175]),
        num_steps=config.num_steps,
        dt=config.dt,
    )

    extent = np.diag(np.square(np.asarray(config.extent_axes, dtype=float)))
    targets = (
        TargetTrajectory(label="A", states=states_a, extent=extent),
        TargetTrajectory(label="B", states=states_b, extent=extent),
    )

    scans = tuple(_sample_scan(k, targets, config, rng) for k in range(config.num_steps))
    return Scenario(config=config, targets=targets, scans=scans)


def oracle_cells(scan: Scan, group_during_unresolved: bool = True) -> list[list[int]]:
    """Return oracle measurement cells for debugging and upper-bound baselines.

    If ``group_during_unresolved`` is true, measurements from all unresolved
    members are returned as one cell during the merge interval. Otherwise,
    measurements are separated by their physical origin labels. Clutter is always
    returned as singleton cells.
    """

    if scan.measurements.size == 0:
        return []

    cells: list[list[int]] = []
    assigned: set[int] = set()

    if group_during_unresolved and scan.is_unresolved:
        group_cell = [
            i for i, origin in enumerate(scan.origins) if origin in scan.unresolved_members
        ]
        if group_cell:
            cells.append(group_cell)
            assigned.update(group_cell)
    else:
        for label in sorted({origin for origin in scan.origins if origin is not None}):
            cell = [i for i, origin in enumerate(scan.origins) if origin == label]
            if cell:
                cells.append(cell)
                assigned.update(cell)

    for i, origin in enumerate(scan.origins):
        if origin is None and i not in assigned:
            cells.append([i])

    return cells


def _constant_velocity_states(
    start: np.ndarray, velocity: np.ndarray, num_steps: int, dt: float
) -> np.ndarray:
    states = np.zeros((num_steps, 4), dtype=float)
    for k in range(num_steps):
        position = start + k * dt * velocity
        states[k, :2] = position
        states[k, 2:] = velocity
    return states


def _sample_scan(
    k: int,
    targets: Iterable[TargetTrajectory],
    config: ScenarioConfig,
    rng: np.random.Generator,
) -> Scan:
    unresolved_members = (
        frozenset(target.label for target in targets)
        if config.merge_start <= k <= config.merge_end
        else frozenset()
    )

    measurements: list[np.ndarray] = []
    origins: list[Label | None] = []

    measurement_cov = _measurement_covariance(config)
    for target in targets:
        count = rng.poisson(config.measurement_rate)
        if count == 0:
            continue
        center = target.states[k, :2]
        target_measurements = rng.multivariate_normal(center, measurement_cov, size=count)
        measurements.extend(target_measurements)
        origins.extend([target.label] * count)

    clutter_count = rng.poisson(config.clutter_rate)
    if clutter_count:
        clutter = rng.uniform(
            low=-config.clutter_extent,
            high=config.clutter_extent,
            size=(clutter_count, 2),
        )
        measurements.extend(clutter)
        origins.extend([None] * clutter_count)

    if not measurements:
        measurement_array = np.empty((0, 2), dtype=float)
    else:
        measurement_array = np.vstack(measurements)
        permutation = rng.permutation(measurement_array.shape[0])
        measurement_array = measurement_array[permutation]
        origins = [origins[i] for i in permutation]

    return Scan(
        k=k,
        measurements=measurement_array,
        origins=tuple(origins),
        unresolved_members=unresolved_members,
    )


def _measurement_covariance(config: ScenarioConfig) -> np.ndarray:
    extent = np.diag(np.square(np.asarray(config.extent_axes, dtype=float)))
    noise = np.square(config.measurement_noise_std) * np.eye(2)
    return extent + noise
