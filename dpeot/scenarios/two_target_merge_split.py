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
    measurement_rates: tuple[float, float] | None = None
    clutter_rate: float = 2.0
    clutter_extent: float = 12.0
    extent_axes: tuple[float, float] = (0.8, 0.25)
    extent_axes_b: tuple[float, float] | None = None
    measurement_noise_std: float = 0.08
    crossing_y_offset: float = 0.35
    seed: int = 7

    def validate(self) -> None:
        if self.num_steps <= 2:
            raise ValueError("num_steps must be greater than 2")
        if not (0 <= self.merge_start <= self.merge_end < self.num_steps):
            raise ValueError("merge_start and merge_end must define a valid interval")
        if self.measurement_rate <= 0:
            raise ValueError("measurement_rate must be positive")
        if self.measurement_rates is not None and any(rate <= 0 for rate in self.measurement_rates):
            raise ValueError("measurement_rates must be positive")
        if self.clutter_rate < 0:
            raise ValueError("clutter_rate cannot be negative")
        if self.measurement_noise_std < 0:
            raise ValueError("measurement_noise_std cannot be negative")
        if self.crossing_y_offset < 0:
            raise ValueError("crossing_y_offset cannot be negative")


@dataclass(frozen=True)
class TargetTrajectory:
    """Ground-truth trajectory for one labeled target."""

    label: Label
    states: np.ndarray  # shape: (num_steps, 4), columns x, y, vx, vy
    extent: np.ndarray  # shape: (2, 2)
    measurement_rate: float


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
    targets: tuple[TargetTrajectory, ...]
    scans: tuple[Scan, ...]

    @property
    def labels(self) -> tuple[Label, ...]:
        return tuple(target.label for target in self.targets)


def generate_two_target_merge_split(config: ScenarioConfig | None = None) -> Scenario:
    """Generate a two-target unresolved-group benchmark instance.

    The targets cross near the center of the surveillance region. During the
    configured unresolved interval, downstream trackers should prefer a group
    hypothesis carrying the member set {"A", "B"} instead of treating the merged
    measurement cloud as a newly born unlabeled object.
    """

    config = config or ScenarioConfig()
    config.validate()
    targets = _crossing_targets(config)
    return _scenario_from_targets(
        config,
        targets,
        unresolved_interval=(config.merge_start, config.merge_end),
    )


def generate_near_miss_no_merge(config: ScenarioConfig | None = None) -> Scenario:
    """Generate two targets that pass close to each other without merging."""

    config = config or ScenarioConfig()
    config.validate()
    crossing_step = 0.5 * (config.num_steps - 1)
    x_velocity = 8.0 / max(crossing_step, 1.0)
    separation = 2.8
    rates = config.measurement_rates or (config.measurement_rate, config.measurement_rate)
    targets = _two_targets_from_states(
        config,
        states_a=_constant_velocity_states(
            start=np.array([-8.0, -0.5 * separation]),
            velocity=np.array([x_velocity, 0.0]),
            num_steps=config.num_steps,
            dt=config.dt,
        ),
        states_b=_constant_velocity_states(
            start=np.array([8.0, 0.5 * separation]),
            velocity=np.array([-x_velocity, 0.0]),
            num_steps=config.num_steps,
            dt=config.dt,
        ),
        rates=rates,
    )
    return _scenario_from_targets(config, targets, unresolved_interval=None)


def generate_parallel_close_tracks(config: ScenarioConfig | None = None) -> Scenario:
    """Generate two close, parallel, continuously resolved target tracks."""

    config = config or ScenarioConfig()
    config.validate()
    x_velocity = 12.0 / max((config.num_steps - 1) * config.dt, config.dt)
    separation = 2.8
    rates = config.measurement_rates or (config.measurement_rate, config.measurement_rate)
    targets = _two_targets_from_states(
        config,
        states_a=_constant_velocity_states(
            start=np.array([-6.0, -0.5 * separation]),
            velocity=np.array([x_velocity, 0.0]),
            num_steps=config.num_steps,
            dt=config.dt,
        ),
        states_b=_constant_velocity_states(
            start=np.array([-6.0, 0.5 * separation]),
            velocity=np.array([x_velocity, 0.0]),
            num_steps=config.num_steps,
            dt=config.dt,
        ),
        rates=rates,
    )
    return _scenario_from_targets(config, targets, unresolved_interval=None)


def generate_single_large_extended_target(config: ScenarioConfig | None = None) -> Scenario:
    """Generate one large extended target that should not form a two-label group."""

    config = config or ScenarioConfig()
    config.validate()
    x_velocity = 12.0 / max((config.num_steps - 1) * config.dt, config.dt)
    states = _constant_velocity_states(
        start=np.array([-6.0, 0.0]),
        velocity=np.array([x_velocity, 0.0]),
        num_steps=config.num_steps,
        dt=config.dt,
    )
    extent = np.diag(np.square(np.asarray((1.7, 0.65), dtype=float)))
    target = TargetTrajectory(
        label="A",
        states=states,
        extent=extent,
        measurement_rate=2.0 * config.measurement_rate,
    )
    return _scenario_from_targets(config, (target,), unresolved_interval=None)


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


def _crossing_targets(config: ScenarioConfig) -> tuple[TargetTrajectory, TargetTrajectory]:
    crossing_step = 0.5 * (config.num_steps - 1)
    x_velocity = 8.0 / max(crossing_step, 1.0)
    y_velocity = config.crossing_y_offset / max(crossing_step * config.dt, config.dt)
    rates = config.measurement_rates or (config.measurement_rate, config.measurement_rate)
    return _two_targets_from_states(
        config,
        states_a=_constant_velocity_states(
            start=np.array([-8.0, -config.crossing_y_offset]),
            velocity=np.array([x_velocity, y_velocity]),
            num_steps=config.num_steps,
            dt=config.dt,
        ),
        states_b=_constant_velocity_states(
            start=np.array([8.0, config.crossing_y_offset]),
            velocity=np.array([-x_velocity, -y_velocity]),
            num_steps=config.num_steps,
            dt=config.dt,
        ),
        rates=rates,
    )


def _two_targets_from_states(
    config: ScenarioConfig,
    states_a: np.ndarray,
    states_b: np.ndarray,
    rates: tuple[float, float],
) -> tuple[TargetTrajectory, TargetTrajectory]:
    extent_a = np.diag(np.square(np.asarray(config.extent_axes, dtype=float)))
    extent_b_axes = config.extent_axes_b or config.extent_axes
    extent_b = np.diag(np.square(np.asarray(extent_b_axes, dtype=float)))
    return (
        TargetTrajectory(label="A", states=states_a, extent=extent_a, measurement_rate=rates[0]),
        TargetTrajectory(label="B", states=states_b, extent=extent_b, measurement_rate=rates[1]),
    )


def _scenario_from_targets(
    config: ScenarioConfig,
    targets: tuple[TargetTrajectory, ...],
    unresolved_interval: tuple[int, int] | None,
) -> Scenario:
    rng = np.random.default_rng(config.seed)
    scans = tuple(
        _sample_scan(
            k,
            targets,
            config,
            rng,
            unresolved_members=_unresolved_members_for_scan(k, targets, unresolved_interval),
        )
        for k in range(config.num_steps)
    )
    return Scenario(config=config, targets=targets, scans=scans)


def _unresolved_members_for_scan(
    k: int,
    targets: Iterable[TargetTrajectory],
    unresolved_interval: tuple[int, int] | None,
) -> frozenset[Label]:
    if unresolved_interval is None:
        return frozenset()
    start, end = unresolved_interval
    if start <= k <= end:
        return frozenset(target.label for target in targets)
    return frozenset()


def _sample_scan(
    k: int,
    targets: Iterable[TargetTrajectory],
    config: ScenarioConfig,
    rng: np.random.Generator,
    unresolved_members: frozenset[Label],
) -> Scan:
    measurements: list[np.ndarray] = []
    origins: list[Label | None] = []

    for target in targets:
        count = rng.poisson(target.measurement_rate)
        if count == 0:
            continue
        center = target.states[k, :2]
        measurement_cov = _measurement_covariance(target.extent, config.measurement_noise_std)
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


def _measurement_covariance(extent: np.ndarray, measurement_noise_std: float) -> np.ndarray:
    noise = np.square(measurement_noise_std) * np.eye(2)
    return extent + noise
