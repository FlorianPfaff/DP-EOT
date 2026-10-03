"""Versioned finite-resolution study; truth never enters the tracker input."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np


STUDY_VERSION = "group-value-v1"
PHASES = {"pilot": 1, "calibrate": 2, "confirm": 3, "stress": 4, "runtime": 5}
ROOT_SEED = 20261003
Group = tuple[int, ...]
Grouping = tuple[Group, ...]


@dataclass(frozen=True)
class StudyCondition:
    name: str
    geometry: str = "crossing"
    resolution: float = 2.0
    acceleration_std: float = 0.02
    clutter_rate: float = 2.0
    asymmetric: bool = False
    maneuver: bool = False
    cloud: str = "gaussian"
    noise_multiplier: float = 1.0
    difficult: bool = False
    negative_control: bool = False


CONDITIONS = (
    StudyCondition("crossing_r1", resolution=1.0),
    StudyCondition("crossing_r2"),
    StudyCondition("crossing_r4", resolution=4.0, difficult=True),
    StudyCondition("acceleration", acceleration_std=0.08, difficult=True),
    StudyCondition("maneuver", maneuver=True, difficult=True),
    StudyCondition("asymmetric", asymmetric=True),
    StudyCondition("three_targets", geometry="three", difficult=True),
    StudyCondition("near_miss", geometry="near_miss", negative_control=True),
    StudyCondition("parallel", geometry="parallel", negative_control=True),
    StudyCondition("clutter", clutter_rate=10.0, difficult=True),
    StudyCondition("heavy_tails", cloud="student", difficult=True),
    StudyCondition("noise_mismatch", noise_multiplier=2.0, difficult=True),
)


@dataclass(frozen=True)
class SensorModel:
    extents: np.ndarray
    rates: np.ndarray
    resolution: float
    noise_std: float = 0.08
    clutter_rate: float = 2.0
    bounds: tuple[float, float] = (-30.0, 30.0)


@dataclass(frozen=True)
class ObservationScan:
    k: int
    measurements: np.ndarray


@dataclass(frozen=True)
class TrackerInput:
    labels: tuple[str, ...]
    scans: tuple[ObservationScan, ...]
    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    sensor: SensorModel
    acceleration_std: float
    dt: float = 1.0


@dataclass(frozen=True)
class StudyTruth:
    states: np.ndarray  # scan, target, (x, y, vx, vy)
    groupings: tuple[Grouping, ...]


@dataclass(frozen=True)
class StudyInstance:
    inputs: TrackerInput
    truth: StudyTruth
    condition: StudyCondition
    phase: str
    trial: int


def random_stream(phase: str, condition_index: int, trial: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([ROOT_SEED, PHASES[phase], condition_index, trial, stream])
    )


def motion_matrices(count: int, dt: float, acceleration_std: float) -> tuple[np.ndarray, np.ndarray]:
    single = np.eye(4)
    single[:2, 2:] = dt * np.eye(2)
    acceleration = np.vstack((0.5 * dt**2 * np.eye(2), dt * np.eye(2)))
    return np.kron(np.eye(count), single), np.kron(
        np.eye(count), acceleration_std**2 * acceleration @ acceleration.T
    )


def sensor_grouping(positions: np.ndarray, resolution: float) -> Grouping:
    components = [{i} for i in range(len(positions))]
    for i, j in combinations(range(len(positions)), 2):
        if np.linalg.norm(positions[i] - positions[j]) < resolution:
            left = next(c for c in components if i in c)
            right = next(c for c in components if j in c)
            if left is not right:
                left.update(right)
                components.remove(right)
    return tuple(sorted(tuple(sorted(c)) for c in components))


def group_moments(positions: np.ndarray, sensor: SensorModel, group: Group) -> tuple[np.ndarray, np.ndarray, float]:
    indices = list(group)
    rates = sensor.rates[indices]
    weights = rates / rates.sum()
    center = weights @ positions[indices]
    residual = positions[indices] - center
    shape = np.einsum("i,ijk->jk", weights, sensor.extents[indices])
    shape += np.einsum("i,ij,ik->jk", weights, residual, residual)
    shape += sensor.noise_std**2 * np.eye(2)
    return center, shape, float(rates.sum())


def sample_observations(
    states: np.ndarray, sensor: SensorModel, rng: np.random.Generator, cloud: str = "gaussian"
) -> tuple[tuple[ObservationScan, ...], tuple[Grouping, ...]]:
    scans, groupings = [], []
    for k, current in enumerate(states):
        grouping = sensor_grouping(current[:, :2], sensor.resolution)
        measurements = []
        for group in grouping:
            center, shape, rate = group_moments(current[:, :2], sensor, group)
            count = rng.poisson(rate)
            points = rng.multivariate_normal(np.zeros(2), shape, size=count)
            if cloud == "student":
                # Covariance-matched t_5 noise exposes non-Gaussian shape mismatch.
                points *= np.sqrt(3.0 / rng.chisquare(5, size=count))[:, None]
            elif cloud != "gaussian":
                raise ValueError(f"unknown cloud: {cloud}")
            measurements.extend(points + center)
        clutter = rng.uniform(*sensor.bounds, size=(rng.poisson(sensor.clutter_rate), 2))
        measurements.extend(clutter)
        points = np.asarray(measurements, dtype=float).reshape(-1, 2)
        points = points[rng.permutation(len(points))]
        scans.append(ObservationScan(k, points))
        groupings.append(grouping)
    return tuple(scans), tuple(groupings)


def generate_study_instance(
    condition: StudyCondition, phase: str, condition_index: int, trial: int, num_steps: int = 41
) -> StudyInstance:
    if phase not in PHASES or num_steps < 3:
        raise ValueError("invalid phase or trajectory length")
    initial = np.array([[-8.0, -0.35, 0.4, 0.0175], [8.0, 0.35, -0.4, -0.0175]])
    if condition.geometry == "near_miss":
        initial[:, 1] = [-3.0, 3.0]
        initial[:, 3] = 0.0
    elif condition.geometry == "parallel":
        initial = np.array([[-8.0, -3.0, 0.4, 0.0], [-8.0, 3.0, 0.4, 0.0]])
    elif condition.geometry == "three":
        initial = np.vstack((initial, [-7.0, 5.0, 0.3, 0.0]))
    elif condition.geometry == "single":
        initial = initial[:1]
    elif condition.geometry != "crossing":
        raise ValueError(f"unknown geometry: {condition.geometry}")
    count = len(initial)
    states = np.empty((num_steps, count, 4))
    states[0] = initial
    motion_rng = random_stream(phase, condition_index, trial, 1)
    for k in range(1, num_steps):
        acceleration = motion_rng.normal(0.0, condition.acceleration_std, size=(count, 2))
        if condition.maneuver and 18 <= k <= 22:
            acceleration[0] += [0.0, 0.12]
        states[k, :, :2] = states[k - 1, :, :2] + states[k - 1, :, 2:] + 0.5 * acceleration
        states[k, :, 2:] = states[k - 1, :, 2:] + acceleration
    extents = np.repeat(np.diag([0.8**2, 0.25**2])[None], count, axis=0)
    rates = np.full(count, 12.0)
    if condition.asymmetric:
        extents[1] = np.diag([1.15**2, 0.18**2])
        rates[:2] = [16.0, 8.0]
    declared_sensor = SensorModel(extents, rates, condition.resolution, clutter_rate=condition.clutter_rate)
    actual_sensor = replace(declared_sensor, noise_std=0.08 * condition.noise_multiplier)
    scans, groupings = sample_observations(
        states, actual_sensor, random_stream(phase, condition_index, trial, 2), condition.cloud
    )
    deviations = np.tile([0.5, 0.5, 0.1, 0.1], count)
    estimate = initial.ravel() + random_stream(phase, condition_index, trial, 0).normal(size=4 * count) * deviations
    inputs = TrackerInput(
        tuple(chr(65 + i) for i in range(count)), scans, estimate,
        np.diag(deviations**2), declared_sensor, condition.acceleration_std
    )
    return StudyInstance(inputs, StudyTruth(states, groupings), condition, phase, trial)
