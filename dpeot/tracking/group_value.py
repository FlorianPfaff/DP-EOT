"""Small joint-state filters for falsifying the unresolved-group value claim.

All methods share observations and candidate cell allocations. The group and
labeled-joint variants deliberately implement equivalent observation operators:
changing the representation alone must not manufacture a performance advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log, pi

import numpy as np

from dpeot.partitions.distance_partition import distance_partition
from dpeot.scenarios.group_value import Group, Grouping, TrackerInput, group_moments, motion_matrices


METHODS = ("prediction_only", "labeled_coast", "labeled_hypothesis", "coupled_group", "labeled_joint", "oracle_mode")
DISTANCE_THRESHOLDS = (0.5, 0.75, 1.0, 1.25, 1.5)


@dataclass(frozen=True)
class StudyFilterConfig:
    threshold: float = 0.0
    max_hypotheses: int = 8
    cells_per_source: int = 3


@dataclass(frozen=True)
class JointHypothesis:
    mean: np.ndarray
    covariance: np.ndarray
    log_weight: float
    grouping: Grouping


@dataclass(frozen=True)
class StudyTrace:
    means: np.ndarray
    covariances: np.ndarray
    groupings: tuple[Grouping, ...]
    hypothesis_weights: tuple[np.ndarray, ...]
    hypothesis_means: tuple[np.ndarray, ...]
    hypothesis_covariances: tuple[np.ndarray, ...]
    detector_scores: np.ndarray
    retained_mass: np.ndarray
    proposal_failures: np.ndarray


@dataclass(frozen=True)
class Cell:
    indices: tuple[int, ...]
    center: np.ndarray
    scatter: np.ndarray


@dataclass(frozen=True)
class SourceCell:
    group: Group
    cell: Cell
    operator: np.ndarray
    noise: np.ndarray
    log_factor: float


def target_partitions(count: int) -> tuple[Grouping, ...]:
    if count == 0:
        return ((),)
    result = []
    for partition in target_partitions(count - 1):
        result.append(partition + ((count - 1,),))
        for index in range(len(partition)):
            result.append(partition[:index] + (partition[index] + (count - 1,),) + partition[index + 1:])
    return tuple(sorted(result))


def has_group(grouping: Grouping) -> bool:
    return any(len(group) > 1 for group in grouping)


def logsumexp(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    maximum = float(values.max())
    if not np.isfinite(maximum):
        return maximum
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def gaussian_logpdf(residual: np.ndarray, covariance: np.ndarray) -> float:
    sign, determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("nonpositive Gaussian covariance")
    return float(-0.5 * (len(residual) * log(2 * pi) + determinant + residual @ np.linalg.solve(covariance, residual)))


def observation_operator(group: Group, inputs: TrackerInput, representation: str = "group") -> np.ndarray:
    operator = np.zeros((2, len(inputs.initial_mean)))
    weights = inputs.sensor.rates[list(group)]
    weights = weights / weights.sum()
    if representation == "group":
        for index, weight in zip(group, weights):
            operator[:, 4 * index:4 * index + 2] = weight * np.eye(2)
    else:
        # The labeled representation builds the same linear map without a group state.
        selectors = np.eye(len(inputs.initial_mean)).reshape(len(inputs.labels), 4, -1)
        operator = np.einsum("i,ijk->jk", weights, selectors[list(group), :2])
    return operator


def joint_update(mean: np.ndarray, covariance: np.ndarray, observation: np.ndarray, operator: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    innovation = operator @ covariance @ operator.T + noise
    gain = np.linalg.solve(innovation, operator @ covariance).T
    updated_mean = mean + gain @ (observation - operator @ mean)
    remainder = np.eye(len(mean)) - gain @ operator
    updated_covariance = remainder @ covariance @ remainder.T + gain @ noise @ gain.T
    return updated_mean, 0.5 * (updated_covariance + updated_covariance.T)


def candidate_cells(measurements: np.ndarray) -> tuple[Cell, ...]:
    indices = set()
    for threshold in DISTANCE_THRESHOLDS:
        indices.update(tuple(sorted(cell)) for cell in distance_partition(measurements, threshold))
    result = []
    for selected in sorted(indices):
        points = measurements[list(selected)]
        center = points.mean(axis=0)
        residual = points - center
        result.append(Cell(selected, center, residual.T @ residual))
    return tuple(result)


def _source_options(
    inputs: TrackerInput, hypothesis: JointHypothesis, group: Group,
    cells: tuple[Cell, ...], config: StudyFilterConfig, representation: str,
) -> tuple[SourceCell | None, ...]:
    positions = hypothesis.mean.reshape(-1, 4)[:, :2]
    _, shape, rate = group_moments(positions, inputs.sensor, group)
    operator = observation_operator(group, inputs, representation)
    _, logdet = np.linalg.slogdet(shape)
    inverse = np.linalg.inv(shape)
    options = []
    for cell in cells:
        n = len(cell.indices)
        noise = shape / n
        residual = cell.center - operator @ hypothesis.mean
        predictive = operator @ hypothesis.covariance @ operator.T + noise
        if residual @ np.linalg.solve(predictive, residual) > 36.0:
            continue
        # Product Gaussian point likelihood factored into scatter and centroid.
        # This uses the Poisson *set* density; no per-cell n! belongs here.
        factor = n * log(rate) - 0.5 * (
            (n - 1) * (2 * log(2 * pi) + logdet)
            + float(np.trace(inverse @ cell.scatter)) + 2 * log(n)
        )
        option = SourceCell(group, cell, operator, noise, factor)
        clutter_density = max(inputs.sensor.clutter_rate / (inputs.sensor.bounds[1] - inputs.sensor.bounds[0])**2, 1e-300)
        rank = factor + gaussian_logpdf(residual, predictive) - n * log(clutter_density)
        options.append((rank, option))
    options.sort(key=lambda item: (-item[0], item[1].cell.indices))
    return (None,) + tuple(option for _, option in options[:config.cells_per_source])


def _block_diagonal(matrices: list[np.ndarray]) -> np.ndarray:
    result = np.zeros((2 * len(matrices), 2 * len(matrices)))
    for i, matrix in enumerate(matrices):
        result[2 * i:2 * i + 2, 2 * i:2 * i + 2] = matrix
    return result


def allocation_log_likelihood(
    inputs: TrackerInput, hypothesis: JointHypothesis, selected: tuple[SourceCell, ...],
    measurements: np.ndarray,
) -> float:
    assigned_indices = {index for option in selected for index in option.cell.indices}
    num_clutter = len(measurements) - len(assigned_indices)
    low, high = inputs.sensor.bounds
    outside = set(np.flatnonzero(np.any((measurements < low) | (measurements > high), axis=1)))
    if outside - assigned_indices:
        return -np.inf
    density = inputs.sensor.clutter_rate / (inputs.sensor.bounds[1] - inputs.sensor.bounds[0])**2
    if num_clutter and density == 0:
        return -np.inf
    score = -float(inputs.sensor.rates.sum()) - inputs.sensor.clutter_rate
    if num_clutter:
        score += num_clutter * log(density)
    if not selected:
        return score
    operator = np.vstack([option.operator for option in selected])
    observation = np.concatenate([option.cell.center for option in selected])
    noise = _block_diagonal([option.noise for option in selected])
    return score + sum(option.log_factor for option in selected) + gaussian_logpdf(
        observation - operator @ hypothesis.mean, operator @ hypothesis.covariance @ operator.T + noise
    )


def _candidates(
    inputs: TrackerInput, hypothesis: JointHypothesis, measurements: np.ndarray,
    cells: tuple[Cell, ...], config: StudyFilterConfig, method: str, oracle: Grouping | None,
) -> tuple[list[JointHypothesis], float, bool]:
    representation = "labeled" if method == "labeled_joint" else "group"
    options_by_group = {}
    allocations = []
    partitions = (oracle,) if oracle is not None else target_partitions(len(inputs.labels))
    for grouping in partitions:
        options = []
        for group in grouping:
            if group not in options_by_group:
                options_by_group[group] = _source_options(inputs, hypothesis, group, cells, config, representation)
            options.append(options_by_group[group])
        for allocation in product(*options):
            selected = tuple(option for option in allocation if option is not None)
            indices = [i for option in selected for i in option.cell.indices]
            if len(set(indices)) != len(indices):
                continue
            score = allocation_log_likelihood(inputs, hypothesis, selected, measurements)
            if np.isfinite(score):
                allocations.append((score, grouping, selected))
    best_group = max((score for score, grouping, _ in allocations if has_group(grouping)), default=-np.inf)
    best_resolved = max((score for score, grouping, _ in allocations if not has_group(grouping)), default=-np.inf)
    delta = best_group - best_resolved if allocations else float("nan")
    use_group = best_group > best_resolved + config.threshold
    retained = [entry for entry in allocations if oracle is not None or has_group(entry[1]) == use_group]
    retained.sort(key=lambda entry: -entry[0])
    candidates = []
    # Keep all child weights for a pre-pruning, candidate-conditional diagnostic.
    for score, grouping, selected in retained:
        updating = tuple(option for option in selected if len(option.group) == 1 or method in {"coupled_group", "labeled_joint", "oracle_mode"})
        if updating:
            mean, covariance = joint_update(
                hypothesis.mean, hypothesis.covariance,
                np.concatenate([option.cell.center for option in updating]),
                np.vstack([option.operator for option in updating]),
                _block_diagonal([option.noise for option in updating]),
            )
        else:
            mean, covariance = hypothesis.mean.copy(), hypothesis.covariance.copy()
        candidates.append(JointHypothesis(mean, covariance, hypothesis.log_weight + score, grouping))
    proposal_failed = not candidates
    if proposal_failed:
        # A proposal bank can miss every valid allocation. Expose this as zero
        # retained mass upstream and coast; never consult truth to repair it.
        candidates = [JointHypothesis(hypothesis.mean, hypothesis.covariance, hypothesis.log_weight - 1e6, tuple((i,) for i in range(len(inputs.labels))))]
    return candidates, delta, proposal_failed


def prune_hypotheses(candidates: list[JointHypothesis], limit: int) -> tuple[list[JointHypothesis], float]:
    combined: dict[tuple, JointHypothesis] = {}
    for hypothesis in candidates:
        key = (hypothesis.grouping, hypothesis.mean.round(11).tobytes(), hypothesis.covariance.round(11).tobytes())
        if key in combined:
            previous = combined[key]
            combined[key] = JointHypothesis(previous.mean, previous.covariance, float(np.logaddexp(previous.log_weight, hypothesis.log_weight)), previous.grouping)
        else:
            combined[key] = hypothesis
    ordered = sorted(combined.values(), key=lambda h: -h.log_weight)
    total = logsumexp([h.log_weight for h in ordered])
    retained = ordered[:limit]
    normalization = logsumexp([h.log_weight for h in retained])
    return [JointHypothesis(h.mean, h.covariance, h.log_weight - normalization, h.grouping) for h in retained], float(np.exp(normalization - total))


def run_study_filter(
    inputs: TrackerInput, method: str, config: StudyFilterConfig | None = None,
    *, oracle_groupings: tuple[Grouping, ...] | None = None,
) -> StudyTrace:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if (method == "oracle_mode") != (oracle_groupings is not None):
        raise ValueError("only the explicit oracle diagnostic accepts truth groupings")
    if oracle_groupings is not None and len(oracle_groupings) != len(inputs.scans):
        raise ValueError("oracle advice length mismatch")
    config = config or StudyFilterConfig()
    if config.max_hypotheses < 1 or config.cells_per_source < 1:
        raise ValueError("hypothesis and cell budgets must be positive")
    count = len(inputs.labels)
    if not 1 <= count <= 3:
        raise ValueError("study supports one to three established tracks")
    resolved = tuple((i,) for i in range(count))
    hypotheses = [JointHypothesis(inputs.initial_mean.copy(), inputs.initial_covariance.copy(), 0.0, resolved)]
    transition, process = motion_matrices(count, inputs.dt, inputs.acceleration_std)
    means, covariances, groupings, weights, hypothesis_means, hypothesis_covariances, scores, masses, failures = [], [], [], [], [], [], [], [], []
    for scan_index, scan in enumerate(inputs.scans):
        if scan_index:
            hypotheses = [JointHypothesis(transition @ h.mean, transition @ h.covariance @ transition.T + process, h.log_weight, h.grouping) for h in hypotheses]
        delta, mass = float("nan"), 1.0
        failed = False
        if method != "prediction_only":
            points = scan.measurements[np.lexsort((scan.measurements[:, 1], scan.measurements[:, 0]))]
            cells = candidate_cells(points)
            candidates, deltas = [], []
            for hypothesis in hypotheses:
                children, delta, proposal_failed = _candidates(
                    inputs, hypothesis, points, cells, config, method,
                    oracle_groupings[scan_index] if oracle_groupings is not None else None,
                )
                candidates.extend(children)
                deltas.append(delta)
                failed = failed or proposal_failed
            delta = deltas[0]
            hypotheses, mass = prune_hypotheses(candidates, 1 if method == "labeled_coast" else config.max_hypotheses)
            if failed:
                mass = 0.0
        best = hypotheses[0]
        means.append(best.mean)
        covariances.append(best.covariance)
        groupings.append(best.grouping)
        weights.append(np.exp([h.log_weight for h in hypotheses]))
        hypothesis_means.append(np.asarray([h.mean for h in hypotheses]))
        hypothesis_covariances.append(np.asarray([h.covariance for h in hypotheses]))
        scores.append(delta)
        masses.append(mass)
        failures.append(failed)
    return StudyTrace(np.asarray(means), np.asarray(covariances), tuple(groupings), tuple(weights), tuple(hypothesis_means), tuple(hypothesis_covariances), np.asarray(scores), np.asarray(masses), np.asarray(failures))
