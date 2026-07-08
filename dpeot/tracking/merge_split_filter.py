"""Minimal filters for the two-target unresolved-group benchmark.

These filters are deliberately small. Their purpose is to generate the first
falsifiable result for the DP-EOT paper: explicit unresolved-group member labels
should reduce post-split identity switches relative to clustering-only baselines.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import permutations, product
from math import inf

import numpy as np

from dpeot.scenarios.two_target_merge_split import Scan, Scenario
from dpeot.tracking.partition_scoring import cell_centroid, log_cell_likelihood
from dpeot.tracking.unresolved_group import ResolvedTarget, UnresolvedGroup, merge_targets


Partitioner = Callable[[Scan], list[list[int]]]
Label = str


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for the minimal merge/split benchmark filters."""

    update_gain: float = 0.65
    group_update_gain: float = 0.55
    process_noise: float = 0.02
    measurement_noise_scale: float = 0.08
    missed_cell_log_score: float = -35.0
    collapse_velocity_damping: float = 0.0


@dataclass(frozen=True)
class FilterRunResult:
    """Output trace used by metrics and experiments."""

    method_name: str
    assignments: list[dict[Label, Label]]
    estimated_positions: list[dict[Label, np.ndarray]]
    group_membership_trace: list[frozenset[Label]]


def run_identity_aware_group_filter(
    scenario: Scenario,
    partitioner: Partitioner,
    config: FilterConfig | None = None,
) -> FilterRunResult:
    """Run the proposed minimal identity-aware unresolved-group filter.

    During unresolved scans, the aggregate group is updated from the merged cell,
    but member targets keep their labels and continue to propagate as latent
    subtracks. At split, outgoing cells are scored against the predicted member
    states, which preserves identity through the crossing in the initial
    benchmark.
    """

    config = config or FilterConfig()
    targets = _initial_targets(scenario)
    estimated_positions: list[dict[Label, np.ndarray]] = []
    group_members: list[frozenset[Label]] = []
    group: UnresolvedGroup | None = None

    for scan in scenario.scans:
        if scan.k > 0:
            targets = tuple(
                target.predict(scenario.config.dt, process_noise=config.process_noise)
                for target in targets
            )

        cells = partitioner(scan)

        if scan.is_unresolved:
            group = merge_targets(targets) if group is None else merge_targets(targets)
            group_cell = _best_group_cell(scan.measurements, cells, group, config)
            if group_cell is not None:
                group = _update_group_with_cell(
                    group, scan.measurements, group_cell, config.group_update_gain
                )
            estimated_positions.append(_positions_from_targets(targets))
            group_members.append(group.member_labels)
            continue

        group = None
        assigned_cells = _assign_cells_to_targets(scan.measurements, cells, targets, config)
        targets = tuple(
            _update_target_with_cell(
                target,
                scan.measurements,
                assigned_cells.get(target.label, []),
                config.update_gain,
            )
            for target in targets
        )
        estimated_positions.append(_positions_from_targets(targets))
        group_members.append(frozenset())

    assignments = truth_to_estimated_label_assignments(scenario, estimated_positions)
    return FilterRunResult(
        method_name="proposed_group_labels",
        assignments=assignments,
        estimated_positions=estimated_positions,
        group_membership_trace=group_members,
    )


def run_distance_collapse_baseline(
    scenario: Scenario,
    partitioner: Partitioner,
    config: FilterConfig | None = None,
) -> FilterRunResult:
    """Run a baseline that collapses identities during the unresolved interval.

    The baseline updates both labels to the merged centroid and damps velocity.
    After separation it assigns labels by canonical left-to-right order, which
    mimics a common failure mode of trackers that localize the merged object but
    do not retain member-label memory.
    """

    config = config or FilterConfig()
    targets = _initial_targets(scenario)
    estimated_positions: list[dict[Label, np.ndarray]] = []
    group_members: list[frozenset[Label]] = []
    has_collapsed = False

    for scan in scenario.scans:
        if scan.k > 0:
            targets = tuple(
                target.predict(scenario.config.dt, process_noise=config.process_noise)
                for target in targets
            )

        cells = partitioner(scan)

        if scan.is_unresolved:
            group = merge_targets(targets)
            group_cell = _best_group_cell(scan.measurements, cells, group, config)
            if group_cell is not None:
                centroid = cell_centroid(scan.measurements, group_cell)
                targets = tuple(
                    _collapse_target_to_centroid(target, centroid, config)
                    for target in targets
                )
            has_collapsed = True
            estimated_positions.append(_positions_from_targets(targets))
            group_members.append(frozenset())
            continue

        if has_collapsed:
            targets = _targets_from_x_order_cells(targets, scan.measurements, cells)
        else:
            assigned_cells = _assign_cells_to_targets(scan.measurements, cells, targets, config)
            targets = tuple(
                _update_target_with_cell(
                    target,
                    scan.measurements,
                    assigned_cells.get(target.label, []),
                    config.update_gain,
                )
                for target in targets
            )

        estimated_positions.append(_positions_from_targets(targets))
        group_members.append(frozenset())

    assignments = truth_to_estimated_label_assignments(scenario, estimated_positions)
    return FilterRunResult(
        method_name="distance_collapse",
        assignments=assignments,
        estimated_positions=estimated_positions,
        group_membership_trace=group_members,
    )


def run_x_order_clustering_baseline(
    scenario: Scenario,
    partitioner: Partitioner,
    config: FilterConfig | None = None,
) -> FilterRunResult:
    """Run an unlabeled clustering baseline with canonical left-to-right labels."""

    config = config or FilterConfig()
    targets = _initial_targets(scenario)
    estimated_positions: list[dict[Label, np.ndarray]] = []
    group_members: list[frozenset[Label]] = []

    for scan in scenario.scans:
        if scan.k > 0:
            targets = tuple(
                target.predict(scenario.config.dt, process_noise=config.process_noise)
                for target in targets
            )
        cells = partitioner(scan)
        targets = _targets_from_x_order_cells(targets, scan.measurements, cells)
        estimated_positions.append(_positions_from_targets(targets))
        group_members.append(frozenset())

    assignments = truth_to_estimated_label_assignments(scenario, estimated_positions)
    return FilterRunResult(
        method_name="x_order_clustering",
        assignments=assignments,
        estimated_positions=estimated_positions,
        group_membership_trace=group_members,
    )


def run_oracle_identity_baseline(scenario: Scenario) -> FilterRunResult:
    """Return a perfect-identity upper bound based on the scenario truth."""

    estimated_positions: list[dict[Label, np.ndarray]] = []
    group_members: list[frozenset[Label]] = []
    for scan in scenario.scans:
        estimated_positions.append(
            {
                target.label: target.states[scan.k, :2].copy()
                for target in scenario.targets
            }
        )
        group_members.append(scan.unresolved_members)

    assignments = truth_to_estimated_label_assignments(scenario, estimated_positions)
    return FilterRunResult(
        method_name="oracle_identity",
        assignments=assignments,
        estimated_positions=estimated_positions,
        group_membership_trace=group_members,
    )


def truth_to_estimated_label_assignments(
    scenario: Scenario,
    estimated_positions: Sequence[dict[Label, np.ndarray]],
) -> list[dict[Label, Label]]:
    """Assign each true label to the nearest estimated label at every scan."""

    true_labels = scenario.labels
    assignments: list[dict[Label, Label]] = []

    for k, estimates in enumerate(estimated_positions):
        estimate_items = list(estimates.items())
        if not estimate_items:
            assignments.append({})
            continue

        if len(estimate_items) < len(true_labels):
            scan_assignment: dict[Label, Label] = {}
            for true_label in true_labels:
                true_position = _truth_position(scenario, true_label, k)
                estimated_label, _ = min(
                    estimate_items,
                    key=lambda item: float(np.linalg.norm(item[1] - true_position)),
                )
                scan_assignment[true_label] = estimated_label
            assignments.append(scan_assignment)
            continue

        best_cost = inf
        best_assignment: dict[Label, Label] = {}
        for permuted_items in permutations(estimate_items, len(true_labels)):
            cost = 0.0
            candidate: dict[Label, Label] = {}
            for true_label, (estimated_label, estimated_position) in zip(true_labels, permuted_items):
                true_position = _truth_position(scenario, true_label, k)
                cost += float(np.linalg.norm(estimated_position - true_position))
                candidate[true_label] = estimated_label
            if cost < best_cost:
                best_cost = cost
                best_assignment = candidate
        assignments.append(best_assignment)

    return assignments


def mean_unlabeled_position_error(scenario: Scenario, result: FilterRunResult) -> float:
    """Return mean best-matching position error, ignoring labels."""

    errors: list[float] = []
    for k, estimates in enumerate(result.estimated_positions):
        estimated_positions = list(estimates.values())
        if len(estimated_positions) < len(scenario.labels):
            continue
        true_positions = [
            _truth_position(scenario, label, k)
            for label in scenario.labels
        ]
        best_cost = inf
        for permuted_estimates in permutations(estimated_positions, len(true_positions)):
            cost = sum(
                float(np.linalg.norm(estimate - truth))
                for estimate, truth in zip(permuted_estimates, true_positions)
            )
            best_cost = min(best_cost, cost / len(true_positions))
        errors.append(best_cost)
    return float(np.mean(errors)) if errors else float("nan")


def _initial_targets(scenario: Scenario) -> tuple[ResolvedTarget, ...]:
    covariance = np.diag([0.25, 0.25, 0.05, 0.05])
    return tuple(
        ResolvedTarget(
            label=target.label,
            mean=target.states[0].copy(),
            covariance=covariance.copy(),
            extent=target.extent.copy(),
            measurement_rate=scenario.config.measurement_rate,
        )
        for target in scenario.targets
    )


def _positions_from_targets(targets: Sequence[ResolvedTarget]) -> dict[Label, np.ndarray]:
    return {target.label: target.mean[:2].copy() for target in targets}


def _truth_position(scenario: Scenario, label: Label, k: int) -> np.ndarray:
    for target in scenario.targets:
        if target.label == label:
            return target.states[k, :2]
    raise KeyError(label)


def _update_target_with_cell(
    target: ResolvedTarget,
    measurements: np.ndarray,
    cell: Sequence[int],
    gain: float,
) -> ResolvedTarget:
    if not cell:
        return target
    centroid = cell_centroid(measurements, cell)
    mean = target.mean.copy()
    residual = centroid - mean[:2]
    mean[:2] = mean[:2] + gain * residual
    return ResolvedTarget(
        label=target.label,
        mean=mean,
        covariance=target.covariance,
        extent=target.extent,
        measurement_rate=max(1.0, 0.8 * target.measurement_rate + 0.2 * len(cell)),
    )


def _update_group_with_cell(
    group: UnresolvedGroup,
    measurements: np.ndarray,
    cell: Sequence[int],
    gain: float,
) -> UnresolvedGroup:
    centroid = cell_centroid(measurements, cell)
    mean = group.mean.copy()
    residual = centroid - mean[:2]
    mean[:2] = mean[:2] + gain * residual
    return UnresolvedGroup(
        member_labels=group.member_labels,
        mean=mean,
        covariance=group.covariance,
        extent=group.extent,
        measurement_rate=max(1.0, 0.8 * group.measurement_rate + 0.2 * len(cell)),
    )


def _collapse_target_to_centroid(
    target: ResolvedTarget,
    centroid: np.ndarray,
    config: FilterConfig,
) -> ResolvedTarget:
    mean = target.mean.copy()
    mean[:2] = centroid
    mean[2:] = config.collapse_velocity_damping * mean[2:]
    return ResolvedTarget(
        label=target.label,
        mean=mean,
        covariance=target.covariance,
        extent=target.extent,
        measurement_rate=target.measurement_rate,
    )


def _assign_cells_to_targets(
    measurements: np.ndarray,
    cells: Sequence[Sequence[int]],
    targets: Sequence[ResolvedTarget],
    config: FilterConfig,
) -> dict[Label, list[int]]:
    candidate_indices: list[int | None] = list(range(len(cells))) + [None]
    best_score = -inf
    best_assignment: dict[Label, list[int]] = {target.label: [] for target in targets}

    for choice in product(candidate_indices, repeat=len(targets)):
        nonempty_choice = [index for index in choice if index is not None]
        if len(set(nonempty_choice)) != len(nonempty_choice):
            continue

        score = 0.0
        assignment: dict[Label, list[int]] = {}
        for target, cell_index in zip(targets, choice):
            if cell_index is None:
                score += config.missed_cell_log_score
                assignment[target.label] = []
                continue
            cell = list(cells[cell_index])
            score += _score_cell_for_target(measurements, cell, target, config)
            assignment[target.label] = cell

        if score > best_score:
            best_score = score
            best_assignment = assignment

    return best_assignment


def _best_group_cell(
    measurements: np.ndarray,
    cells: Sequence[Sequence[int]],
    group: UnresolvedGroup,
    config: FilterConfig,
) -> list[int] | None:
    if not cells:
        return None
    pseudo_target = ResolvedTarget(
        label="group",
        mean=group.mean,
        covariance=group.covariance,
        extent=group.extent,
        measurement_rate=group.measurement_rate,
    )
    return list(
        max(cells, key=lambda cell: _score_cell_for_target(measurements, cell, pseudo_target, config))
    )


def _score_cell_for_target(
    measurements: np.ndarray,
    cell: Sequence[int],
    target: ResolvedTarget,
    config: FilterConfig,
) -> float:
    covariance = target.extent + (config.measurement_noise_scale**2) * np.eye(2)
    return log_cell_likelihood(
        measurements=measurements,
        cell=cell,
        predicted_position=target.mean[:2],
        expected_rate=target.measurement_rate,
        covariance=covariance,
    )


def _targets_from_x_order_cells(
    targets: Sequence[ResolvedTarget],
    measurements: np.ndarray,
    cells: Sequence[Sequence[int]],
) -> tuple[ResolvedTarget, ...]:
    labels = sorted(target.label for target in targets)
    nonempty_cells = [list(cell) for cell in cells if len(cell) > 0]
    if not nonempty_cells:
        return tuple(targets)

    ranked_cells = sorted(nonempty_cells, key=len, reverse=True)[: len(labels)]
    ranked_cells = sorted(ranked_cells, key=lambda cell: float(cell_centroid(measurements, cell)[0]))

    targets_by_label = {target.label: target for target in targets}
    updated_targets: list[ResolvedTarget] = []
    for label, cell in zip(labels, ranked_cells):
        updated_targets.append(
            _update_target_with_cell(targets_by_label[label], measurements, cell, gain=1.0)
        )

    for label in labels[len(ranked_cells) :]:
        updated_targets.append(targets_by_label[label])

    return tuple(updated_targets)
