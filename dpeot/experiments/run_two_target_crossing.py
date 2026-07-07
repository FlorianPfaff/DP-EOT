"""Run the initial two-target merge/split benchmark scaffold.

This script does not claim to implement the final DP-EOT tracker. It verifies
that the scenario generator, partition baselines, and identity metrics are wired
together before implementing the full identity-aware merge/split filter.
"""

from __future__ import annotations

from collections.abc import Callable

from dpeot.metrics.identity import (
    count_identity_switches,
    label_recovery_accuracy,
    split_recovery_delay,
)
from dpeot.partitions.distance_partition import distance_partition
from dpeot.partitions.dp_partition import DPPartitionConfig, dp_partition
from dpeot.partitions.oracle_partition import oracle_partition
from dpeot.scenarios.two_target_merge_split import (
    Scenario,
    ScenarioConfig,
    generate_two_target_merge_split,
)


def main() -> None:
    config = ScenarioConfig()
    scenario = generate_two_target_merge_split(config)

    print("DP-EOT two-target merge/split scaffold")
    print(f"scans: {config.num_steps}")
    print(f"unresolved interval: [{config.merge_start}, {config.merge_end}]")
    print(f"labels: {scenario.labels}")
    print(f"average measurements/scan: {_average_measurements(scenario):.2f}")
    print()

    partitioners: dict[str, Callable] = {
        "distance": lambda scan: distance_partition(scan.measurements, threshold=1.25),
        "dp_proposal": lambda scan: dp_partition(
            scan.measurements, DPPartitionConfig(alpha=1.0, covariance_scale=0.7)
        ),
        "oracle_group": lambda scan: oracle_partition(scan, group_during_unresolved=True),
        "oracle_origin": lambda scan: oracle_partition(scan, group_during_unresolved=False),
    }

    print("partition summary")
    for name, partitioner in partitioners.items():
        summary = _partition_summary(scenario, partitioner)
        print(
            f"  {name:13s} "
            f"cells/scan={summary['cells_per_scan']:.2f} "
            f"cells/unresolved={summary['cells_per_unresolved_scan']:.2f}"
        )

    print()
    print("identity metric smoke test")
    true_labels = scenario.labels
    split_index = config.merge_end + 1
    ideal = _assignment_trace(config.num_steps, true_labels, swap_after_split=False)
    swapped = _assignment_trace(config.num_steps, true_labels, swap_after_split=True, split_index=split_index)

    for name, assignments in {"ideal": ideal, "swapped_after_split": swapped}.items():
        print(
            f"  {name:18s} "
            f"switches={count_identity_switches(assignments)} "
            f"recovery={label_recovery_accuracy(assignments[split_index:], true_labels):.2f} "
            f"split_delay={split_recovery_delay(assignments, true_labels, split_index)}"
        )


def _average_measurements(scenario: Scenario) -> float:
    return sum(scan.measurements.shape[0] for scan in scenario.scans) / len(scenario.scans)


def _partition_summary(scenario: Scenario, partitioner: Callable) -> dict[str, float]:
    cell_counts = []
    unresolved_cell_counts = []
    for scan in scenario.scans:
        cells = partitioner(scan)
        cell_counts.append(len(cells))
        if scan.is_unresolved:
            unresolved_cell_counts.append(len(cells))

    return {
        "cells_per_scan": sum(cell_counts) / len(cell_counts),
        "cells_per_unresolved_scan": sum(unresolved_cell_counts) / len(unresolved_cell_counts),
    }


def _assignment_trace(
    num_steps: int,
    true_labels: tuple[str, str],
    swap_after_split: bool,
    split_index: int = 0,
) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    for k in range(num_steps):
        if swap_after_split and k >= split_index:
            assignments.append({true_labels[0]: true_labels[1], true_labels[1]: true_labels[0]})
        else:
            assignments.append({true_labels[0]: true_labels[0], true_labels[1]: true_labels[1]})
    return assignments


if __name__ == "__main__":
    main()
