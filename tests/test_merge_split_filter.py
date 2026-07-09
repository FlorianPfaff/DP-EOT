from dpeot.metrics.identity import label_recovery_accuracy, split_recovery_delay
from dpeot.partitions.oracle_partition import oracle_partition
from dpeot.scenarios.two_target_merge_split import ScenarioConfig, generate_two_target_merge_split
from dpeot.tracking.merge_split_filter import (
    run_detected_group_filter,
    run_identity_aware_group_filter,
    run_labeled_split_hypothesis_baseline,
    run_x_order_clustering_baseline,
)


def test_identity_aware_group_filter_recovers_labels_after_split() -> None:
    scenario = generate_two_target_merge_split(
        ScenarioConfig(num_steps=41, merge_start=17, merge_end=23, seed=3)
    )
    result = run_identity_aware_group_filter(
        scenario,
        partitioner=lambda scan: oracle_partition(scan, group_during_unresolved=True),
    )
    split_index = scenario.config.merge_end + 1

    assert label_recovery_accuracy(result.assignments[split_index:], scenario.labels) == 1.0
    assert split_recovery_delay(result.assignments, scenario.labels, split_index) == 0


def test_detected_group_filter_recovers_labels_without_oracle_interval() -> None:
    scenario = generate_two_target_merge_split(
        ScenarioConfig(num_steps=41, merge_start=17, merge_end=23, seed=3)
    )
    result = run_detected_group_filter(
        scenario,
        partitioner=lambda scan: oracle_partition(scan, group_during_unresolved=True),
    )
    split_index = scenario.config.merge_end + 1
    unresolved_indices = [i for i, scan in enumerate(scenario.scans) if scan.is_unresolved]

    assert label_recovery_accuracy(result.assignments[split_index:], scenario.labels) == 1.0
    assert split_recovery_delay(result.assignments, scenario.labels, split_index) == 0
    assert any(result.group_membership_trace[i] == frozenset(scenario.labels) for i in unresolved_indices)


def test_labeled_split_hypothesis_baseline_recovers_labels_without_group_membership() -> None:
    scenario = generate_two_target_merge_split(
        ScenarioConfig(num_steps=41, merge_start=17, merge_end=23, seed=3)
    )
    result = run_labeled_split_hypothesis_baseline(
        scenario,
        partitioner=lambda scan: oracle_partition(scan, group_during_unresolved=True),
    )
    split_index = scenario.config.merge_end + 1

    assert label_recovery_accuracy(result.assignments[split_index:], scenario.labels) == 1.0
    assert split_recovery_delay(result.assignments, scenario.labels, split_index) == 0
    assert all(not members for members in result.group_membership_trace)


def test_left_to_right_clustering_baseline_swaps_after_crossing() -> None:
    scenario = generate_two_target_merge_split(
        ScenarioConfig(num_steps=41, merge_start=17, merge_end=23, seed=4)
    )
    result = run_x_order_clustering_baseline(
        scenario,
        partitioner=lambda scan: oracle_partition(scan, group_during_unresolved=False),
    )
    split_index = scenario.config.merge_end + 1

    assert label_recovery_accuracy(result.assignments[split_index:], scenario.labels) == 0.0
