from dpeot.metrics.identity import (
    count_identity_switches,
    count_post_split_identity_switches,
    count_resolved_period_identity_switches,
    group_membership_accuracy,
    label_recovery_accuracy,
    split_recovery_delay,
)


def test_identity_switches_detect_post_split_swap() -> None:
    assignments = [
        {"A": "A", "B": "B"},
        {"A": "A", "B": "B"},
        {"A": "B", "B": "A"},
    ]

    assert count_identity_switches(assignments) == 2
    assert label_recovery_accuracy(assignments[2:], ("A", "B")) == 0.0
    assert split_recovery_delay(assignments, ("A", "B"), split_index=2) is None


def test_identity_switch_count_can_be_windowed() -> None:
    assignments = [
        {"A": "A", "B": "B"},
        {"A": "A", "B": "B"},
        {"A": "B", "B": "A"},
        {"A": "A", "B": "B"},
        {"A": "A", "B": "B"},
    ]

    assert count_identity_switches(assignments) == 4
    assert count_identity_switches(assignments, start=0, end=2) == 0
    assert count_identity_switches(assignments, start=2, end=4) == 2
    assert count_post_split_identity_switches(assignments, split_index=4) == 0


def test_resolved_period_identity_switches_skip_unresolved_boundaries() -> None:
    assignments = [
        {"A": "A", "B": "B"},
        {"A": "A", "B": "B"},
        {"A": "B", "B": "A"},
        {"A": "A", "B": "B"},
        {"A": "A", "B": "B"},
        {"A": "B", "B": "A"},
    ]
    unresolved_mask = [False, False, True, True, False, False]

    assert count_identity_switches(assignments) == 6
    assert count_resolved_period_identity_switches(assignments, unresolved_mask) == 2


def test_group_membership_accuracy_uses_jaccard_score() -> None:
    estimated = [frozenset({"A", "B"}), frozenset({"A"})]
    truth = [frozenset({"A", "B"}), frozenset({"A", "B"})]

    assert group_membership_accuracy(estimated, truth) == 0.75
