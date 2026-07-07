from dpeot.metrics.identity import (
    count_identity_switches,
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


def test_group_membership_accuracy_uses_jaccard_score() -> None:
    estimated = [frozenset({"A", "B"}), frozenset({"A"})]
    truth = [frozenset({"A", "B"}), frozenset({"A", "B"})]

    assert group_membership_accuracy(estimated, truth) == 0.75
